"""
Generate embeddings for text content in Supabase tables using Gemini embeddings.
"""
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from supabase import create_client, Client


@dataclass
class TableConfig:
    """Configuration for a Supabase table."""
    name: str
    content_column: str = "content"
    embedding_column: str = "embedding"
    id_column: str = "id"


class GeminiEmbeddingGenerator:
    """
    Generate embeddings using Google's Gemini model for Supabase table content.
    """
    
    def __init__(
        self, 
        supabase_url: str, 
        supabase_key: str, 
        gemini_api_key: str,
        model_name: str = "models/text-embedding-004"
    ):
        """
        Initialize the embedding generator.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key (not anon key)
            gemini_api_key: Google Gemini API key
            model_name: Gemini embedding model name
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model_name = model_name
        
        # Rate limiting
        self.requests_per_minute = 60
        self.request_delay = 60 / self.requests_per_minute
        
    def chunk_text(self, text: str, max_bytes: int = 30000) -> List[str]:
        """
        Split text into chunks that fit within the API payload limit.
        
        Args:
            text: Text to chunk
            max_bytes: Maximum bytes per chunk (30KB to be safe with 36KB limit)
            
        Returns:
            List of text chunks
        """
        # Estimate bytes (rough approximation)
        if len(text.encode('utf-8')) <= max_bytes:
            return [text]
        
        # Split by paragraphs first, then sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            if len(test_chunk.encode('utf-8')) <= max_bytes:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Single paragraph is too large, split by sentences
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    for sentence in sentences:
                        test_sentence = temp_chunk + (". " if temp_chunk else "") + sentence
                        if len(test_sentence.encode('utf-8')) <= max_bytes:
                            temp_chunk = test_sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                    if temp_chunk:
                        chunks.append(temp_chunk)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema to understand required columns."""
        try:
            # Get a sample row to understand the structure
            response = self.supabase.table(table_name).select("*").limit(1).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"Error getting table schema: {e}")
            return {}

    def create_chunk_row(self, table_config: TableConfig, original_id: int, chunk_text: str, chunk_index: int) -> Optional[int]:
        """
        Create a new row for a text chunk.
        
        Args:
            table_config: Table configuration
            original_id: ID of original row
            chunk_text: Text content of the chunk
            chunk_index: Index of this chunk (0, 1, 2, etc.)
            
        Returns:
            ID of created row or None if failed
        """
        try:
            # Get original row to copy required fields
            original_response = self.supabase.table(table_config.name).select("*").eq(
                table_config.id_column, original_id
            ).execute()
            
            if not original_response.data:
                print(f"Could not find original row {original_id}")
                return None
                
            original_row = original_response.data[0]
            
            # Create new row data, copying from original but with chunked content
            new_row_data = original_row.copy()
            
            # Remove the ID field so a new one gets generated
            if table_config.id_column in new_row_data:
                del new_row_data[table_config.id_column]
            
            # Update content with chunk
            chunk_title = f"[CHUNK {chunk_index} OF {original_id}] "
            new_row_data[table_config.content_column] = chunk_title + chunk_text
            
            # Modify filing_url to make it unique for chunks
            if 'filing_url' in new_row_data:
                original_url = new_row_data['filing_url']
                new_row_data['filing_url'] = f"{original_url}#chunk{chunk_index}"
            
            # Clear embedding field
            if table_config.embedding_column in new_row_data:
                new_row_data[table_config.embedding_column] = None
            
            # Insert new row
            response = self.supabase.table(table_config.name).insert(new_row_data).execute()
            
            if response.data:
                return response.data[0][table_config.id_column]
            return None
            
        except Exception as e:
            print(f"Error creating chunk row: {e}")
            return None
    
    def get_rows_without_embeddings(self, table_config: TableConfig, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch rows that don't have embeddings yet, excluding chunk rows.
        
        Args:
            table_config: Table configuration
            limit: Maximum number of rows to fetch
            
        Returns:
            List of row dictionaries
        """
        try:
            response = self.supabase.table(table_config.name).select(
                f"{table_config.id_column}, {table_config.content_column}"
            ).is_(
                table_config.embedding_column, "null"
            ).not_.like(
                table_config.content_column, "[CHUNK%"
            ).limit(limit).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"Error fetching rows from {table_config.name}: {e}")
            return []
    
    def update_embedding(self, table_config: TableConfig, row_id: int, embedding: List[float]) -> bool:
        """
        Update a row with its embedding.
        
        Args:
            table_config: Table configuration
            row_id: Row ID to update
            embedding: Embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.supabase.table(table_config.name).update({
                table_config.embedding_column: embedding
            }).eq(table_config.id_column, row_id).execute()
            
            return bool(response.data)
            
        except Exception as e:
            print(f"Error updating embedding for row {row_id}: {e}")
            return False
    
    def process_table(self, table_config: TableConfig, batch_size: int = 50) -> None:
        """
        Process all rows in a table to generate embeddings.
        
        Args:
            table_config: Table configuration
            batch_size: Number of rows to process in each batch
        """
        print(f"\nProcessing table: {table_config.name}")
        
        total_processed = 0
        total_successful = 0
        
        while True:
            # Fetch batch of rows without embeddings
            rows = self.get_rows_without_embeddings(table_config, batch_size)
            
            if not rows:
                print(f"No more rows to process in {table_config.name}")
                break
            
            print(f"Processing batch of {len(rows)} rows...")
            
            batch_successful = 0
            for row in rows:
                row_id = row[table_config.id_column]
                content = row[table_config.content_column]
                
                if not content or not content.strip():
                    print(f"Skipping row {row_id}: empty content")
                    continue
                
                # Check if content needs chunking
                chunks = self.chunk_text(content)
                
                if len(chunks) == 1:
                    # Single chunk, process normally
                    embedding = self.generate_embedding(content)
                    
                    if embedding:
                        success = self.update_embedding(table_config, row_id, embedding)
                        if success:
                            batch_successful += 1
                            print(f"✓ Updated row {row_id}")
                        else:
                            print(f"✗ Failed to update row {row_id}")
                    else:
                        print(f"✗ Failed to generate embedding for row {row_id}")
                else:
                    # Multiple chunks, create separate rows
                    print(f"Text too large for row {row_id}, creating {len(chunks)} chunk rows")
                    chunks_successful = 0
                    
                    for chunk_index, chunk in enumerate(chunks):
                        # Create new row for chunk
                        chunk_row_id = self.create_chunk_row(table_config, row_id, chunk, chunk_index)
                        
                        if chunk_row_id:
                            # Generate embedding for chunk
                            embedding = self.generate_embedding(chunk)
                            
                            if embedding:
                                success = self.update_embedding(table_config, chunk_row_id, embedding)
                                if success:
                                    chunks_successful += 1
                                    print(f"✓ Created and embedded chunk {chunk_index} as row {chunk_row_id}")
                                else:
                                    print(f"✗ Failed to update chunk {chunk_index} (row {chunk_row_id})")
                            else:
                                print(f"✗ Failed to generate embedding for chunk {chunk_index}")
                        else:
                            print(f"✗ Failed to create chunk row {chunk_index}")
                        
                        time.sleep(self.request_delay)
                    
                    # Delete original row since we've created chunks
                    if chunks_successful > 0:
                        try:
                            self.supabase.table(table_config.name).delete().eq(
                                table_config.id_column, row_id
                            ).execute()
                            batch_successful += 1
                            print(f"✓ Original row {row_id} deleted after creating {chunks_successful}/{len(chunks)} chunks")
                        except Exception as e:
                            print(f"✗ Failed to delete original row {row_id}: {e}")
                    else:
                        print(f"✗ No chunks successful for row {row_id}")
                
                # Rate limiting
                time.sleep(self.request_delay)
            
            total_processed += len(rows)
            total_successful += batch_successful
            
            print(f"Batch complete: {batch_successful}/{len(rows)} successful")
            print(f"Total progress: {total_successful}/{total_processed} successful")
        
        print(f"Table {table_config.name} complete: {total_successful}/{total_processed} successful")
    
    def process_all_tables(self, table_configs: List[TableConfig]) -> None:
        """
        Process all specified tables.
        
        Args:
            table_configs: List of table configurations
        """
        print(f"Starting embedding generation for {len(table_configs)} tables")
        
        for table_config in table_configs:
            try:
                self.process_table(table_config)
            except Exception as e:
                print(f"Error processing table {table_config.name}: {e}")
                continue
        
        print("All tables processed!")


def get_table_configs() -> List[TableConfig]:
    """
    Get configuration for all tables that need embeddings.
    
    Returns:
        List of table configurations
    """
    return [
        TableConfig("10-Ks"),
        TableConfig("10-Qs"), 
        TableConfig("conference_transcripts"),
        TableConfig("earnings_calls")
    ]


def main():
    """
    Main function to generate embeddings for all tables.
    
    Requires environment variables:
        SUPABASE_URL: Supabase project URL
        SUPABASE_SERVICE_KEY: Supabase service role key (not anon key)
        GEMINI_API_KEY: Google Gemini API key
    """
    # Get environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')  # Use service key for table updates
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    # Validate required environment variables
    missing_vars = []
    if not supabase_url:
        missing_vars.append('SUPABASE_URL')
    if not supabase_key:
        missing_vars.append('SUPABASE_ANON_KEY')
    if not gemini_api_key:
        missing_vars.append('GEMINI_API_KEY')
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("  SUPABASE_URL - Your Supabase project URL")
        print("  SUPABASE_ANON_KEY - Your Supabase service role key")
        print("  GEMINI_API_KEY - Your Google Gemini API key")
        return
    
    # Initialize embedding generator
    try:
        generator = GeminiEmbeddingGenerator(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            gemini_api_key=gemini_api_key
        )
    except Exception as e:
        print(f"Error initializing embedding generator: {e}")
        return
    
    # Get table configurations
    table_configs = get_table_configs()
    
    print(f"Found {len(table_configs)} tables to process:")
    for config in table_configs:
        print(f"  - {config.name}")
    
    # Process all tables
    generator.process_all_tables(table_configs)


if __name__ == "__main__":
    main()