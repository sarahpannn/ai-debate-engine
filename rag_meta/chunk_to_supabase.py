"""
Supabase loader for SEC filing data with text chunking capabilities.
"""
import os
import re
from typing import List

from supabase import create_client, Client
import sec_parser as sp
from secedgar import FilingType as FT

from grab_sec import parse_html_from_url, get_custom_parser_steps, get_filing_urls
from get_htm_urls import get_primary_htm_url_from_txt
from chunking import DocumentChunk, TextChunker

class SupabaseRAGLoader:
    """
    Loader for processing SEC filings and storing chunked text in Supabase.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the Supabase RAG loader.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anonymous key
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.chunker = TextChunker()
    
    def process_filing(self, filing_url: str, ticker: str, filing_type: str = "10-K") -> List[DocumentChunk]:
        """Process a single SEC filing and return chunks."""
        try:
            # Get HTML URL from text filing URL
            htm_url = get_primary_htm_url_from_txt(filing_url)
            if not htm_url:
                print(f"No HTML found for {filing_url}")
                return []
            
            # Parse the filing
            parser = sp.Edgar10QParser(get_steps=get_custom_parser_steps)
            tree = parse_html_from_url(htm_url, parser)
            
            if not tree:
                print(f"Failed to parse {htm_url}")
                return []
            
            # Extract text (tree is already rendered as string from sp.render)
            text = str(tree)
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
            # Extract date information from the text
            
            # Look for date patterns in the text
            date_patterns = [
                r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD format
                r'(\d{4})(\d{2})(\d{2})',    # YYYYMMDD format
            ]
            
            year = None
            quarter = None
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Get the most recent year from the matches
                    years = [int(match[0]) for match in matches if int(match[0]) >= 2020]
                    if years:
                        year = max(years)
                        # For 10-Q, try to determine quarter based on months found
                        if filing_type == "10-Q":
                            months = [int(match[1]) for match in matches if int(match[0]) == year]
                            if months:
                                max_month = max(months)
                                if max_month <= 3:
                                    quarter = "Q1"
                                elif max_month <= 6:
                                    quarter = "Q2"
                                elif max_month <= 9:
                                    quarter = "Q3"
                                else:
                                    quarter = "Q4"
                        break
            
            # Fallback if no date found
            if not year:
                year = 2024
            
            # Create title
            if filing_type == "10-Q" and quarter:
                title = f"{year} {quarter} {filing_type}"
            else:
                title = f"{year} {filing_type}"
            
            # Chunk the text
            chunks = self.chunker.chunk_text(text)
            
            # Create DocumentChunk objects with title prefixed to content
            document_chunks = []
            for i, chunk in enumerate(chunks):
                # Prefix the title to the content
                content_with_title = f"{title}: {chunk}"
                
                doc_chunk = DocumentChunk(
                    content=content_with_title,
                    chunk_index=i,
                    filing_type=filing_type,
                    filing_url=filing_url,
                    ticker=ticker,
                    word_count=len(content_with_title.split()),
                    title=title
                )
                document_chunks.append(doc_chunk)
            
            return document_chunks
            
        except Exception as e:
            print(f"Error processing {filing_url}: {e}")
            return []
    
    def load_chunks_to_supabase(self, chunks: List[DocumentChunk], filing_type: str) -> bool:
        """Load document chunks into Supabase."""
        try:
            chunk_data = []
            for chunk in chunks:
                chunk_dict = {
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'filing_type': chunk.filing_type,
                    'filing_url': chunk.filing_url,
                    'ticker': chunk.ticker,
                    'word_count': chunk.word_count,
                    'title': chunk.title
                }
                chunk_data.append(chunk_dict)
            
            # Determine table name based on filing type
            if filing_type == "10-K":
                table_name = '10-Ks'
            elif filing_type == "10-Q":
                table_name = '10-Qs'
            else:
                table_name = 'document_chunks_other'
            
            # Insert into Supabase
            result = self.supabase.table(table_name).insert(chunk_data).execute()
            
            if result.data:
                print(f"Successfully loaded {len(chunk_data)} chunks")
                return True
            else:
                print("Failed to load chunks")
                return False
                
        except Exception as e:
            print(f"Error loading chunks to Supabase: {e}")
            return False
    
    def process_all_filings(self, ticker: str = 'gtlb', filing_type: FT = FT.FILING_10K):
        """Process all filings of a given type for a ticker."""
        try:
            # get_filing_urls returns dict with ticker as key
            all_urls = get_filing_urls(ticker=ticker, filing_type=filing_type)
            
            if not all_urls or ticker.lower() not in all_urls:
                print(f"No URLs found for ticker {ticker}")
                return
            
            filing_urls = all_urls[ticker.lower()]
            filing_name = filing_type.value if hasattr(filing_type, 'value') else str(filing_type)
            print(f"Found {len(filing_urls)} {filing_name} filings for {ticker}")
            
            total_chunks = 0
            for url in filing_urls:
                print(f"Processing {url}")
                chunks = self.process_filing(url, ticker, filing_name)
                
                if chunks:
                    success = self.load_chunks_to_supabase(chunks, filing_type=filing_name)
                    if success:
                        total_chunks += len(chunks)
                        print(f"Loaded {len(chunks)} chunks from {url}")
                    else:
                        print(f"Failed to load chunks from {url}")
                else:
                    print(f"No chunks generated from {url}")
            
            print(f"Total chunks loaded: {total_chunks}")
            
        except Exception as e:
            print(f"Error in process_all_filings: {e}")
    
    def process_all_10k_filings(self, ticker: str = 'gtlb'):
        self.process_all_filings(ticker, FT.FILING_10K)
    
    def process_all_10q_filings(self, ticker: str = 'gtlb'):
        self.process_all_filings(ticker, FT.FILING_10Q)

def main():
    """
    Main function to process SEC filings and load them into Supabase.
    
    Requires environment variables:
        SUPABASE_URL: Supabase project URL
        SUPABASE_ANON_KEY: Supabase anonymous key
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("Error: Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
        return
    
    # Default ticker for processing
    ticker = 'gtlb'
    
    print(f"Processing SEC filings for ticker: {ticker}")
    
    loader = SupabaseRAGLoader(supabase_url, supabase_key)
    
    print("Processing 10-K filings...")
    loader.process_all_10k_filings(ticker)
    
    print("Processing 10-Q filings...")
    loader.process_all_10q_filings(ticker)
    
    print("Completed processing all filings.")

if __name__ == "__main__":
    main()