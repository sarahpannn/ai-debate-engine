import os
import re
from pathlib import Path
from typing import List

import fitz
from supabase import create_client

from chunking import DocumentChunk, TextChunker


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_pages.append(text)
    
    doc.close()
    return "\n\n".join(text_pages)


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_metadata(text: str) -> dict:
    """Extract basic metadata from text."""
    # Extract date (various formats)
    date_match = re.search(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        text, re.IGNORECASE
    )
    
    # Extract quarter info
    quarter_match = re.search(
        r'\b(Q[1-4]|First|Second|Third|Fourth)\s+(Quarter|quarter)\b',
        text, re.IGNORECASE
    )
    
    # Extract year from date or text
    year = None
    if date_match:
        year_match = re.search(r'\d{4}', date_match.group(0))
        if year_match:
            year = int(year_match.group(0))
    else:
        # Look for year patterns in text
        year_matches = re.findall(r'\b(20[2-3]\d)\b', text)
        if year_matches:
            year = int(max(year_matches))
    
    return {
        "date": date_match.group(0) if date_match else None,
        "quarter": quarter_match.group(0) if quarter_match else None,
        "year": year,
    }


class PDFProcessor:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.chunker = TextChunker()
        self.supabase = create_client(supabase_url, supabase_key)
    
    def process_to_chunks(self, pdf_path: str, doc_type: str, ticker: str = "GTLB") -> List[DocumentChunk]:
        """Process PDF into document chunks."""
        # Extract and clean text
        raw_text = extract_text_from_pdf(pdf_path)
        clean_content = clean_text(raw_text)
        
        # Extract metadata
        metadata = extract_metadata(clean_content)
        
        # Create title based on document type
        year = metadata.get("year", 2024)
        quarter = metadata.get("quarter")
        
        if doc_type == "earnings_call":
            if quarter:
                title = f"{year} {quarter} Earnings Call"
            else:
                title = f"{year} Earnings Call"
            filing_type = "earnings"
        elif doc_type == "conference_transcript":
            if quarter:
                title = f"{year} {quarter} Conference Transcript"
            else:
                title = f"{year} Conference Transcript"
            filing_type = "conf"
        else:
            # Generic document
            if quarter:
                title = f"{year} {quarter} {doc_type.title()}"
            else:
                title = f"{year} {doc_type.title()}"
            filing_type = doc_type.title()
        
        # Chunk the text
        chunks = self.chunker.chunk_text(clean_content)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk in enumerate(chunks):
            content_with_title = f"{title}: {chunk}"
            
            doc_chunk = DocumentChunk(
                content=content_with_title,
                chunk_index=i,
                filing_type=filing_type,
                filing_url=str(pdf_path),
                ticker=ticker.upper(),
                word_count=len(content_with_title.split()),
                title=title
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def load_chunks_to_supabase(self, chunks: List[DocumentChunk], table_name: str) -> bool:
        """Load document chunks into specified Supabase table."""
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
            
            # Insert into specified Supabase table
            result = self.supabase.table(table_name).insert(chunk_data).execute()
            
            if result.data:
                print(f"✓ Loaded {len(chunk_data)} chunks to {table_name}")
                return True
            else:
                print(f"✗ Failed to load chunks to {table_name}")
                return False
                
        except Exception as e:
            print(f"✗ Error loading chunks to {table_name}: {e}")
            return False


def process_pdfs_to_supabase(
    input_dir: str, 
    doc_type: str, 
    table_name: str, 
    ticker: str = "GTLB"
) -> None:
    """Process all PDFs in directory and load chunks to Supabase."""
    # Get Supabase credentials from environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("Error: Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
        return
    
    processor = PDFProcessor(supabase_url, supabase_key)
    input_path = Path(input_dir)
    
    total_chunks = 0
    processed_files = 0
    
    print(f"Processing {doc_type}s from {input_dir} to {table_name} table...")
    
    for pdf_file in input_path.glob("*.pdf"):
        try:
            print(f"Processing {pdf_file.name}...")
            chunks = processor.process_to_chunks(str(pdf_file), doc_type, ticker)
            
            if chunks:
                success = processor.load_chunks_to_supabase(chunks, table_name)
                if success:
                    total_chunks += len(chunks)
                    processed_files += 1
                else:
                    print(f"✗ Failed to load chunks from {pdf_file.name}")
            else:
                print(f"✗ No chunks generated from {pdf_file.name}")
                
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")
    
    print(f"\nSummary:")
    print(f"Files processed: {processed_files}")
    print(f"Total chunks loaded to {table_name}: {total_chunks}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs and upload to Supabase")
    parser.add_argument("input", help="Path to directory containing PDF files")
    parser.add_argument("--type", required=True, 
                       choices=["earnings_call", "conference_transcript"], 
                       help="Type of document")
    parser.add_argument("--table", required=True, 
                       help="Supabase table name (e.g., earnings_calls, conference_transcripts)")
    parser.add_argument("--ticker", default="GTLB", 
                       help="Stock ticker symbol (default: GTLB)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.is_dir():
        print("Error: Input must be a directory containing PDF files")
        return
    
    process_pdfs_to_supabase(str(input_path), args.type, args.table, args.ticker)


if __name__ == "__main__":
    main()