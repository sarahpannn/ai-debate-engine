from typing import List
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    content: str
    chunk_index: int
    filing_type: str
    filing_url: str
    ticker: str
    word_count: int
    title: str


class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks of specified word count."""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            if end == len(words):
                break
                
            start = end - self.overlap
        
        return chunks