"""
Efficient querying functions for Supabase/pgvector database.
Supports semantic search across financial documents with various filtering options.
"""
import os
import json
from typing import List, Optional
from dataclasses import dataclass

import google.generativeai as genai
from supabase import create_client, Client


@dataclass 
class SearchResult:
    """Represents a search result from the database."""
    content: str
    similarity_score: float
    table_name: str
    row_id: int
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    filing_url: Optional[str] = None
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    word_count: Optional[int] = None


@dataclass
class SearchFilters:
    """Filters for database searches."""
    tables: Optional[List[str]] = None  # Which tables to search
    tickers: Optional[List[str]] = None  # Stock symbols to filter by
    filing_types: Optional[List[str]] = None  # Document types to filter by
    date_range: Optional[tuple] = None  # (start_date, end_date) 
    min_similarity: float = 0.0  # Minimum similarity score
    limit: int = 10  # Maximum results to return


class FinancialDocumentSearcher:
    """
    Semantic search engine for financial documents stored in Supabase with pgvector.
    """
    
    # Available tables and their configurations
    TABLE_CONFIGS = {
        "10-Ks": {
            "content_column": "content",
            "embedding_column": "embedding", 
            "id_column": "id",
            "metadata_columns": ["ticker", "filing_url", "title", "chunk_index", "word_count"]
        },
        "10-Qs": {
            "content_column": "content",
            "embedding_column": "embedding",
            "id_column": "id", 
            "metadata_columns": ["ticker", "filing_url", "title", "chunk_index", "word_count"]
        },
        "earnings_calls": {
            "content_column": "content",
            "embedding_column": "embedding",
            "id_column": "id",
            "metadata_columns": ["ticker", "filing_type", "filing_url", "title", "chunk_index", "word_count"]
        },
        "conference_transcripts": {
            "content_column": "content", 
            "embedding_column": "embedding",
            "id_column": "id",
            "metadata_columns": ["ticker", "filing_type", "filing_url", "title", "chunk_index", "word_count"]
        }
    }
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        gemini_api_key: str,
        model_name: str = "models/text-embedding-004"
    ):
        """
        Initialize the search engine.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            gemini_api_key: Google Gemini API key
            model_name: Gemini embedding model name
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Configure Gemini for query embeddings
        genai.configure(api_key=gemini_api_key)
        self.model_name = model_name
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"  # Different task type for queries
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None
    
    def search_table(
        self, 
        table_name: str, 
        query_embedding: List[float],
        filters: SearchFilters
    ) -> List[SearchResult]:
        """
        Search a specific table using vector similarity.
        
        Args:
            table_name: Name of table to search
            query_embedding: Query embedding vector
            filters: Search filters to apply
            
        Returns:
            List of search results
        """
        if table_name not in self.TABLE_CONFIGS:
            print(f"Unknown table: {table_name}")
            return []
            
        config = self.TABLE_CONFIGS[table_name]
        
        try:
            # Build select clause with all relevant columns
            select_columns = [
                config["id_column"],
                config["content_column"],
                config["embedding_column"]
            ] + config["metadata_columns"]
            
            # Start query
            query = self.supabase.table(table_name).select(
                ", ".join(select_columns)
            )
            
            # Apply filters
            if filters.tickers and "ticker" in config["metadata_columns"]:
                query = query.in_("ticker", filters.tickers)
            
            if filters.filing_types and "filing_type" in config["metadata_columns"]:
                query = query.in_("filing_type", filters.filing_types)
            
            # Exclude chunk markers from results
            query = query.not_.like(config["content_column"], "[CHUNK%")
            
            # Execute query
            response = query.execute()
            
            if not response.data:
                return []
            
            # Calculate similarities and filter results
            results = []
            for i, row in enumerate(response.data):
                embedding_data = row.get(config["embedding_column"])
                if not embedding_data:
                    continue
                
                # Parse embedding if it's stored as string
                if isinstance(embedding_data, str):
                    try:
                        embedding_data = json.loads(embedding_data)
                    except:
                        continue
                
                # Calculate cosine similarity
                similarity = FinancialDocumentSearcher._cosine_similarity(
                    query_embedding, 
                    embedding_data
                )
                
                if similarity >= filters.min_similarity:
                    result = SearchResult(
                        content=row[config["content_column"]],
                        similarity_score=similarity,
                        table_name=table_name,
                        row_id=row[config["id_column"]],
                        ticker=row.get("ticker"),
                        filing_type=row.get("filing_type"),
                        filing_url=row.get("filing_url"),
                        title=row.get("title"),
                        chunk_index=row.get("chunk_index"),
                        word_count=row.get("word_count")
                    )
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results[:filters.limit]
            
        except Exception as e:
            print(f"Error searching table {table_name}: {e}")
            return []
    
    def search(
        self, 
        query: str, 
        filters: Optional[SearchFilters] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search across financial documents using semantic similarity.
        
        Args:
            query: Natural language search query
            filters: Optional search filters
            
        Returns:
            List of search results sorted by relevance
        """
        if filters is None:
            filters = SearchFilters()
        
        # Handle direct keyword arguments
        if 'limit' in kwargs:
            filters.limit = kwargs['limit']
        if 'min_similarity' in kwargs:
            filters.min_similarity = kwargs['min_similarity']
        
        # Generate embedding for query
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            print("Failed to generate query embedding")
            return []
        
        # Determine which tables to search
        tables_to_search = filters.tables or list(self.TABLE_CONFIGS.keys())
        
        # Search each table
        all_results = []
        for table_name in tables_to_search:
            table_results = self.search_table(table_name, query_embedding, filters)
            all_results.extend(table_results)
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return all_results[:filters.limit]
    
    def search_by_ticker(
        self, 
        query: str, 
        ticker: str,
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents related to a specific company ticker.
        
        Args:
            query: Search query
            ticker: Stock symbol (e.g., "AAPL", "GTLB")
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            
        Returns:
            Search results for the specified ticker
        """
        filters = SearchFilters(
            tickers=[ticker.upper()],
            limit=limit,
            min_similarity=min_similarity
        )
        return self.search(query, filters)
    
    def search_by_document_type(
        self,
        query: str,
        doc_type: str,
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search within specific document types.
        
        Args:
            query: Search query
            doc_type: Document type ("10-K", "10-Q", "earnings_call", etc.)
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            
        Returns:
            Search results for the specified document type
        """
        # Map doc types to table names
        table_mapping = {
            "10-K": ["10-Ks"],
            "10-Q": ["10-Qs"], 
            "earnings_call": ["earnings_calls"],
            "conference": ["conference_transcripts"]
        }
        
        tables = table_mapping.get(doc_type, [doc_type])
        
        filters = SearchFilters(
            tables=tables,
            limit=limit,
            min_similarity=min_similarity
        )
        return self.search(query, filters)
    
    def get_similar_documents(
        self,
        reference_content: str,
        exclude_tables: Optional[List[str]] = None,
        limit: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Find documents similar to a reference document.
        
        Args:
            reference_content: Content to find similar documents for
            exclude_tables: Tables to exclude from search
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            
        Returns:
            Similar documents
        """
        tables_to_search = [
            table for table in self.TABLE_CONFIGS.keys()
            if not exclude_tables or table not in exclude_tables
        ]
        
        filters = SearchFilters(
            tables=tables_to_search,
            limit=limit,
            min_similarity=min_similarity
        )
        
        return self.search(reference_content, filters)
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if not isinstance(vec1, list) or not isinstance(vec2, list):
                return 0.0
            
            if len(vec1) != len(vec2):
                return 0.0
            
            if all(x == 0 for x in vec1) or all(x == 0 for x in vec2):
                return 0.0
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except:
            return 0.0
    
    def print_results(self, results: List[SearchResult], max_content_length: int = 200):
        """
        Pretty print search results.
        
        Args:
            results: Search results to print
            max_content_length: Maximum content length to display
        """
        if not results:
            print("No results found.")
            return
        
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.table_name}] Similarity: {result.similarity_score:.3f}")
            
            if result.ticker:
                print(f"   Ticker: {result.ticker}")
            if result.title:
                print(f"   Title: {result.title}")
            if result.filing_type:
                print(f"   Type: {result.filing_type}")
            
            # Truncate content for display
            content = result.content
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            print(f"   Content: {content}")
            print()


def create_searcher_from_env() -> FinancialDocumentSearcher:
    """
    Create searcher instance using environment variables.
    
    Returns:
        Configured searcher instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY') 
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    missing_vars = []
    if not supabase_url:
        missing_vars.append('SUPABASE_URL')
    if not supabase_key:
        missing_vars.append('SUPABASE_ANON_KEY')
    if not gemini_api_key:
        missing_vars.append('GEMINI_API_KEY')
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return FinancialDocumentSearcher(supabase_url, supabase_key, gemini_api_key)


# Convenience functions for common queries
def search_financial_documents(query: str, **kwargs) -> List[SearchResult]:
    """Quick search across all financial documents."""
    searcher = create_searcher_from_env()
    return searcher.search(query, **kwargs)


def search_by_company(query: str, ticker: str, **kwargs) -> List[SearchResult]:
    """Quick search for a specific company."""
    searcher = create_searcher_from_env()
    return searcher.search_by_ticker(query, ticker, **kwargs)


def search_earnings_calls(query: str, **kwargs) -> List[SearchResult]:
    """Quick search within earnings calls only."""
    searcher = create_searcher_from_env()
    return searcher.search_by_document_type(query, "earnings_call", **kwargs)


def search_10k_filings(query: str, **kwargs) -> List[SearchResult]:
    """Quick search within 10-K filings only.""" 
    searcher = create_searcher_from_env()
    return searcher.search_by_document_type(query, "10-K", **kwargs)


if __name__ == "__main__":
    # Example usage
    try:
        searcher = create_searcher_from_env()
        
        # Test basic search
        results = searcher.search("duo artificial intelligence usage", 
                                 SearchFilters(limit=5, min_similarity=0.0))
        searcher.print_results(results)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Search error: {e}")