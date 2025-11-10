"""
SEC filing parser for extracting and processing SEC documents.
"""
import warnings
from typing import List, Optional
from datetime import date

from secedgar import filings, FilingType as FT
import sec_parser as sp


def parse_sec_filings(
    ticker: str = 'GTLB',
    filing_type: FT = FT.FILING_10K,
    start_date: date = date(2023, 1, 1),
    end_date: date = date.today(),
    user_agent: str = "sarahpan@mit.edu"
) -> List[str]:
    """
    Parse SEC filings for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        filing_type: Type of SEC filing to retrieve
        start_date: Start date for filing search
        end_date: End date for filing search
        user_agent: User agent for SEC requests
        
    Returns:
        List of URLs for the filings
    """
    filing_downloader = filings(
        cik_lookup=ticker,
        filing_type=filing_type,
        user_agent=user_agent
    )
    
    return filing_downloader.get_urls()


def parse_filing_html(html_content: str) -> List:
    """
    Parse HTML content from SEC filing.
    
    Args:
        html_content: HTML content of the filing
        
    Returns:
        List of parsed elements
    """
    parser = sp.Edgar10QParser()
    return parser.parse(html_content)


def main():
    """Main function to demonstrate SEC filing parsing."""
    urls = parse_sec_filings()
    print(f"Found {len(urls)} filings")
    
    # Example: convert txt URL to htm URL
    for url in urls[:1]:  # Process only first URL as example
        if url.endswith('.txt'):
            html_url = url[:-3] + 'htm'
            print(f"HTML URL: {html_url}")


if __name__ == "__main__":
    main()
