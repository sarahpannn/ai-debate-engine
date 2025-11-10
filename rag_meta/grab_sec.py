"""
SEC filing downloader and parser for extracting text content from various filing types.
"""
import os
from typing import List, Dict, Optional
from datetime import date

import requests
from tqdm import tqdm
from secedgar import filings, FilingType as FT
import sec_parser as sp
from sec_parser.processing_steps import (
    TopSectionManagerFor10Q, 
    IndividualSemanticElementExtractor, 
    TopSectionTitleCheck
)

from get_htm_urls import get_primary_htm_url_from_txt


# Configuration constants
DEFAULT_TICKER = 'GTLB'
DEFAULT_FILING_TYPE = FT.FILING_10K
DEFAULT_START_DATE = date(2023, 1, 1)
DEFAULT_USER_AGENT = "sarahpan@mit.edu"

def get_filing_urls(
    ticker: str = DEFAULT_TICKER, 
    filing_type: FT = DEFAULT_FILING_TYPE,
    user_agent: str = DEFAULT_USER_AGENT
) -> List[str]:
    """
    Retrieve SEC filing URLs for a given ticker and filing type.
    
    Args:
        ticker: Stock ticker symbol
        filing_type: Type of SEC filing to retrieve
        user_agent: User agent for SEC requests
        
    Returns:
        List of filing URLs
    """
    results = filings(
        cik_lookup=ticker,
        filing_type=filing_type,
        user_agent=user_agent
    )
    
    return results.get_urls()

def get_custom_parser_steps() -> List:
    """
    Get custom parsing steps that exclude 10-Q specific components.
    
    Returns:
        List of parsing steps without 10-Q related components
    """
    all_steps = sp.Edgar10QParser().get_default_steps()
    
    steps_without_top_section_manager = [
        step for step in all_steps 
        if not isinstance(step, TopSectionManagerFor10Q)
    ]

    def get_checks_without_top_section_title_check():
        all_checks = sp.Edgar10QParser().get_default_single_element_checks()
        return [
            check for check in all_checks 
            if not isinstance(check, TopSectionTitleCheck)
        ]
    
    return [
        IndividualSemanticElementExtractor(get_checks=get_checks_without_top_section_title_check) 
        if isinstance(step, IndividualSemanticElementExtractor) 
        else step
        for step in steps_without_top_section_manager
    ]


def preview_text(text: str, num_lines: int = 5) -> None:
    """
    Print the first n lines of text for preview purposes.
    
    Args:
        text: Text to preview
        num_lines: Number of lines to display
    """
    lines = text.split("\n")[:num_lines]
    print("\n".join(lines) + "\n...")


def extract_text_from_elements(elements) -> str:
    """
    Extract plain text from parsed SEC elements.
    
    Args:
        elements: List of parsed SEC elements
        
    Returns:
        Concatenated text from all elements
    """
    text_parts = []
    
    for element in elements:
        if hasattr(element, 'text') and element.text:
            text_parts.append(element.text.strip())
    
    return ' '.join(text_parts)


def parse_html_from_url(
    url: str, 
    parser: sp.Edgar10QParser,
    user_agent: str = DEFAULT_USER_AGENT
) -> str:
    """
    Download and parse HTML content from a URL.
    
    Args:
        url: URL to fetch HTML from
        parser: SEC parser instance
        user_agent: User agent for HTTP requests
        
    Returns:
        Extracted text content or empty string on error
    """
    try: 
        headers = {"User-Agent": user_agent}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""
    
    elements = parser.parse(html)
    return extract_text_from_elements(elements)

def main():
    """
    Main function to demonstrate SEC filing processing.
    """
    # Process only 10-K filings for demo
    filing_types_to_process = [FT.FILING_10K]
    
    all_urls = {}
    for filing_type in filing_types_to_process:
        ticker_urls = get_filing_urls(filing_type=filing_type)
        all_urls[filing_type] = ticker_urls.get(DEFAULT_TICKER.lower(), [])
    
    print(f"Found URLs for filing types: {list(all_urls.keys())}")
    
    # Initialize parser with custom steps
    parser = sp.Edgar10QParser(get_steps=get_custom_parser_steps)
    
    total_filings = 0
    no_html_count = 0
    
    # Process filings
    for filing_type, urls in all_urls.items():
        print(f"\nProcessing {filing_type} filings...")
        
        for url in urls[:1]:  # Process only first URL for demo
            total_filings += 1
            
            # Get HTML URL from the text filing URL
            html_url = get_primary_htm_url_from_txt(url)
            
            if html_url is None:
                no_html_count += 1
                print(f"No HTML found for {url}")
                continue
            
            # Parse the HTML content
            extracted_text = parse_html_from_url(html_url, parser)
            
            if extracted_text:
                preview_text(extracted_text)
            else:
                print("No text extracted from filing")
            
            break  # Process only one filing for demo
        break  # Process only one filing type for demo
    
    print(f"\nSummary: Total filings: {total_filings}, No HTML found: {no_html_count}")


if __name__ == "__main__":
    main()
