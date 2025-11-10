"""
Utilities for extracting HTML URLs from SEC EDGAR submission text files.
"""
import re
import requests
from typing import Optional, Sequence

# Default user agent for SEC requests
DEFAULT_USER_AGENT = {"User-Agent": "sarahpan@mit.edu"}

# Regex patterns for parsing EDGAR documents
DOCUMENT_PATTERN = re.compile(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>")


def _create_tag_pattern(tag: str) -> re.Pattern:
    """Create regex pattern for extracting tag content from EDGAR documents."""
    return re.compile(fr"(?im)<{tag}>\s*(.+?)\s*(?=<)", re.S)


def _get_submission_directory(txt_url: str) -> str:
    """Extract the directory path from a submission text URL."""
    return txt_url.rsplit("/", 1)[0] + "/"

def get_primary_htm_url_from_txt(
    txt_url: str,
    preferred_types: Sequence[str] = ("10-K", "10-Q", "8-K"),
    user_agent: dict = None
) -> Optional[str]:
    """
    Extract the primary HTML URL from an EDGAR submission text file.
    
    Args:
        txt_url: URL to the EDGAR submission text file
        preferred_types: Sequence of preferred filing types to prioritize
        user_agent: Custom user agent headers (defaults to DEFAULT_USER_AGENT)
        
    Returns:
        URL to the primary HTML document, or None if not found
        
    Strategy:
        1. First, look for documents with preferred types that have HTML files
        2. Fallback to any document with an HTML file
    """
    headers = user_agent or DEFAULT_USER_AGENT
    
    try:
        response = requests.get(txt_url, headers=headers, timeout=60)
        response.raise_for_status()
        content = response.text
    except requests.RequestException as e:
        print(f"Error fetching {txt_url}: {e}")
        return None

    # Extract all document blocks
    documents = []
    for block in DOCUMENT_PATTERN.findall(content):
        type_pattern = _create_tag_pattern("TYPE")
        filename_pattern = _create_tag_pattern("FILENAME")
        
        type_match = type_pattern.search(block)
        filename_match = filename_pattern.search(block)
        
        documents.append({
            "type": (type_match.group(1).strip().upper() if type_match else ""),
            "filename": (filename_match.group(1).strip() if filename_match else ""),
        })

    if not documents:
        return None

    # First pass: look for preferred types with HTML files
    for preferred_type in preferred_types:
        for doc in documents:
            if (doc["type"] == preferred_type and 
                doc["filename"] and 
                doc["filename"].lower().endswith((".htm", ".html"))):
                return _get_submission_directory(txt_url) + doc["filename"]

    # Second pass: any document with HTML file
    for doc in documents:
        filename = doc["filename"]
        if filename and filename.lower().endswith((".htm", ".html")):
            return _get_submission_directory(txt_url) + filename

    return None

def main():
    """Test the HTML URL extraction functionality."""
    test_url = "https://www.sec.gov/Archives/edgar/data/1653482/000162828023009925/0001628280-23-009925.txt"
    
    print(f"Testing URL: {test_url}")
    html_url = get_primary_htm_url_from_txt(test_url)
    
    if html_url:
        print(f"Found HTML URL: {html_url}")
    else:
        print("No HTML URL found")


if __name__ == "__main__":
    main()

