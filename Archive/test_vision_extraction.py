#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified test script for vision-based extraction from PDFs
This directly uses the vision model to extract text from a specific page or page range
"""

import os
import logging
import argparse
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Import vision extraction function from smart chunking agent
from b_smart_chunking_agent import (
    extract_policy_content_with_vision,
    setup_sync_azure_openai_client
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the vision extraction functionality"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Vision-based PDF content extraction')
    parser.add_argument('--pdf', '-p', required=True, help='Path to PDF file')
    parser.add_argument('--start-page', '-s', type=int, required=True, help='Start page (0-indexed)')
    parser.add_argument('--end-page', '-e', type=int, default=None, help='End page (0-indexed, default: same as start)')
    parser.add_argument('--arabic', '-a', action='store_true', help='Indicate if document is in Arabic')
    parser.add_argument('--print-only', '-o', action='store_true', help='Only print to console without saving to file')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        return
    
    # Set end page to start page if not specified
    end_page = args.end_page if args.end_page is not None else args.start_page
    
    # Verify page numbers are valid
    doc = fitz.open(args.pdf)
    if args.start_page < 0 or args.start_page >= len(doc) or end_page < 0 or end_page >= len(doc):
        logger.error(f"Invalid page range: {args.start_page}-{end_page} (document has {len(doc)} pages)")
        doc.close()
        return
    doc.close()
    
    # Create metadata with language information
    metadata = {"is_arabic": args.arabic}
    
    logger.info(f"Extracting content from {args.pdf}, pages {args.start_page+1}-{end_page+1}")
    logger.info(f"Document is marked as {'Arabic' if args.arabic else 'non-Arabic'}")
    
    # Set up the Azure OpenAI client
    client = setup_sync_azure_openai_client()
    
    # Extract content using vision
    logger.info("Starting vision extraction...")
    extracted_content = extract_policy_content_with_vision(
        client,
        args.pdf,
        args.start_page,
        end_page,
        metadata
    )
    
    # Print statistics and content
    char_count = len(extracted_content)
    line_count = extracted_content.count('\n') + 1
    
    logger.info(f"Extraction complete. Extracted {char_count} characters, approximately {line_count} lines")
    
    print("\n" + "="*80)
    print(f"EXTRACTED CONTENT (Pages {args.start_page+1}-{end_page+1}):")
    print("="*80)
    print(extracted_content)
    print("="*80)
    
    # Save the extracted content to a file if not print-only
    if not args.print_only:
        output_filename = f"vision_output.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(extracted_content)
        
        logger.info(f"Saved extracted content to {output_filename}")

if __name__ == "__main__":
    main() 