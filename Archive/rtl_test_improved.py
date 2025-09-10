#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved RTL Text Handling Test

This script tests alternative approaches for handling right-to-left Arabic text,
focusing on preserving natural reading order without reversing lines.
"""

import os
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
import argparse
import json
import re
from typing import List, Dict, Any
from bidi import algorithm  # Import bidi algorithm for RTL handling

# Load environment variables
load_dotenv()

def get_page_text_standard(pdf_path, page_num):
    """Extract text from a PDF page using standard method."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text()
    doc.close()
    return text

def get_page_text_with_blocks(pdf_path, page_num):
    """
    Extract text from a PDF page using blocks that preserve reading order.
    This may work better for complex layouts with columns or mixed directions.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get blocks of text with their bounding boxes and sort by position
    blocks = page.get_text("blocks")
    
    # Sort blocks by vertical position (top to bottom)
    # For Arabic, we may want to sort right to left within each row as well
    blocks.sort(key=lambda b: (b[1], -b[0]))  # Sort by y, then by -x for RTL
    
    text_parts = [block[4] for block in blocks]
    text = "\n\n".join(text_parts)
    
    doc.close()
    return text

def get_page_text_with_html(pdf_path, page_num):
    """
    Extract text from a PDF page as HTML, which may preserve directionality.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get text as HTML (may contain direction markers)
    html = page.get_text("html")
    
    # Add RTL direction attribute to the body
    html = html.replace("<body", "<body dir='rtl'")
    
    doc.close()
    return html

def get_page_text_with_bidi(pdf_path, page_num):
    """
    Extract text from a PDF page and process with bidi algorithm.
    This properly handles RTL text at the line level.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    raw_text = page.get_text()
    doc.close()
    
    # Process the text line by line with bidi algorithm
    lines = raw_text.split('\n')
    processed_lines = []
    
    for line in lines:
        if line.strip():
            # Apply bidi algorithm to each line
            processed_line = algorithm.get_display(line)
            processed_lines.append(processed_line)
        else:
            processed_lines.append('')
    
    # Join the processed lines
    processed_text = '\n'.join(processed_lines)
    return processed_text

def clean_arabic_text(text):
    """
    Clean Arabic text by handling special cases and fixing common extraction issues.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Preserve sentence endings
    text = re.sub(r'([.!?،؛])\s*', r'\1\n', text)
    
    # Direction markers if needed
    text = f"\u202B{text}\u202C"  # RLE and PDF markers
    
    return text.strip()

def translate_with_azure(text, source_lang='ar', target_lang='en'):
    """Translate text using Azure Translator."""
    subscription_key = os.environ.get('AZURE_TRANSLATOR_KEY')
    # Always use the global Translator endpoint
    endpoint = 'https://api.cognitive.microsofttranslator.com'
    location = os.environ.get('AZURE_TRANSLATOR_LOCATION', 'global')
    
    path = '/translate'
    constructed_url = endpoint + path
    
    params = {
        'api-version': '3.0',
        'from': source_lang,
        'to': target_lang
    }
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json'
    }
    
    # Split text into chunks if it's too long (Azure has a limit)
    if len(text) > 5000:
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        translated_text = ""
        
        for chunk in chunks:
            body = [{'text': chunk}]
            response = requests.post(constructed_url, params=params, headers=headers, json=body)
            response_json = response.json()
            
            if response.status_code == 200:
                translated_chunk = response_json[0]['translations'][0]['text']
                translated_text += translated_chunk + " "
            else:
                print(f"Translation error: {response.status_code}, {response.text}")
                return None
        
        return translated_text.strip()
    else:
        body = [{'text': text}]
        response = requests.post(constructed_url, params=params, headers=headers, json=body)
        
        if response.status_code == 200:
            return response.json()[0]['translations'][0]['text']
        else:
            print(f"Translation error: {response.status_code}, {response.text}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Test improved RTL text handling for Arabic PDF extraction')
    parser.add_argument('--pdf', '-p', required=True, help='Path to PDF file')
    parser.add_argument('--page', '-n', type=int, default=5, help='Page number to test (0-indexed)')
    parser.add_argument('--output', '-o', default='rtl_improved_test_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file {args.pdf} not found.")
        return
    
    # Get text using different approaches
    standard_text = get_page_text_standard(args.pdf, args.page)
    block_text = get_page_text_with_blocks(args.pdf, args.page)
    html_text = get_page_text_with_html(args.pdf, args.page)
    cleaned_text = clean_arabic_text(standard_text)
    bidi_text = get_page_text_with_bidi(args.pdf, args.page)
    
    # Translate using different source formats
    standard_translation = translate_with_azure(standard_text)
    block_translation = translate_with_azure(block_text)
    cleaned_translation = translate_with_azure(cleaned_text)
    bidi_translation = translate_with_azure(bidi_text)
    
    # Print results
    print("\n===== STANDARD TEXT EXTRACTION =====")
    print(standard_text[:500] + "..." if len(standard_text) > 500 else standard_text)
    
    print("\n===== BLOCK-BASED TEXT EXTRACTION =====")
    print(block_text[:500] + "..." if len(block_text) > 500 else block_text)
    
    print("\n===== CLEANED TEXT =====")
    print(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
    
    print("\n===== HTML TEXT =====")
    print(html_text[:500] + "..." if len(html_text) > 500 else html_text)
    
    print("\n===== BIDI PROCESSED TEXT =====")
    print(bidi_text[:500] + "..." if len(bidi_text) > 500 else bidi_text)
    
    print("\n===== STANDARD TRANSLATION =====")
    print(standard_translation[:500] + "..." if standard_translation and len(standard_translation) > 500 else standard_translation)
    
    print("\n===== BLOCK-BASED TRANSLATION =====")
    print(block_translation[:500] + "..." if block_translation and len(block_translation) > 500 else block_translation)
    
    print("\n===== CLEANED TEXT TRANSLATION =====")
    print(cleaned_translation[:500] + "..." if cleaned_translation and len(cleaned_translation) > 500 else cleaned_translation)
    
    print("\n===== BIDI PROCESSED TRANSLATION =====")
    print(bidi_translation[:500] + "..." if bidi_translation and len(bidi_translation) > 500 else bidi_translation)
    
    # Save results to JSON file
    results = {
        "standard_text": standard_text,
        "block_text": block_text,
        "html_text": html_text,
        "cleaned_text": cleaned_text,
        "bidi_text": bidi_text,
        "standard_translation": standard_translation,
        "block_translation": block_translation,
        "cleaned_translation": cleaned_translation,
        "bidi_translation": bidi_translation
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 