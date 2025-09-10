#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table and Text Extraction Script with Translation

This script extracts both tables and full text from PDF documents.
It uses img2table for table detection, pytesseract for text extraction,
and provides translation for Arabic text.

Usage:
  python extract_tables_and_text.py --pdf_path "path/to/pdf" [--pages "1,5-10"] [--output_dir "output"]
"""

import os
import argparse
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
import io
import pandas as pd
import re
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from googletrans import Translator

# Initialize translator
translator = Translator()

def is_arabic_text(text):
    """
    Check if text contains substantial Arabic characters
    
    Args:
        text: Text to analyze
        
    Returns:
        Boolean indicating if text is primarily Arabic
    """
    # Arabic Unicode range
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    arabic_matches = arabic_pattern.findall(text)
    
    # Calculate the percentage of Arabic content
    total_chars = len(text.strip())
    if total_chars == 0:
        return False
        
    arabic_chars = sum(len(match) for match in arabic_matches)
    arabic_percentage = arabic_chars / total_chars
    
    # Return True if more than 30% is Arabic
    return arabic_percentage > 0.3

def translate_text(text, source_lang="ar", target_lang="en"):
    """
    Translate text from source language to target language
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: 'ar' for Arabic)
        target_lang: Target language code (default: 'en' for English)
        
    Returns:
        Translated text or original text if translation fails
    """
    if not text or len(text.strip()) < 10:  # Skip very short text
        return text
        
    try:
        # Break text into smaller chunks to avoid translation service limits
        # Translation services often have character limits per request
        max_chunk_size = 5000
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, add current chunk to results
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"
                
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                translation = translator.translate(chunk, src=source_lang, dest=target_lang)
                translated_chunks.append(translation.text)
            else:
                translated_chunks.append("")
                
        # Join chunks back together
        return "\n".join(translated_chunks)
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def extract_page_text(pdf_path, page_num, lang="ara+eng", dpi=300, translate=True):
    """
    Extract full text from a specific page using Tesseract directly
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to extract (0-indexed)
        lang: Language for OCR
        dpi: DPI for rendering (higher = better quality but slower)
        translate: Whether to translate Arabic text to English
        
    Returns:
        Dictionary containing original and translated text (if applicable)
    """
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            return {"original": f"Error: Page {page_num} out of range (document has {len(doc)} pages)"}
        
        # Get the page
        page = doc[page_num]
        
        # Render page to an image (PNG)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        
        # Perform OCR
        original_text = pytesseract.image_to_string(img, lang=lang)
        
        # Clean up
        doc.close()
        
        result = {"original": original_text}
        
        # Translate if needed and contains Arabic
        if translate and is_arabic_text(original_text):
            print(f"  Arabic text detected on page {page_num+1}, translating...")
            translated_text = translate_text(original_text)
            result["translated"] = translated_text
        
        return result
    
    except Exception as e:
        return {"original": f"Error extracting text: {str(e)}"}

def parse_page_ranges(page_ranges_str, max_pages):
    """
    Parse page ranges string like "1,3-5,7" into a list of page numbers
    
    Args:
        page_ranges_str: String with page ranges (1-indexed for user friendliness)
        max_pages: Maximum number of pages in the document
        
    Returns:
        List of 0-indexed page numbers
    """
    if not page_ranges_str:
        return list(range(max_pages))
    
    pages = []
    for range_str in page_ranges_str.split(','):
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            # Convert to 0-indexed
            pages.extend(range(start-1, min(end, max_pages)))
        else:
            # Convert to 0-indexed
            page = int(range_str) - 1
            if 0 <= page < max_pages:
                pages.append(page)
    
    return sorted(list(set(pages)))  # Remove duplicates and sort

def save_tables_to_excel(tables, output_path):
    """
    Save extracted tables to an Excel file with multiple sheets
    
    Args:
        tables: Dict mapping page numbers to lists of ExtractedTable objects
        output_path: Path to save the Excel file
        
    Returns:
        Boolean indicating success
    """
    try:
        with pd.ExcelWriter(output_path) as writer:
            for page_num, page_tables in tables.items():
                for i, table in enumerate(page_tables):
                    sheet_name = f"Page{page_num+1}_Table{i+1}"
                    # Truncate sheet name if too long (Excel limitation)
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    
                    # Save DataFrame to Excel sheet
                    table.df.to_excel(writer, sheet_name=sheet_name)
        
        return True
    except Exception as e:
        print(f"Error saving tables to Excel: {e}")
        return False

def save_text_to_file(text, output_path):
    """
    Save extracted text to a file
    
    Args:
        text: Text content
        output_path: Path to save the text file
        
    Returns:
        Boolean indicating success
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text to file: {e}")
        return False

def main():
    local_pdf_path = "/Users/karthik/Projects/Github/doc-agent-policy/Docs/Gazette /1st Arabic 2024.pdf"
    pages= '42'
    parser = argparse.ArgumentParser(description="Extract tables and text from PDF documents")
    parser.add_argument("--pdf_path", default=local_pdf_path, help="Path to the PDF file")
    parser.add_argument("--pages", default=pages, help="Page ranges (e.g., '1,3-5,7') or leave empty for all pages")
    parser.add_argument("--output_dir", default="output", help="Directory to save the output files")
    parser.add_argument("--lang", default="ara+eng", help="OCR language(s)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering (higher = better quality but slower)")
    parser.add_argument("--no_tables", action="store_true", help="Skip table extraction")
    parser.add_argument("--no_text", action="store_true", help="Skip text extraction") 
    parser.add_argument("--no_translation", action="store_true", help="Skip translation of Arabic text")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get PDF document information
    try:
        doc = fitz.open(args.pdf_path)
        max_pages = len(doc)
        doc.close()
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return
    
    # Parse page ranges
    pages_to_process = parse_page_ranges(args.pages, max_pages)
    
    print(f"Processing PDF: {args.pdf_path}")
    print(f"Total pages: {max_pages}")
    print(f"Pages to process: {len(pages_to_process)} pages")
    
    # Extract tables
    if not args.no_tables:
        print("\n--- Extracting Tables ---")
        try:
            # Initialize OCR for table extraction
            ocr = TesseractOCR(n_threads=1, lang=args.lang)
            
            # Load document
            pdf_doc = PDF(args.pdf_path, detect_rotation=True)
            
            # Extract tables
            tables = pdf_doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=50
            )
            
            # Filter tables to only include specified pages
            filtered_tables = {p: tables.get(p, []) for p in pages_to_process if p in tables}
            
            # Display table information
            total_tables = sum(len(page_tables) for page_tables in filtered_tables.values())
            print(f"Found {total_tables} tables across {len(filtered_tables)} pages")
            
            for page_num, page_tables in filtered_tables.items():
                print(f"\nPage {page_num+1}: Found {len(page_tables)} tables")
                
                for i, table in enumerate(page_tables):
                    print(f"  Table {i+1}:")
                    print(f"  - Rows: {len(table.df)}")
                    print(f"  - Columns: {len(table.df.columns)}")
                    
                    # Check if table content is Arabic and translate if needed
                    if not args.no_translation:
                        # Check if any cell in the DataFrame contains Arabic
                        contains_arabic = False
                        for col in table.df.columns:
                            for cell in table.df[col].astype(str):
                                if is_arabic_text(cell):
                                    contains_arabic = True
                                    break
                            if contains_arabic:
                                break
                                
                        if contains_arabic:
                            print(f"  - Arabic content detected in table, translating...")
                            # Create a copy for the translated table
                            translated_df = table.df.copy()
                            
                            # Translate each cell
                            for col in translated_df.columns:
                                for idx in translated_df.index:
                                    cell = str(translated_df.at[idx, col])
                                    if is_arabic_text(cell):
                                        translated_df.at[idx, col] = translate_text(cell)
                            
                            # Save both original and translated tables
                            table_excel_path = os.path.join(args.output_dir, f"page{page_num+1}_table{i+1}.xlsx")
                            translated_excel_path = os.path.join(args.output_dir, f"page{page_num+1}_table{i+1}_translated.xlsx")
                            
                            # Save original and translated DataFrames to separate sheets in the same file
                            with pd.ExcelWriter(table_excel_path) as writer:
                                table.df.to_excel(writer, sheet_name="Original")
                                translated_df.to_excel(writer, sheet_name="Translated")
                                
                            print(f"  - Saved to: {table_excel_path} (with both original and translated sheets)")
                        else:
                            # Save only original table (no Arabic content)
                            table_excel_path = os.path.join(args.output_dir, f"page{page_num+1}_table{i+1}.xlsx")
                            table.df.to_excel(table_excel_path)
                            print(f"  - Saved to: {table_excel_path}")
                    else:
                        # Save only original table (translation disabled)
                        table_excel_path = os.path.join(args.output_dir, f"page{page_num+1}_table{i+1}.xlsx")
                        table.df.to_excel(table_excel_path)
                        print(f"  - Saved to: {table_excel_path}")
            
            # Save all tables to a single Excel file
            if total_tables > 0:
                all_tables_excel_path = os.path.join(args.output_dir, "all_tables.xlsx")
                save_tables_to_excel(filtered_tables, all_tables_excel_path)
                print(f"\nAll tables saved to: {all_tables_excel_path}")
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
    
    # Extract full text
    if not args.no_text:
        print("\n--- Extracting Full Text ---")
        
        # Extract text from each page
        for page_num in pages_to_process:
            print(f"Processing page {page_num+1}...")
            
            # Extract text using Tesseract (with optional translation)
            text_result = extract_page_text(
                args.pdf_path, 
                page_num, 
                args.lang, 
                args.dpi, 
                translate=not args.no_translation
            )
            
            # Save original text to file
            original_text_path = os.path.join(args.output_dir, f"page{page_num+1}_text.txt")
            save_text_to_file(text_result["original"], original_text_path)
            print(f"  Original text saved to: {original_text_path}")
            
            # Save translated text if available
            if "translated" in text_result:
                translated_text_path = os.path.join(args.output_dir, f"page{page_num+1}_text_translated.txt")
                save_text_to_file(text_result["translated"], translated_text_path)
                print(f"  Translated text saved to: {translated_text_path}")
        
        # Create combined text files
        combined_original = ""
        combined_translated = ""
        has_translated = False
        
        for page_num in pages_to_process:
            # Add original text
            original_text_path = os.path.join(args.output_dir, f"page{page_num+1}_text.txt")
            with open(original_text_path, 'r', encoding='utf-8') as f:
                page_text = f.read()
                combined_original += f"\n\n---------- PAGE {page_num+1} ----------\n\n"
                combined_original += page_text
            
            # Add translated text if available
            translated_text_path = os.path.join(args.output_dir, f"page{page_num+1}_text_translated.txt")
            if os.path.exists(translated_text_path):
                has_translated = True
                with open(translated_text_path, 'r', encoding='utf-8') as f:
                    page_text = f.read()
                    combined_translated += f"\n\n---------- PAGE {page_num+1} ----------\n\n"
                    combined_translated += page_text
        
        # Save combined original text
        combined_original_path = os.path.join(args.output_dir, "full_text.txt")
        save_text_to_file(combined_original, combined_original_path)
        print(f"\nCombined original text saved to: {combined_original_path}")
        
        # Save combined translated text if available
        if has_translated:
            combined_translated_path = os.path.join(args.output_dir, "full_text_translated.txt")
            save_text_to_file(combined_translated, combined_translated_path)
            print(f"Combined translated text saved to: {combined_translated_path}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 