#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Preprocessing Module

This module provides functions for extracting text, tables, and structure from PDF documents.
It includes functionality for:
- Text extraction with language detection
- Heading detection
- Table detection and conversion to markdown format
- Filtering tables by confidence level
- Exporting results to files
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from collections import defaultdict
from statistics import mode, StatisticsError
from langdetect import detect
import tiktoken  # For token counting
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parallel workers for preprocessing (tunable via env)
PREPROC_WORKERS = int(os.environ.get("PREPROC_WORKERS", "4"))


def detect_language_safe(text: str, default="en") -> str:
    """
    Detect language of text with fallback to default
    
    Args:
        text: Text to analyze
        default: Default language code to return if detection fails
        
    Returns:
        ISO language code (e.g., 'en', 'fr', 'ar')
    """
    if not text or len(text.strip()) < 20:  # Need sufficient text for detection
        return default
    
    try:
        # Check for Arabic characters (simplified)
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        if len(arabic_pattern.findall(text)) > len(text) * 0.3:  # If >30% is Arabic
            return "ar"
        
        # Use langdetect as fallback
        return detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        return default


def count_tokens(text: str, model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-5")) -> int:
    """
    Count tokens in text using tiktoken
    
    Args:
        text: Text to count tokens in
        model: Model name to use for tokenization
        
    Returns:
        Number of tokens
    """
    try:
        try:
            # Try model-specific encoding first
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback to a widely compatible encoding for GPT families
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Final fallback: rough estimate
        return len(text) // 4


def extract_table_as_markdown(doc, page_num, table_bbox):
    """
    Extract a table from a PDF and convert it to markdown format
    
    Args:
        doc: PyMuPDF document object
        page_num: Page number (0-indexed)
        table_bbox: Bounding box of the table (x0, y0, x1, y1)
        
    Returns:
        Table in markdown format or None if not a valid table (less than 2x2)
    """
    page = doc[page_num]
    x0, y0, x1, y1 = table_bbox
    
    # Extract text within the table area
    table_text = page.get_text("text", clip=(x0, y0, x1, y1))
    
    # Get text blocks within the table area for better structure analysis
    blocks = page.get_text("dict", clip=(x0, y0, x1, y1))["blocks"]
    text_blocks = [b for b in blocks if b.get("type") == 0]  # Text blocks only
    
    # Group blocks by their y-position to identify rows
    rows = defaultdict(list)
    for block in text_blocks:
        for line in block.get("lines", []):
            y_mid = (line["bbox"][1] + line["bbox"][3]) / 2
            # Round to nearest few pixels to group nearby lines
            row_key = round(y_mid / 5) * 5
            
            # Extract text from spans
            text = " ".join([span.get("text", "").strip() for span in line.get("spans", [])])
            if text.strip():
                # Store with x-position for later sorting within row
                rows[row_key].append((line["bbox"][0], text))
    
    # Sort rows by y-position (top to bottom)
    sorted_rows = sorted(rows.items())
    
    # Check if we have at least 2 rows
    if len(sorted_rows) < 2:
        return None
    
    if not sorted_rows:
        # Fallback to simple text extraction if structured extraction fails
        lines = table_text.split('\n')
        # Try to detect delimiter
        if any('|' in line for line in lines):
            # Already has pipe delimiters
            # Check if it has at least 2 rows and 2 columns
            valid_rows = [line for line in lines if '|' in line and line.count('|') >= 3]  # Need at least 2 columns (3 pipe chars)
            if len(valid_rows) < 2:
                return None
            return "\n".join(lines)
        else:
            # Try to create a simple markdown table
            # Filter out empty lines
            non_empty_lines = [line for line in lines if line.strip()]
            if len(non_empty_lines) < 2:  # Need at least 2 rows
                return None
                
            # Count columns in each line
            col_counts = [len(line.split()) for line in non_empty_lines]
            if max(col_counts) < 2:  # Need at least 2 columns
                return None
                
            max_cols = max(col_counts)
            header = "| " + " | ".join(["Column " + str(i+1) for i in range(max_cols)]) + " |"
            separator = "| " + " | ".join(["---" for _ in range(max_cols)]) + " |"
            
            table_rows = []
            for line in non_empty_lines:
                if not line.strip():
                    continue
                cells = line.split()
                # Pad with empty cells if needed
                while len(cells) < max_cols:
                    cells.append("")
                table_rows.append("| " + " | ".join(cells) + " |")
            
            return header + "\n" + separator + "\n" + "\n".join(table_rows)
    
    # Determine the number of columns by looking at the first few rows
    column_counts = [len(row[1]) for row in sorted_rows[:min(3, len(sorted_rows))]]
    
    # Check if we have at least 2 columns in any of the first rows
    if max(column_counts) < 2:
        return None
        
    num_columns = max(column_counts)
    
    # Create markdown table
    markdown_rows = []
    
    # Process each row
    for i, (_, cells) in enumerate(sorted_rows):
        # Sort cells by x-position (left to right)
        sorted_cells = [text for _, text in sorted(cells)]
        
        # Pad with empty cells if needed
        while len(sorted_cells) < num_columns:
            sorted_cells.append("")
        
        # Limit to the determined number of columns
        sorted_cells = sorted_cells[:num_columns]
        
        # Create markdown row
        markdown_row = "| " + " | ".join(sorted_cells) + " |"
        markdown_rows.append(markdown_row)
        
        # Add separator after the first row
        if i == 0:
            separator = "| " + " | ".join(["---" for _ in range(len(sorted_cells))]) + " |"
            markdown_rows.append(separator)
    
    return "\n".join(markdown_rows)


def is_valid_table(markdown_table):
    """
    Check if a markdown table is valid (at least 2x2)
    
    Args:
        markdown_table: Table in markdown format
        
    Returns:
        Boolean indicating if the table is valid
    """
    if not markdown_table:
        return False
        
    lines = markdown_table.strip().split('\n')
    
    # Need at least 3 lines (header, separator, and at least one data row)
    if len(lines) < 3:
        return False
    
    # Check if each line has at least 3 pipe characters (2 columns)
    for line in lines:
        if line.count('|') < 3:  # Need at least 2 columns (3 pipe chars)
            return False
    
    # Additional check: Count actual data cells (not just pipe characters)
    # This helps catch cases where there might be pipes but not actual data
    data_rows = [line for line in lines if not line.strip().startswith('|--')]
    if len(data_rows) < 2:  # Need at least 2 data rows
        return False
        
    # Check that we have actual content in multiple cells
    non_empty_cells = 0
    for row in data_rows:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Remove empty elements from start/end
        non_empty_cells += sum(1 for cell in cells if cell)
    
    # Need at least 4 non-empty cells for a meaningful 2x2 table
    # or at least 3 cells with content if it's a table with headers
    return non_empty_cells >= 3


def detect_tables_in_pdf(pdf_path, page_numbers=None):
    """
    Detect tables in a PDF document using PyMuPDF (fitz)
    Only structures with at least 2 rows and 2 columns are considered tables.
    
    Args:
        pdf_path: Path to the PDF file
        page_numbers: List of specific page numbers to process (0-indexed, None for all pages)
        
    Returns:
        Dictionary mapping page numbers to lists of detected tables
    """
    tables_by_page = defaultdict(list)
    
    try:
        doc = fitz.open(pdf_path)
        
        # Process all pages or only specified ones
        pages_to_process = page_numbers if page_numbers is not None else range(len(doc))
        
        for page_num in pages_to_process:
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Method 1: Look for table-like structures using lines
            blocks = page.get_text("dict")["blocks"]
            lines = []
            
            # Extract horizontal and vertical lines
            for block in blocks:
                if block.get("type") == 1:  # Line type
                    lines.append(block)
            
            # Look for grid patterns formed by intersecting lines
            if len(lines) > 4:  # Minimum lines needed for a simple table
                # Group lines by orientation (horizontal vs vertical)
                h_lines = [line for line in lines if abs(line["bbox"][1] - line["bbox"][3]) < 2]
                v_lines = [line for line in lines if abs(line["bbox"][0] - line["bbox"][2]) < 2]
                
                # If we have both horizontal and vertical lines, we might have a table
                # Need at least 3 horizontal lines (2 rows) and 3 vertical lines (2 columns)
                if len(h_lines) >= 3 and len(v_lines) >= 3:
                    # Calculate the bounding box of the potential table
                    x0 = min([line["bbox"][0] for line in v_lines])
                    y0 = min([line["bbox"][1] for line in h_lines])
                    x1 = max([line["bbox"][2] for line in v_lines])
                    y1 = max([line["bbox"][3] for line in h_lines])
                    
                    # Extract table content as markdown
                    markdown_table = extract_table_as_markdown(doc, page_num, (x0, y0, x1, y1))
                    
                    # Only add if it's a valid table (at least 2x2)
                    if markdown_table and is_valid_table(markdown_table):
                        tables_by_page[page_num].append({
                            "bbox": (x0, y0, x1, y1),
                            "method": "line_detection",
                            "confidence": 0.7,
                            "markdown": markdown_table
                        })
            
            # Method 2: Look for text blocks arranged in a grid pattern
            text_blocks = [b for b in blocks if b.get("type") == 0]  # Text blocks
            
            if len(text_blocks) > 4:
                # Sort blocks by y-coordinate (top to bottom)
                rows = defaultdict(list)
                for block in text_blocks:
                    # Use the middle y-coordinate to group by rows
                    y_mid = (block["bbox"][1] + block["bbox"][3]) / 2
                    # Round to nearest 5 pixels to group nearby rows
                    row_key = round(y_mid / 5) * 5
                    rows[row_key].append(block)
                
                # Check if we have multiple rows with similar number of blocks
                row_lengths = [len(row) for row in rows.values()]
                
                # Need at least 2 rows and 2 columns
                if len(rows) >= 2 and max(row_lengths) >= 2:
                    # Check if first few rows have similar number of columns
                    if len(set(row_lengths[:min(3, len(row_lengths))])) <= 2:
                        # Likely a table if first few rows have similar number of columns
                        all_blocks = [block for row in rows.values() for block in row]
                        if all_blocks:
                            x0 = min([block["bbox"][0] for block in all_blocks])
                            y0 = min([block["bbox"][1] for block in all_blocks])
                            x1 = max([block["bbox"][2] for block in all_blocks])
                            y1 = max([block["bbox"][3] for block in all_blocks])
                            
                            # Extract table content as markdown
                            markdown_table = extract_table_as_markdown(doc, page_num, (x0, y0, x1, y1))
                            
                            # Only add if it's a valid table (at least 2x2)
                            if markdown_table and is_valid_table(markdown_table):
                                tables_by_page[page_num].append({
                                    "bbox": (x0, y0, x1, y1),
                                    "method": "text_alignment",
                                    "confidence": 0.6,
                                    "rows": len(rows),
                                    "columns": max(row_lengths),
                                    "markdown": markdown_table
                                })
            
            # Method 3: Use PyMuPDF's built-in table detection if available
            try:
                # This requires a newer version of PyMuPDF
                page_tables = page.find_tables()
                if page_tables and hasattr(page_tables, "tables"):
                    for table in page_tables.tables:
                        # Check if the table has at least 2 rows and 2 columns
                        rows = table.rows if hasattr(table, "rows") else 0
                        cols = table.cols if hasattr(table, "cols") else 0
                        
                        # More stringent check for PyMuPDF native detection
                        # Sometimes it reports rows/cols incorrectly
                        if rows >= 2 and cols >= 2:
                            # Extract table content as markdown
                            markdown_table = extract_table_as_markdown(doc, page_num, tuple(table.bbox))
                            
                            # Apply our strict validation
                            if markdown_table and is_valid_table(markdown_table):
                                # Additional check: Count actual cells with content
                                lines = markdown_table.strip().split('\n')
                                data_rows = [line for line in lines if not line.strip().startswith('|--')]
                                
                                # Count cells with actual content
                                content_cells = 0
                                for row in data_rows:
                                    cells = [cell.strip() for cell in row.split('|')[1:-1]]
                                    content_cells += sum(1 for cell in cells if cell)
                                
                                # Only add if we have enough cells with content
                                if content_cells >= 3:
                                    tables_by_page[page_num].append({
                                        "bbox": tuple(table.bbox),
                                        "method": "pymupdf_native",
                                        "confidence": 0.9,
                                        "rows": rows,
                                        "columns": cols,
                                        "markdown": markdown_table
                                    })
            except Exception:
                # Older version or other issue, continue with other methods
                pass
                
        doc.close()
        return dict(tables_by_page)
    
    except Exception as e:
        print(f"Error detecting tables: {e}")
        return dict(tables_by_page)


def extract_text_from_pdf(pdf_path, detect_tables=True):
    """
    Extract text and structure from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        detect_tables: Whether to detect tables in the document
        
    Returns:
        List of dictionaries, one per page, with extracted information
    """
    # Detect tables first if requested
    tables_by_page = detect_tables_in_pdf(pdf_path) if detect_tables else {}
    
    # Determine total pages without keeping doc open during threading
    try:
        with fitz.open(pdf_path) as tmp:
            total_pages = len(tmp)
    except Exception as e:
        print(f"Error opening document: {e}")
        return []
    
    def _process_page(idx: int) -> Dict[str, Any]:
        page_dict: Dict[str, Any] = {
            "page_number": idx + 1,
            "text": "",
            "language": "en",
            "headings": [],
            "has_tables": False,
            "tables": []
        }
        try:
            doc = fitz.open(pdf_path)
            if idx >= len(doc):
                doc.close()
                return page_dict
            page = doc[idx]
            text = page.get_text()
            page_dict["text"] = text
            page_dict["language"] = detect_language_safe(text)
            blocks = page.get_text("dict")["blocks"]
            headings = []
            font_sizes = []
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_sizes.append(span.get("size", 0))
            if font_sizes:
                try:
                    common_size = mode(font_sizes)
                except StatisticsError:
                    common_size = sum(font_sizes) / len(font_sizes)
                for block in blocks:
                    if block.get("type") == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("size", 0) > common_size * 1.2 and len(span.get("text", "").strip()) > 3:
                                    heading_text = span.get("text", "").strip()
                                    if not re.match(r'^\d+$', heading_text) and heading_text not in headings:
                                        headings.append(heading_text)
            page_dict["headings"] = headings
            page_tables = tables_by_page.get(idx, [])
            page_dict["has_tables"] = len(page_tables) > 0
            page_dict["tables"] = page_tables
            doc.close()
            return page_dict
        except Exception as e:
            try:
                doc.close()
            except Exception:
                pass
            return page_dict
    
    results: List[Optional[Dict[str, Any]]] = [None] * total_pages
    with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as ex:
        futs = [ex.submit(_process_page, i) for i in range(total_pages)]
        for f in as_completed(futs):
            data = f.result()
            idx = data.get("page_number", 1) - 1
            if 0 <= idx < total_pages:
                results[idx] = data
    
    return [r for r in results if r is not None]


def filter_tables_by_confidence(pages, min_confidence=0.8):
    """
    Filter tables in the pages data structure by confidence level
    
    Args:
        pages: List of page dictionaries from extract_text_from_pdf
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        New list of page dictionaries with only high-confidence tables
    """
    filtered_pages = []
    
    for page in pages:
        # Create a copy of the page dictionary
        filtered_page = page.copy()
        
        # Filter tables by confidence
        high_confidence_tables = [
            table for table in page["tables"] 
            if table.get("confidence", 0) >= min_confidence
        ]
        
        # Update the page with filtered tables
        filtered_page["tables"] = high_confidence_tables
        filtered_page["has_tables"] = len(high_confidence_tables) > 0
        
        filtered_pages.append(filtered_page)
    
    return filtered_pages


def dump_text_to_chat(pages, max_pages=None, chars_per_page=1000, output_file=None, min_confidence=0.0):
    """
    Dump extracted text in a readable format to chat and optionally to a file
    
    Args:
        pages: List of page dictionaries from extract_text_from_pdf
        max_pages: Maximum number of pages to display (None for all)
        chars_per_page: Maximum characters to display per page
        output_file: Path to save the output (None for no file output)
        min_confidence: Minimum confidence threshold for tables (0.0 to 1.0)
    """
    # Filter tables by confidence if needed
    if min_confidence > 0:
        pages = filter_tables_by_confidence(pages, min_confidence)
    
    if max_pages is None:
        max_pages = len(pages)
    else:
        max_pages = min(max_pages, len(pages))
    
    output = []
    output.append(f"Displaying text from {max_pages} of {len(pages)} total pages:\n")
    output.append("="*80)
    
    for i, page in enumerate(pages[:max_pages]):
        output.append(f"\n{'='*40} PAGE {page['page_number']} {'='*40}\n")
        
        # Display truncated text if needed
        text = page["text"]
        if len(text) > chars_per_page:
            output.append(f"{text[:chars_per_page]}...\n[truncated - {len(text)} total characters]")
        else:
            output.append(text)
        
        # Show headings if available
        if page["headings"]:
            output.append("\nHeadings on this page:")
            for heading in page["headings"]:
                output.append(f"  - {heading}")
        
        # Show table info if available
        if page["has_tables"]:
            output.append(f"\nThis page contains {len(page['tables'])} tables:")
            for j, table in enumerate(page['tables']):
                output.append(f"\nTable {j+1} (confidence: {table['confidence']:.2f}):")
                if "markdown" in table:
                    output.append(f"\n{table['markdown']}")
                else:
                    output.append("[Table content not available in markdown format]")
        
        output.append("\n" + "="*80)
    
    if max_pages < len(pages):
        output.append(f"\n... {len(pages) - max_pages} more pages not shown ...")
    
    # Join all output lines
    full_output = "\n".join(output)
    
    # Print to console
    print(full_output)
    
    # Save to file if requested
    if output_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_output)
            print(f"\nOutput saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving to file: {e}")


def dump_text_to_file(pages, output_file, max_pages=None, chars_per_page=None, min_confidence=0.0):
    """
    Save extracted text to a file repository
    
    Args:
        pages: List of page dictionaries from extract_text_from_pdf
        output_file: Path to save the output
        max_pages: Maximum number of pages to save (None for all)
        chars_per_page: Maximum characters to save per page (None for all)
        min_confidence: Minimum confidence threshold for tables (0.0 to 1.0)
    """
    # Filter tables by confidence if needed
    if min_confidence > 0:
        pages = filter_tables_by_confidence(pages, min_confidence)
    
    if max_pages is None:
        max_pages = len(pages)
    else:
        max_pages = min(max_pages, len(pages))
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Document Export - {len(pages)} total pages\n")
            f.write("="*80 + "\n\n")
            
            for i, page in enumerate(pages[:max_pages]):
                f.write(f"{'='*40} PAGE {page['page_number']} {'='*40}\n\n")
                
                # Write text, truncated if needed
                text = page["text"]
                if chars_per_page and len(text) > chars_per_page:
                    f.write(f"{text[:chars_per_page]}...\n[truncated - {len(text)} total characters]\n")
                else:
                    f.write(f"{text}\n")
                
                # Write headings if available
                if page["headings"]:
                    f.write("\nHeadings on this page:\n")
                    for heading in page["headings"]:
                        f.write(f"  - {heading}\n")
                
                # Write table info and content if available
                if page["has_tables"]:
                    f.write(f"\nThis page contains {len(page['tables'])} tables:\n")
                    for j, table in enumerate(page['tables']):
                        f.write(f"\nTable {j+1} (confidence: {table['confidence']:.2f}):\n")
                        if "markdown" in table:
                            f.write(f"\n{table['markdown']}\n")
                        else:
                            f.write("[Table content not available in markdown format]\n")
                
                f.write("\n" + "="*80 + "\n\n")
            
            if max_pages < len(pages):
                f.write(f"\n... {len(pages) - max_pages} more pages not included ...\n")
        
        print(f"Document successfully exported to: {output_file}")
        return True
    except Exception as e:
        print(f"Error exporting document: {e}")
        return False


def save_tables_only(pages, output_file, min_confidence=0.0, format="markdown"):
    """
    Save only the tables from a document to a file
    
    Args:
        pages: List of page dictionaries from extract_text_from_pdf
        output_file: Path to save the output
        min_confidence: Minimum confidence threshold for tables (0.0 to 1.0)
        format: Output format ("markdown" or "json")
        
    Returns:
        Boolean indicating success or failure
    """
    # Filter tables by confidence if needed
    if min_confidence > 0:
        pages = filter_tables_by_confidence(pages, min_confidence)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if format.lower() == "json":
            # Extract tables into a structured format
            tables_data = []
            for page in pages:
                if page["has_tables"]:
                    for table in page["tables"]:
                        table_data = {
                            "page": page["page_number"],
                            "confidence": table.get("confidence", 0),
                            "method": table.get("method", "unknown"),
                            "rows": table.get("rows", 0),
                            "columns": table.get("columns", 0),
                            "markdown": table.get("markdown", "")
                        }
                        tables_data.append(table_data)
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tables_data, f, indent=2)
        else:
            # Save as markdown
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Tables Extracted from Document\n\n")
                
                for page in pages:
                    if page["has_tables"]:
                        f.write(f"## Page {page['page_number']}\n\n")
                        for j, table in enumerate(page['tables']):
                            f.write(f"### Table {j+1} (confidence: {table['confidence']:.2f})\n\n")
                            if "markdown" in table:
                                f.write(f"{table['markdown']}\n\n")
                            else:
                                f.write("*Table content not available in markdown format*\n\n")
                        f.write("---\n\n")
        
        print(f"Tables successfully exported to: {output_file}")
        return True
    except Exception as e:
        print(f"Error exporting tables: {e}")
        return False


def analyze_document_structure(pdf_path, output_file=None):
    """
    Analyze the structure of a PDF document and generate statistics
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the analysis (None for no file output)
        
    Returns:
        Dictionary with document statistics
    """
    pages = extract_text_from_pdf(pdf_path)
    
    # Calculate statistics
    stats = {
        "total_pages": len(pages),
        "pages_with_tables": sum(1 for page in pages if page["has_tables"]),
        "total_tables": sum(len(page["tables"]) for page in pages),
        "pages_with_headings": sum(1 for page in pages if page["headings"]),
        "total_headings": sum(len(page["headings"]) for page in pages),
        "languages": {},
        "table_confidence": {
            "high": sum(1 for page in pages for table in page["tables"] if table.get("confidence", 0) >= 0.8),
            "medium": sum(1 for page in pages for table in page["tables"] if 0.6 <= table.get("confidence", 0) < 0.8),
            "low": sum(1 for page in pages for table in page["tables"] if table.get("confidence", 0) < 0.6)
        },
        "page_stats": []
    }
    
    # Count languages
    for page in pages:
        lang = page["language"]
        stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
    
    # Collect per-page statistics
    for page in pages:
        page_stats = {
            "page_number": page["page_number"],
            "token_count": count_tokens(page["text"]),
            "has_tables": page["has_tables"],
            "table_count": len(page["tables"]),
            "heading_count": len(page["headings"]),
            "language": page["language"]
        }
        stats["page_stats"].append(page_stats)
    
    # Save to file if requested
    if output_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"Document analysis saved to: {output_file}")
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    return stats


# Example usage
if __name__ == "__main__":
    # This code runs when the script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Preprocessing Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--tables-only", action="store_true", help="Extract only tables")
    parser.add_argument("--analyze", action="store_true", help="Analyze document structure")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence for tables (0.0-1.0)")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze document structure
        output_file = args.output or "document_analysis.json"
        analyze_document_structure(args.pdf_path, output_file)
    elif args.tables_only:
        # Extract only tables
        pages = extract_text_from_pdf(args.pdf_path)
        output_file = args.output or "tables_output.md"
        format_type = "json" if args.format == "json" else "markdown"
        save_tables_only(pages, output_file, args.min_confidence, format_type)
    else:
        # Extract full text
        pages = extract_text_from_pdf(args.pdf_path)
        if args.max_pages:
            pages = pages[:args.max_pages]
        
        if args.output:
            dump_text_to_file(pages, args.output, min_confidence=args.min_confidence)
        else:
            dump_text_to_chat(pages, min_confidence=args.min_confidence)
