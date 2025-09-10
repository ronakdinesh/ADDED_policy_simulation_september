#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table Detection Test Script

This script provides a standalone version of the table detection methods from a_preprocessing.py.
It allows testing table detection on specific pages of a PDF document.

Usage:
  python test_table_detection.py --pdf_path path/to/document.pdf [--page_num 5] [--all_pages] [--debug]
"""

import os
import re
import json
import fitz  # PyMuPDF
import argparse
from collections import defaultdict
from statistics import mode, StatisticsError


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
    
    # Need at least 3 non-empty cells for a meaningful table
    return non_empty_cells >= 3


def detect_time_based_tables(text):
    """
    Detect time-based tables using regular expressions
    
    Args:
        text: Text to analyze
        
    Returns:
        Boolean indicating if time-based table patterns were found
    """
    # Patterns for time-based tables (works in English and when translated from Arabic)
    time_patterns = [
        # Match time formats like "09:00", "7:30 AM", "06:30 صباحاً"
        r"\d{1,2}[:\.]\d{2}\s*(AM|PM|صباحاً|مساءً|صباحا|مساء)?", 
        
        # Match labels like Morning/Evening periods
        r"(Morning|Evening|Period|صباحية|مسائية|الفترة).*(From|To|من|إلى)",
        
        # Match from/to with times
        r"(From|To|من|إلى).*?\d{1,2}[:\.]\d{2}"
    ]
    
    matches = 0
    for pattern in time_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
    
    # If we found multiple time-related patterns, likely a time table
    return matches >= 2


def detect_aligned_columns(text):
    """
    Detect aligned columns in text that might indicate a table
    
    Args:
        text: Text to analyze
        
    Returns:
        Boolean indicating if aligned columns were detected
    """
    lines = text.split('\n')
    if len(lines) < 3:
        return False
    
    # Look for consistent spacing patterns
    space_patterns = []
    for line in lines[:min(5, len(lines))]:
        if not line.strip():
            continue
            
        # Find positions of whitespace gaps (3+ spaces)
        spaces = [m.start() for m in re.finditer(r'\s{3,}', line)]
        if spaces:
            space_patterns.append(spaces)
    
    # Need at least 3 lines with similar spacing patterns
    if len(space_patterns) < 3:
        return False
    
    # Check if at least 2 lines have similar gap positions
    for i in range(len(space_patterns) - 1):
        for j in range(i + 1, len(space_patterns)):
            # Check if any positions are close (within 3 characters)
            for pos1 in space_patterns[i]:
                if any(abs(pos1 - pos2) <= 3 for pos2 in space_patterns[j]):
                    return True
    
    return False


def detect_tables_in_pdf(pdf_path, page_numbers=None, debug=False):
    """
    Detect tables in a PDF document using PyMuPDF (fitz)
    Only structures with at least 2 rows and 2 columns are considered tables.
    
    Args:
        pdf_path: Path to the PDF file
        page_numbers: List of specific page numbers to process (0-indexed, None for all pages)
        debug: Whether to print debug information
        
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
                
            if debug:
                print(f"\nProcessing page {page_num} (0-indexed)")
                
            page = doc[page_num]
            page_text = page.get_text()
            
            # NEW: Check for time-based tables
            has_time_patterns = detect_time_based_tables(page_text)
            if has_time_patterns and debug:
                print(f"✓ Detected time-based table patterns on page {page_num}")
            
            # NEW: Check for aligned columns
            has_aligned_columns = detect_aligned_columns(page_text)
            if has_aligned_columns and debug:
                print(f"✓ Detected aligned columns on page {page_num}")
            
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
                if (len(h_lines) >= 3 and len(v_lines) >= 3) or (len(h_lines) >= 2 and has_time_patterns):
                    # Calculate the bounding box of the potential table
                    x0 = min([line["bbox"][0] for line in v_lines]) if v_lines else page.rect.x0
                    y0 = min([line["bbox"][1] for line in h_lines]) if h_lines else page.rect.y0
                    x1 = max([line["bbox"][2] for line in v_lines]) if v_lines else page.rect.x1
                    y1 = max([line["bbox"][3] for line in h_lines]) if h_lines else page.rect.y1
                    
                    if debug and (len(h_lines) >= 2 or len(v_lines) >= 2):
                        print(f"✓ Found grid pattern with {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
                    
                    # Extract table content as markdown
                    markdown_table = extract_table_as_markdown(doc, page_num, (x0, y0, x1, y1))
                    
                    # Only add if it's a valid table (at least 2x2)
                    if markdown_table and is_valid_table(markdown_table):
                        confidence = 0.8 if len(h_lines) >= 3 and len(v_lines) >= 3 else 0.7
                        confidence += 0.1 if has_time_patterns else 0
                        confidence = min(1.0, confidence)  # Cap at 1.0
                        
                        tables_by_page[page_num].append({
                            "bbox": (x0, y0, x1, y1),
                            "method": "line_detection",
                            "confidence": confidence,
                            "markdown": markdown_table
                        })
                        
                        if debug:
                            print(f"✅ Added table with confidence {confidence:.2f} (line detection)")
                            print(f"Table content:\n{markdown_table[:200]}...")
            
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
                # Add bonus for time-pattern tables - only need 2 rows
                min_rows = 2 if has_time_patterns else 2
                
                if debug:
                    print(f"Text block rows detected: {len(rows)}")
                    print(f"Row lengths: {row_lengths[:5]} {'...' if len(row_lengths) > 5 else ''}")
                
                if len(rows) >= min_rows and max(row_lengths) >= 2:
                    # Check if first few rows have similar number of columns
                    # For time tables, be more lenient
                    if len(set(row_lengths[:min(3, len(row_lengths))])) <= 2 or has_time_patterns:
                        # Likely a table if first few rows have similar number of columns
                        all_blocks = [block for row in rows.values() for block in row]
                        if all_blocks:
                            x0 = min([block["bbox"][0] for block in all_blocks])
                            y0 = min([block["bbox"][1] for block in all_blocks])
                            x1 = max([block["bbox"][2] for block in all_blocks])
                            y1 = max([block["bbox"][3] for block in all_blocks])
                            
                            if debug:
                                print(f"✓ Found {len(rows)} rows with similar structure")
                            
                            # Extract table content as markdown
                            markdown_table = extract_table_as_markdown(doc, page_num, (x0, y0, x1, y1))
                            
                            # Only add if it's a valid table (at least 2x2)
                            if markdown_table and is_valid_table(markdown_table):
                                confidence = 0.6
                                confidence += 0.1 if has_time_patterns else 0
                                confidence += 0.1 if has_aligned_columns else 0
                                confidence = min(0.95, confidence)  # Cap at 0.95
                                
                                tables_by_page[page_num].append({
                                    "bbox": (x0, y0, x1, y1),
                                    "method": "text_alignment",
                                    "confidence": confidence,
                                    "rows": len(rows),
                                    "columns": max(row_lengths),
                                    "markdown": markdown_table
                                })
                                
                                if debug:
                                    print(f"✅ Added table with confidence {confidence:.2f} (text alignment)")
                                    print(f"Table content:\n{markdown_table[:200]}...")
            
            # Method 3: Use PyMuPDF's built-in table detection if available
            try:
                # This requires a newer version of PyMuPDF
                page_tables = page.find_tables()
                if page_tables and hasattr(page_tables, "tables"):
                    if debug:
                        print(f"✓ PyMuPDF native detection found {len(page_tables.tables)} tables")
                        
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
                                    confidence = 0.9
                                    confidence += 0.05 if has_time_patterns else 0
                                    confidence = min(1.0, confidence)  # Cap at 1.0
                                    
                                    tables_by_page[page_num].append({
                                        "bbox": tuple(table.bbox),
                                        "method": "pymupdf_native",
                                        "confidence": confidence,
                                        "rows": rows,
                                        "columns": cols,
                                        "markdown": markdown_table
                                    })
                                    
                                    if debug:
                                        print(f"✅ Added table with confidence {confidence:.2f} (PyMuPDF native)")
                                        print(f"Table content:\n{markdown_table[:200]}...")
            except Exception as e:
                if debug:
                    print(f"⚠️ PyMuPDF native detection failed: {e}")
                # Older version or other issue, continue with other methods
                pass
                
            # Method 4: Specialized time table detection
            if has_time_patterns and not tables_by_page[page_num]:
                # If we have time patterns but no tables detected yet, try a more focused approach
                
                # Look for lines with time patterns
                lines = page_text.split('\n')
                time_pattern = r"\d{1,2}[:\.]\d{2}\s*(AM|PM|صباحاً|مساءً|صباحا|مساء)?"
                time_lines = []
                
                for i, line in enumerate(lines):
                    if re.search(time_pattern, line, re.IGNORECASE):
                        # Include a few lines before and after for context
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        time_lines.extend(lines[start:end])
                
                if time_lines:
                    if debug:
                        print(f"✓ Found {len(time_lines)} lines with time patterns")
                    
                    # Create a simple markdown table from these lines
                    # First, determine if there's a pattern of columns
                    time_text = "\n".join(time_lines)
                    
                    # Try to extract the table from the whole page as a fallback
                    markdown_table = extract_table_as_markdown(doc, page_num, page.rect)
                    
                    if markdown_table and is_valid_table(markdown_table):
                        tables_by_page[page_num].append({
                            "bbox": tuple(page.rect),
                            "method": "time_pattern_detection",
                            "confidence": 0.75,
                            "markdown": markdown_table
                        })
                        
                        if debug:
                            print(f"✅ Added table with confidence 0.75 (time pattern detection)")
                            print(f"Table content:\n{markdown_table[:200]}...")
            
            # Log page results
            if debug:
                table_count = len(tables_by_page[page_num])
                print(f"Page {page_num}: Found {table_count} tables")
                
        doc.close()
        return dict(tables_by_page)
    
    except Exception as e:
        print(f"Error detecting tables: {e}")
        return dict(tables_by_page)


def save_tables_to_file(tables_by_page, output_file):
    """
    Save detected tables to a file
    
    Args:
        tables_by_page: Dictionary mapping page numbers to lists of detected tables
        output_file: Path to save the output
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Tables Detected in Document\n\n")
            
            if not tables_by_page:
                f.write("No tables detected in document.\n")
                return True
            
            for page_num, tables in tables_by_page.items():
                if tables:
                    f.write(f"## Page {page_num + 1} (0-indexed: {page_num})\n\n")
                    for i, table in enumerate(tables):
                        f.write(f"### Table {i+1}\n\n")
                        f.write(f"- **Method:** {table['method']}\n")
                        f.write(f"- **Confidence:** {table['confidence']:.2f}\n")
                        f.write(f"- **Bounding Box:** {table['bbox']}\n\n")
                        
                        if "markdown" in table:
                            f.write("**Table Content:**\n\n")
                            f.write(f"{table['markdown']}\n\n")
                        else:
                            f.write("*Table content not available in markdown format*\n\n")
                        
                        f.write("---\n\n")
            
        print(f"Tables successfully saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving tables: {e}")
        return False


def display_tables(tables_by_page):
    """
    Display detected tables to the console
    
    Args:
        tables_by_page: Dictionary mapping page numbers to lists of detected tables
    """
    total_tables = sum(len(tables) for tables in tables_by_page.values())
    
    if total_tables == 0:
        print("No tables detected in document.")
        return
        
    print(f"\n{'-'*80}")
    print(f"Found {total_tables} tables in {len(tables_by_page)} pages")
    print(f"{'-'*80}")
    
    for page_num, tables in tables_by_page.items():
        if tables:
            print(f"\nPage {page_num + 1} (0-indexed: {page_num}): {len(tables)} tables")
            for i, table in enumerate(tables):
                print(f"\n  Table {i+1}:")
                print(f"  - Method: {table['method']}")
                print(f"  - Confidence: {table['confidence']:.2f}")
                print(f"  - Bounding Box: {table['bbox']}")
                
                if "markdown" in table:
                    print("\n  Table Content:")
                    
                    # Format the table for display, limiting to first few rows
                    lines = table['markdown'].split('\n')
                    display_lines = lines[:min(8, len(lines))]
                    if len(lines) > 8:
                        display_lines.append("... (table truncated) ...")
                    
                    for line in display_lines:
                        print(f"    {line}")
                    
                    print()


if __name__ == "__main__":
    # Parse command line arguments
    local_pdf_path = "Archive/test_pdfs/test_pdf_1.pdf"
    parser = argparse.ArgumentParser(description="Test table detection in PDF documents")
    parser.add_argument("--pdf_path", default="",required=True, help="Path to the PDF file")
    parser.add_argument("--page_num", type=int, help="Specific page number to process (0-indexed)")
    parser.add_argument("--all_pages", action="store_true", help="Process all pages in the document")
    parser.add_argument("--output", help="Path to save detected tables")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Determine pages to process
    pages_to_process = None
    if args.page_num is not None:
        pages_to_process = [args.page_num]
    elif not args.all_pages:
        # Default to page 0 if neither specific page nor all pages are specified
        pages_to_process = [0]
    
    # Detect tables
    print(f"Detecting tables in {args.pdf_path}")
    if pages_to_process:
        print(f"Processing page(s): {pages_to_process}")
    else:
        print("Processing all pages")
    
    tables_by_page = detect_tables_in_pdf(args.pdf_path, pages_to_process, args.debug)
    
    # Display results
    display_tables(tables_by_page)
    
    # Save to file if requested
    if args.output:
        save_tables_to_file(tables_by_page, args.output)
        print(f"Results saved to {args.output}")
    
    # Summary
    total_tables = sum(len(tables) for tables in tables_by_page.values())
    print(f"\nSummary: Found {total_tables} tables in {len(tables_by_page)} pages") 