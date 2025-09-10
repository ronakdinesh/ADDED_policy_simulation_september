#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for policy extraction, including language detection and translation
"""

import os
import re
import fitz  # PyMuPDF
import requests
import uuid
import json
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables for API keys
load_dotenv()

# Try to import img2table for table detection
try:
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR
    HAS_IMG2TABLE = True
except ImportError:
    logger.warning("img2table not installed. Install with: pip install img2table")
    HAS_IMG2TABLE = False

# Attempt to import langdetect, use a fallback if not available
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    logger.warning("langdetect not installed. Install with: pip install langdetect")
    HAS_LANGDETECT = False

# Import bidi library for RTL text handling
try:
    from bidi import algorithm
    HAS_BIDI = True
except ImportError:
    logger.warning("python-bidi not installed. Install with: pip install python-bidi")
    HAS_BIDI = False

def process_with_bidi(text: str) -> str:
    """
    Process text with bidi algorithm for proper RTL text handling.
    
    This applies the bidirectional algorithm to properly handle right-to-left text
    such as Arabic, processing each line separately for accurate display.
    
    Args:
        text: The text to process
        
    Returns:
        Text processed with the bidi algorithm for proper RTL display
    """
    if not HAS_BIDI:
        logger.warning("bidi library not available, returning original text")
        return text
        
    lines = text.split('\n')
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


def detect_document_language(doc_path: str, num_pages: int = 10) -> Dict[str, Any]:
    """
    Detect the dominant language in a document by analyzing text from several pages.
    
    Args:
        doc_path: Path to the PDF document
        num_pages: Maximum number of pages to check
        
    Returns:
        Dict with language information including dominant language, confidence, etc.
    """
    result = {
        "dominant_language": "en",  # Default to English
        "is_arabic": False,
        "confidence": 0.0,
        "language_counts": {},
        "sample_size_chars": 0
    }
    
    try:
        doc = fitz.open(doc_path)
        total_pages = len(doc)
        sample_pages = min(num_pages, total_pages)
        
        # Start with first few pages as they often contain important content
        text_samples = []
        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text()
            if text.strip():  # Only add non-empty texts
                text_samples.append(text)
        
        # If we couldn't get text from regular extraction, it might be a scanned document
        if not text_samples:
            logger.warning(f"No text extracted from first {sample_pages} pages. Document may be scanned or image-based.")
            doc.close()
            return result
        
        combined_text = " ".join(text_samples)
        sample_size = len(combined_text)
        result["sample_size_chars"] = sample_size
        
        if sample_size < 100:
            logger.warning(f"Text sample too small ({sample_size} chars) for reliable language detection")
            doc.close()
            return result
            
        # Check if there's significant Arabic text using regex for Arabic Unicode range
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', combined_text))
        arabic_percentage = arabic_chars / sample_size if sample_size > 0 else 0
        
        # Count languages if langdetect is available
        language_counts = {}
        if HAS_LANGDETECT:
            # Split text into chunks for more robust detection
            chunk_size = 500
            chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
            
            detected_langs = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only process chunks with enough text
                    try:
                        lang = detect(chunk)
                        detected_langs.append(lang)
                    except LangDetectException:
                        pass  # Skip chunks where detection fails
            
            # Count languages detected
            language_counts = Counter(detected_langs)
            total_detections = sum(language_counts.values())
            
            # Calculate dominant language and confidence
            if total_detections > 0:
                dominant_lang, dominant_count = language_counts.most_common(1)[0]
                confidence = dominant_count / total_detections
                result["dominant_language"] = dominant_lang
                result["confidence"] = confidence
                
                # Sometimes ar (Arabic) is misdetected as other languages
                if arabic_percentage > 0.2:  # If >20% Arabic characters
                    result["dominant_language"] = "ar"
                    result["confidence"] = max(confidence, arabic_percentage)
        else:
            # Fallback if langdetect not available - rely on Arabic character detection
            if arabic_percentage > 0.2:
                result["dominant_language"] = "ar"
                result["confidence"] = arabic_percentage
        
        # Set Arabic flag based on dominant language or character percentage
        result["is_arabic"] = (result["dominant_language"] == "ar" or arabic_percentage > 0.2)
        result["arabic_char_percentage"] = arabic_percentage
        result["language_counts"] = language_counts
        
        doc.close()
        return result
        
    except Exception as e:
        logger.error(f"Error detecting document language: {e}")
        if 'doc' in locals():
            doc.close()
        return result


def chunk_text(text: str, max_chars: int = 4900) -> List[str]:
    """
    Split text into chunks of specified maximum size, respecting paragraph boundaries.
    
    Args:
        text: The text to chunk
        max_chars: Maximum characters per chunk (Azure limit is 5000, leaving buffer)
    
    Returns:
        List of text chunks
    """
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first (using double newline as separator)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # If paragraph alone exceeds limit, need to split it further
        if len(paragraph) > max_chars:
            # First save the current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split large paragraph by sentences or fallback to characters
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_sentence_chunk = ""
            
            for sentence in sentences:
                if len(current_sentence_chunk) + len(sentence) > max_chars:
                    if current_sentence_chunk:
                        chunks.append(current_sentence_chunk.strip())
                    
                    # If single sentence is too long, split by character
                    if len(sentence) > max_chars:
                        for i in range(0, len(sentence), max_chars):
                            chunks.append(sentence[i:i+max_chars].strip())
                    else:
                        current_sentence_chunk = sentence
                else:
                    current_sentence_chunk += " " + sentence if current_sentence_chunk else sentence
            
            if current_sentence_chunk:
                chunks.append(current_sentence_chunk.strip())
        
        # If adding this paragraph would exceed the limit, store current chunk and start new one
        elif len(current_chunk) + len(paragraph) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it contains anything
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def translate_text(text: str, source_lang: str = 'ar', target_lang: str = 'en', use_bidi: bool = True) -> str:
    """
    Translate text using Azure Translator service
    
    Args:
        text: The text to translate
        source_lang: The source language code
        target_lang: The target language code
        use_bidi: Whether to process Arabic text with bidi algorithm before translation
        
    Returns:
        Translated text
    """
    if not text.strip():
        return ""
        
    # Skip translation if source and target are the same
    if source_lang == target_lang:
        return text
    
    # Apply bidi processing for Arabic text if requested
    if use_bidi and source_lang == 'ar' and HAS_BIDI:
        logger.info("Applying bidi processing for Arabic text before translation")
        text = process_with_bidi(text)
    
    # If text is longer than max size, split into chunks
    chunks = chunk_text(text)
    
    # Translator settings
    key = os.environ.get('AZURE_TRANSLATOR_KEY')
    if not key:
        logger.error("Azure Translator API key not found in environment variables")
        return text
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = os.environ.get('AZURE_TRANSLATOR_LOCATION', 'global')
    constructed_url = endpoint + '/translate'
    params = {'api-version': '3.0', 'to': target_lang, 'from': source_lang}
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    @lru_cache(maxsize=512)
    def _cached_translate(single_chunk: str) -> str:
        try:
            r = requests.post(constructed_url, params=params, headers=headers, json=[{'text': single_chunk}])
            r.raise_for_status()
            return r.json()[0]['translations'][0]['text']
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation API error: {str(e)}")
            return single_chunk
    
    max_workers = int(os.environ.get('TRANSLATE_CONCURRENCY', '4'))
    results: List[str] = [""] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_cached_translate, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            idx = fut._args[1] if hasattr(fut, '_args') else futs[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = chunks[idx]
    
    return '\n\n'.join(results)


def translate_document_pages(doc_path: str, start_page: int = 0, end_page: int = 4, 
                           source_lang: str = 'ar', target_lang: str = 'en', use_bidi: bool = True) -> str:
    """
    Extract and translate text from specific pages of a document
    
    Args:
        doc_path: Path to the PDF document
        start_page: First page to translate (0-indexed)
        end_page: Last page to translate (inclusive, 0-indexed)
        source_lang: Source language code
        target_lang: Target language code
        use_bidi: Whether to process Arabic text with bidi algorithm before translation
        
    Returns:
        Translated text from the specified pages
    """
    try:
        doc = fitz.open(doc_path)
        total_pages = len(doc)
        
        # Adjust page range to be within document bounds
        start = max(0, start_page)
        end = min(total_pages - 1, end_page)
        
        if start > end or start >= total_pages:
            logger.warning(f"Invalid page range: {start}-{end} for document with {total_pages} pages")
            doc.close()
            return ""
        
        # Extract text from the specified pages
        pages_text = []
        for page_num in range(start, end + 1):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages_text.append(f"[PAGE {page_num + 1}]\n{text}")
        
        doc.close()
        
        # Combine and translate
        combined_text = "\n\n".join(pages_text)
        if not combined_text.strip():
            logger.warning(f"No text extracted from pages {start+1}-{end+1}")
            return ""
            
        translated_text = translate_text(combined_text, source_lang, target_lang, use_bidi=use_bidi)
        return translated_text
        
    except Exception as e:
        logger.error(f"Error translating document pages: {e}")
        if 'doc' in locals():
            doc.close()
        return ""


def enhance_vision_prompt_for_arabic(original_prompt: str, translated_text: str) -> str:
    """
    Enhance the vision analysis prompt with translated text for Arabic documents
    
    Args:
        original_prompt: The original vision analysis prompt
        translated_text: Translated text from the first few pages
        
    Returns:
        Enhanced prompt with translation information
    """
    prompt_addition = f"""
    IMPORTANT: This document contains Arabic text. I've provided a translation of the initial pages to assist your analysis:

    --- TRANSLATED TEXT ---
    {translated_text}
    --- END TRANSLATED TEXT ---

    When identifying policies:
    1. Extract policy names in their original Arabic form
    2. Use the images to help with structure and page numbers
    3. Use the translated text as reference to understand the correct policy name
    4. Include the policy type (law, decree, resolution, circular,other) in your analysis
    5. Document any uncertainty about policy boundaries or names
    
    For your JSON output, include the policy names in their original Arabic form.
    """
    
    # Add the prompt addition right after the initial instructions
    # Find a good insertion point (after the first paragraph)
    lines = original_prompt.split('\n')
    insertion_point = min(10, len(lines))  # Insert near the beginning
    
    enhanced_prompt = '\n'.join(lines[:insertion_point]) + prompt_addition + '\n'.join(lines[insertion_point:])
    return enhanced_prompt


def page_to_image(page) -> Image.Image:
    """Convert a PDF page to a PIL Image"""
    zoom = 2  # Increase resolution
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    
    # Convert to PIL Image
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return image


def get_document_info(doc_path: str) -> Dict[str, Any]:
    """
    Get basic information about a document
    
    Args:
        doc_path: Path to the PDF document
        
    Returns:
        Dictionary with document information
    """
    try:
        doc = fitz.open(doc_path)
        info = {
            "page_count": len(doc),
            "metadata": doc.metadata,
            "filename": os.path.basename(doc_path),
            "path": doc_path,
            "size_bytes": os.path.getsize(doc_path) if os.path.exists(doc_path) else 0
        }
        doc.close()
        return info
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        if 'doc' in locals():
            doc.close()
        return {
            "page_count": 0,
            "metadata": {},
            "filename": os.path.basename(doc_path),
            "path": doc_path,
            "size_bytes": 0,
            "error": str(e)
        }

# New utility functions added from b_smart_chunking.py

def is_blank_page(page_text: str) -> bool:
    """Check if a page is blank or near-blank"""
    # Remove whitespace
    cleaned_text = re.sub(r'\s+', '', page_text)
    # Consider blank if less than 50 non-whitespace characters
    return len(cleaned_text) < 50


def is_title_page(page_text: str) -> bool:
    """Check if a page is likely a title page"""
    # Title pages typically have limited text and specific patterns
    cleaned_text = page_text.strip()
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    
    # Title pages typically have few lines
    if len(lines) <= 5:
        # Look for year patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        if re.search(year_pattern, cleaned_text):
            return True
            
    # Check for "cover" or equivalent words
    cover_words = ['cover', 'title', 'غلاف', 'عنوان']
    for word in cover_words:
        if word.lower() in cleaned_text.lower():
            return True
            
    return False


def clean_filename(filename: str) -> str:
    """Create a clean, safe filename from a string"""
    if not filename:
        return "untitled"
        
    # Replace non-alphanumeric characters with underscores
    clean = re.sub(r'[^\w\s-]', '_', filename)
    
    # Replace multiple spaces with single underscore
    clean = re.sub(r'\s+', '_', clean)
    
    # Remove leading/trailing underscores
    clean = clean.strip('_')
    
    # Replace multiple underscores with a single one
    clean = re.sub(r'_+', '_', clean)
    
    # If the string is now empty, use default name
    if not clean:
        clean = "untitled"
    
    # Limit length
    if len(clean) > 50:
        clean = clean[:47] + "..."
        
    return clean


def detect_tables_in_content(content: str) -> bool:
    """Detect if content likely contains tables based on patterns"""
    # Simple detection based on patterns
    table_patterns = [
        r"\|\s*\w+\s*\|",  # |  Text  |
        r"\+[-+]+\+",      # +-----+-----+
        r"[^\|]+\|[^\|]+\|[^\|]+",  # Text | Text | Text
        r"[-]{3,}\s*\n.*\n[-]{3,}"  # ---- Header ----
    ]
    
    for pattern in table_patterns:
        if re.search(pattern, content):
            return True
    
    return False


def setup_azure_openai_client(async_client=True):
    """
    Set up the Azure OpenAI client
    
    Args:
        async_client: Whether to return an async client (True) or sync client (False)
        
    Returns:
        Azure OpenAI client (either async or sync)
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://admin-0752-resource.cognitiveservices.azure.com/")
    api_version = os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview")
    
    if async_client:
        # Import here to avoid circular imports
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    else:
        # Import here to avoid circular imports
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    
    return client 

def extract_page_text_with_bidi(doc_path: str, page_num: int) -> str:
    """
    Extract text from a PDF page and process it with the bidi algorithm for proper RTL display.
    
    Args:
        doc_path: Path to the PDF document
        page_num: Page number to extract (0-indexed)
        
    Returns:
        Extracted text processed with bidi algorithm
    """
    try:
        doc = fitz.open(doc_path)
        if page_num < 0 or page_num >= len(doc):
            logger.warning(f"Page {page_num} is out of range for document with {len(doc)} pages")
            doc.close()
            return ""
            
        page = doc[page_num]
        text = page.get_text()
        doc.close()
        
        # Process with bidi algorithm
        return process_with_bidi(text)
        
    except Exception as e:
        logger.error(f"Error extracting text with bidi: {e}")
        if 'doc' in locals():
            doc.close()
        return "" 

def detect_tables_with_img2table(doc_path: str, page_numbers: List[int] = None, early_exit: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """
    Detect tables in a PDF document using img2table library
    
    Args:
        doc_path: Path to the PDF document
        page_numbers: List of page numbers to analyze (0-indexed)
        early_exit: If True, stop processing after finding tables on any page
        
    Returns:
        Dictionary mapping page numbers to lists of detected tables
    """
    # If img2table is not available, return empty result
    if not HAS_IMG2TABLE:
        logger.warning("img2table not available, falling back to vision-only detection")
        return {}
    
    # If no page numbers provided, return empty result
    if not page_numbers:
        logger.warning("No page numbers provided for table detection")
        return {}
        
    # Limit the number of pages to process to avoid memory issues
    if len(page_numbers) > 10:
        logger.warning(f"Too many pages ({len(page_numbers)}) requested for table detection, limiting to first 10")
        page_numbers = page_numbers[:10]
    
    result = {}
    
    try:
        # Verify document exists and can be opened
        try:
            # Quick check if we can open the document
            with fitz.open(doc_path) as test_doc:
                total_pages = len(test_doc)
                logger.info(f"Document has {total_pages} pages total")
                
                # Filter page numbers to be within document bounds
                valid_page_numbers = [p for p in page_numbers if 0 <= p < total_pages]
                if len(valid_page_numbers) != len(page_numbers):
                    logger.warning(f"Filtered out {len(page_numbers) - len(valid_page_numbers)} invalid page numbers")
                    page_numbers = valid_page_numbers
                    
                if not page_numbers:
                    logger.warning("No valid pages to process after filtering")
                    return {}
        except Exception as e:
            logger.error(f"Error opening document: {e}")
            return {}
        
        # Process one page at a time to reduce memory usage
        for page_idx in page_numbers:
            # Convert to 1-indexed for img2table
            img2table_page = page_idx + 1
            
            logger.info(f"Detecting tables on page {img2table_page} with img2table")
            
            try:
                # Initialize OCR with Arabic and English support
                ocr = TesseractOCR(n_threads=1, lang="ara+eng")
                
                try:
                    # Load only this page of the document with page timeout
                    doc = PDF(doc_path, detect_rotation=True, pages=[img2table_page])
                    
                    # Extract tables with a timeout safety mechanism
                    tables = doc.extract_tables(
                        ocr=ocr,
                        implicit_rows=True,
                        implicit_columns=True,
                        borderless_tables=False,
                        min_confidence=80
                    )
                    
                    # Check if we found any tables on this page
                    if img2table_page in tables and tables[img2table_page]:
                        table_count = len(tables[img2table_page])
                        logger.info(f"Found {table_count} tables on page {img2table_page}")
                        
                        # Add minimal table information - just enough for bounding boxes
                        result[page_idx] = []
                        
                        for i, table in enumerate(tables[img2table_page]):
                            # Get bounding box
                            bbox = table.bbox
                            
                            # Convert numpy int values to regular Python integers if needed
                            if hasattr(bbox.x1, 'item'):
                                x0, y0, x1, y1 = bbox.x1.item(), bbox.y1.item(), bbox.x2.item(), bbox.y2.item()
                            else:
                                x0, y0, x1, y1 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                            
                            # Calculate a simple confidence score
                            confidence = 0.85  # Default high confidence since img2table already filtered
                            
                            # Just store the minimum required info for table extraction
                            table_info = {
                                "bbox": (x0, y0, x1, y1),
                                "confidence": confidence,
                                "table_id": i
                            }
                            
                            result[page_idx].append(table_info)
                        
                        # Exit early if we found tables and early_exit is enabled
                        if early_exit and result:
                            logger.info(f"Early exit after finding tables on page {img2table_page}")
                            break
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {img2table_page}: {e}")
                    continue
            
            except Exception as e:
                logger.warning(f"Error processing page {img2table_page}: {e}")
                continue
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting tables with img2table: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def has_tables_in_pdf(doc_path: str, page_numbers: List[int] = None) -> bool:
    """
    Quick check if a PDF document has any tables on the specified pages.
    More memory efficient than full detection.
    
    Args:
        doc_path: Path to the PDF document
        page_numbers: List of page numbers to analyze (0-indexed)
        
    Returns:
        Boolean indicating if tables were detected
    """
    # Process in small batches to reduce memory usage
    batch_size = 5
    page_batches = []
    
    if page_numbers:
        # Split page numbers into batches
        for i in range(0, len(page_numbers), batch_size):
            page_batches.append(page_numbers[i:i+batch_size])
    else:
        # If no page numbers provided, check first 10 pages
        page_batches = [[i for i in range(10)]]
    
    for batch in page_batches:
        # Try img2table first with early exit
        tables = detect_tables_with_img2table(doc_path, batch, early_exit=True)
        if tables:
            return True
        
        # Try original method if available
        try:
            from a_preprocessing import detect_tables_in_pdf as original_detect_tables
            tables = original_detect_tables(doc_path, batch)
            if any(tables.values()):
                return True
        except ImportError:
            pass
    
    return False

# Add a combined function that will try img2table first, then fall back if needed
def detect_tables_in_pdf(doc_path: str, page_numbers: List[int] = None) -> Dict[int, List[Dict[str, Any]]]:
    """
    Detect tables in a PDF document, using img2table if available
    
    Args:
        doc_path: Path to the PDF document
        page_numbers: List of page numbers to analyze (0-indexed)
        
    Returns:
        Dictionary mapping page numbers to lists of detected tables
    """
    # Try img2table first
    tables = detect_tables_with_img2table(doc_path, page_numbers, early_exit=False)
    
    # If we got results, return them
    if tables:
        return tables
    
    # Otherwise, try to use the original method from a_preprocessing if available
    try:
        from a_preprocessing import detect_tables_in_pdf as original_detect_tables
        logger.info("Falling back to original table detection from a_preprocessing")
        return original_detect_tables(doc_path, page_numbers)
    except ImportError:
        logger.warning("Original table detection not available either")
        
        # Return empty results for the requested pages
        if page_numbers:
            return {page: [] for page in page_numbers}
        else:
            return {}

# Helper function to check if a table is valid
def is_valid_table(table: Dict[str, Any]) -> bool:
    """Check if a detected table is valid based on confidence and size"""
    # Check confidence threshold
    if table.get("confidence", 0) < 0.7:
        return False
        
    # Check minimum dimensions
    bbox = table.get("bbox", (0, 0, 0, 0))
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Minimum size check
    min_size = 50  # pixels
    if width < min_size or height < min_size:
        return False
        
    return True 