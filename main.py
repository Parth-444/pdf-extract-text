"""
Invoice Extraction Service v3.0 - PyMuPDF + Tesseract OCR

IMPROVEMENTS OVER v2.0:
1. Uses PyMuPDF's built-in find_tables() instead of custom Y-coordinate grouping
2. Automatic strategy selection based on page type (native vs OCR)
3. Better header detection (automatic, not keyword-based)
4. Handles merged cells and multi-line cell content
5. Fallback to custom extraction if find_tables() fails
6. Improved amount filtering to reduce noise
7. Confidence scoring for extraction quality

This service extracts:
- Raw text (exact characters)
- All GSTINs found (validated format)
- All dates found (multiple formats)
- All amounts found (filtered for likely monetary values)
- Table structures (using find_tables() with fallback)
- Page metadata (text-based vs image-based)
- Confidence scores for LLM guidance

LLM receives this data + image and handles:
- Document classification
- Party identification (which GSTIN is ours vs vendor)
- Field mapping
- Business rules
"""

import os
import re
import base64
import logging
from typing import Optional
import time

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract data path
TESSDATA_PREFIX = os.environ.get("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/5/tessdata")
if os.path.exists(TESSDATA_PREFIX):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

app = FastAPI(
    title="Invoice Extraction Service",
    description="Accurate text extraction from PDFs using PyMuPDF + Tesseract OCR with built-in table detection",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Pydantic Models
# ============================================

class ExtractRequest(BaseModel):
    pdf_base64: str
    ocr_if_needed: bool = True
    ocr_dpi: int = 300
    ocr_language: str = "eng"


class PageInfo(BaseModel):
    page_number: int
    page_type: str  # TEXT_BASED, IMAGE_BASED
    extraction_method: str  # NATIVE, OCR
    char_count: int
    has_vector_graphics: bool  # New: indicates if page has lines/rectangles


class TableRow(BaseModel):
    row_index: int
    cells: list[str]
    y_position: float


class ExtractedTable(BaseModel):
    page: int
    headers: list[str]
    rows: list[TableRow]
    bbox: tuple[float, float, float, float]  # New: table bounding box
    row_count: int
    col_count: int
    extraction_method: str  # FIND_TABLES, FALLBACK_CUSTOM
    confidence: str  # HIGH, MEDIUM, LOW


class ConfidenceScores(BaseModel):
    gstins: str  # HIGH, MEDIUM, LOW
    dates: str
    amounts: str
    tables: str
    overall: str


class ExtractionResult(BaseModel):
    status: str
    extraction_mode: str  # FULL (all native), PARTIAL (some OCR), OCR_ONLY
    
    # Page info
    pages: list[PageInfo]
    total_pages: int
    
    # Extracted values (no interpretation)
    gstins: list[str]
    dates: list[str]
    amounts: list[str]
    
    # Table data
    tables: list[ExtractedTable]
    
    # Raw text
    raw_text: str
    text_by_page: list[str]
    
    # Confidence scores
    confidence: ConfidenceScores
    
    # For LLM
    llm_guidance: str
    
    processing_time_ms: int


# ============================================
# Patterns
# ============================================

# GSTIN: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric
GSTIN_PATTERN = re.compile(r'\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b')

# Date patterns
DATE_PATTERNS = [
    re.compile(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b'),  # DD/MM/YYYY or MM/DD/YYYY
    re.compile(r'\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b'),  # YYYY/MM/DD
    re.compile(r'\b\d{1,2}[\s\-]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-,]?\d{4}\b', re.IGNORECASE),
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-,]?\d{1,2}[\s\-,]?\d{4}\b', re.IGNORECASE),
]

# Table header keywords (used for fallback method)
TABLE_HEADERS = [
    's.no', 'sr.no', 'sl.no', 'sno', 'sr', 'sl',
    'description', 'particulars', 'item', 'product', 'goods',
    'hsn', 'sac', 'hsn/sac',
    'qty', 'quantity', 'units', 'nos',
    'rate', 'price', 'unit price', 'mrp',
    'amount', 'amt', 'total', 'value',
    'igst', 'cgst', 'sgst', 'gst', 'tax',
    'taxable', 'taxable value', 'assessable'
]


# ============================================
# Validation Functions
# ============================================

def validate_gstin(gstin: str) -> bool:
    """Validate GSTIN format and checksum (basic validation)"""
    if not GSTIN_PATTERN.match(gstin):
        return False
    # Basic format is valid
    # Full checksum validation could be added here
    return True


def is_likely_amount(value_str: str, amount_float: float) -> bool:
    """
    Filter amounts to reduce false positives.
    Returns True if the value is likely a monetary amount.
    """
    # Too small - probably not a monetary amount
    if amount_float < 1.0:
        return False
    
    # Too large - probably not realistic for most invoices (100 crore limit)
    if amount_float > 1_000_000_000:
        return False
    
    # Check for percentage indicators in context
    # (This is a basic check - context would need to be passed for full accuracy)
    
    return True


# ============================================
# Page Analysis Functions
# ============================================

def detect_page_type(page: fitz.Page) -> tuple[str, int, bool]:
    """
    Detect if page is text-based or image-based, and if it has vector graphics.
    
    Returns: (page_type, char_count, has_vector_graphics)
    """
    text = page.get_text("text").strip()
    char_count = len(text)
    
    # Check for vector graphics (lines, rectangles) that could define table structure
    try:
        drawings = page.get_drawings()
        has_vector_graphics = len(drawings) > 10  # Arbitrary threshold
    except Exception:
        has_vector_graphics = False
    
    if char_count < 50:
        return "IMAGE_BASED", char_count, has_vector_graphics
    return "TEXT_BASED", char_count, has_vector_graphics


def extract_text_native(page: fitz.Page) -> str:
    """Native text extraction"""
    return page.get_text("text")


def extract_text_ocr(page: fitz.Page, dpi: int = 300, language: str = "eng") -> str:
    """OCR text extraction using Tesseract via PyMuPDF"""
    try:
        tp = page.get_textpage_ocr(dpi=dpi, language=language, full=True)
        return page.get_text(textpage=tp)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


# ============================================
# Table Extraction Functions
# ============================================

def extract_tables_native(
    page: fitz.Page, 
    page_num: int, 
    page_type: str,
    has_vector_graphics: bool
) -> list[ExtractedTable]:
    """
    Use PyMuPDF's built-in find_tables() method.
    
    Strategy selection:
    - For native text PDFs with vector graphics: "lines_strict"
    - For native text PDFs without vector graphics: "text"  
    - For OCR'd/scanned PDFs: "text"
    """
    try:
        # Choose strategy based on page characteristics
        if page_type == "IMAGE_BASED" or not has_vector_graphics:
            # OCR'd pages or pages without grid lines - use text positioning
            strategy = "text"
            min_words_v = 2
            min_words_h = 1
        else:
            # Native PDFs with vector graphics - use lines_strict to avoid 
            # false positives from background colors
            strategy = "lines_strict"
            min_words_v = 3
            min_words_h = 1
        
        logger.info(f"Page {page_num}: Using strategy='{strategy}' "
                   f"(page_type={page_type}, has_vector_graphics={has_vector_graphics})")
        
        tabs = page.find_tables(
            strategy=strategy,
            snap_tolerance=3,
            join_tolerance=3,
            edge_min_length=3,
            min_words_vertical=min_words_v,
            min_words_horizontal=min_words_h
        )
        
        if tabs is None or not tabs.tables:
            # Try fallback with different strategy
            logger.info(f"Page {page_num}: No tables found with '{strategy}', trying 'text' strategy")
            tabs = page.find_tables(
                strategy="text",
                min_words_vertical=2,
                min_words_horizontal=1
            )
        
        if tabs is None or not tabs.tables:
            return []
        
        result = []
        for table in tabs.tables:
            # Get header names (PyMuPDF detects these automatically)
            if table.header and table.header.names:
                headers = [str(name) if name else f"Col{i}" 
                          for i, name in enumerate(table.header.names)]
            else:
                headers = [f"Col{i}" for i in range(table.col_count)]
            
            # Extract rows
            rows = []
            try:
                extracted = table.extract()
            except Exception as e:
                logger.warning(f"Table extract() failed: {e}")
                continue
            
            if not extracted:
                continue
            
            # Determine if header is internal to table
            start_idx = 0
            if table.header and hasattr(table.header, 'external'):
                start_idx = 0 if table.header.external else 1
            
            for idx, row_data in enumerate(extracted[start_idx:], start=1):
                # Filter out None cells and convert to strings
                cells = [str(cell).strip() if cell else "" for cell in row_data]
                
                # Skip empty rows
                if not any(cells):
                    continue
                
                # Get Y position from table.rows if available
                y_pos = 0.0
                if hasattr(table, 'rows') and idx - 1 < len(table.rows):
                    try:
                        y_pos = table.rows[idx - 1].bbox[1]
                    except (AttributeError, IndexError):
                        pass
                
                rows.append(TableRow(
                    row_index=idx,
                    cells=cells,
                    y_position=y_pos
                ))
            
            if rows:
                # Determine confidence based on extraction quality
                confidence = "HIGH"
                if strategy == "text":
                    confidence = "MEDIUM"
                if len(rows) < 2:
                    confidence = "LOW"
                
                result.append(ExtractedTable(
                    page=page_num,
                    headers=headers,
                    rows=rows,
                    bbox=tuple(table.bbox),
                    row_count=len(rows),
                    col_count=table.col_count,
                    extraction_method="FIND_TABLES",
                    confidence=confidence
                ))
        
        return result
        
    except Exception as e:
        logger.warning(f"Native table detection failed on page {page_num}: {e}")
        return []


def extract_tables_fallback(
    page: fitz.Page, 
    page_num: int,
    use_ocr: bool = False,
    ocr_dpi: int = 300,
    ocr_language: str = "eng"
) -> list[ExtractedTable]:
    """
    Fallback table extraction using Y-coordinate grouping.
    Used when find_tables() returns no results.
    """
    try:
        # Get text with positions
        if use_ocr:
            try:
                tp = page.get_textpage_ocr(dpi=ocr_dpi, language=ocr_language, full=True)
                blocks = page.get_text("dict", textpage=tp)["blocks"]
            except Exception:
                blocks = page.get_text("dict")["blocks"]
        else:
            blocks = page.get_text("dict")["blocks"]
        
        spans = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            spans.append({
                                "text": text,
                                "x": bbox[0],
                                "y": bbox[1],
                            })
        
        if not spans:
            return []
        
        # Sort by Y then X
        sorted_spans = sorted(spans, key=lambda s: (s["y"], s["x"]))
        
        # Group into rows by Y-coordinate (within tolerance)
        rows_data = []
        current_row = []
        current_y = None
        y_tolerance = 5
        
        for span in sorted_spans:
            y = span["y"]
            
            if current_y is None:
                current_y = y
                current_row = [span]
            elif abs(y - current_y) <= y_tolerance:
                current_row.append(span)
            else:
                if current_row:
                    row_texts = [s["text"] for s in sorted(current_row, key=lambda s: s["x"])]
                    rows_data.append({"y": current_y, "cells": row_texts})
                current_y = y
                current_row = [span]
        
        if current_row:
            row_texts = [s["text"] for s in sorted(current_row, key=lambda s: s["x"])]
            rows_data.append({"y": current_y, "cells": row_texts})
        
        # Find header row using keywords
        header_idx = None
        for idx, row in enumerate(rows_data):
            row_text = " ".join(row["cells"]).lower()
            matches = sum(1 for kw in TABLE_HEADERS if kw in row_text)
            if matches >= 2:  # Reduced threshold from 3 to 2
                header_idx = idx
                break
        
        if header_idx is None:
            return []
        
        # Build table
        headers = rows_data[header_idx]["cells"]
        table_rows = []
        
        min_y = rows_data[header_idx]["y"]
        max_y = min_y
        
        for idx in range(header_idx + 1, len(rows_data)):
            row = rows_data[idx]
            if len(row["cells"]) >= 2:  # Minimum columns for a valid row
                table_rows.append(TableRow(
                    row_index=len(table_rows) + 1,
                    cells=row["cells"],
                    y_position=row["y"]
                ))
                max_y = row["y"]
                
                # Stop at totals row
                row_text = " ".join(row["cells"]).lower()
                if "total" in row_text and "taxable" not in row_text and "sub" not in row_text:
                    break
        
        if table_rows:
            # Estimate bbox
            page_rect = page.rect
            bbox = (0, min_y, page_rect.width, max_y + 20)
            
            return [ExtractedTable(
                page=page_num,
                headers=headers,
                rows=table_rows,
                bbox=bbox,
                row_count=len(table_rows),
                col_count=len(headers),
                extraction_method="FALLBACK_CUSTOM",
                confidence="LOW"
            )]
        
        return []
        
    except Exception as e:
        logger.error(f"Fallback table extraction failed on page {page_num}: {e}")
        return []


def extract_tables(
    page: fitz.Page,
    page_num: int,
    page_type: str,
    has_vector_graphics: bool,
    use_ocr: bool = False,
    ocr_dpi: int = 300,
    ocr_language: str = "eng"
) -> list[ExtractedTable]:
    """
    Main table extraction function with automatic fallback.
    
    1. Try PyMuPDF's find_tables() first
    2. Fall back to custom Y-coordinate method if no tables found
    """
    # Try native find_tables() first
    tables = extract_tables_native(page, page_num, page_type, has_vector_graphics)
    
    if tables:
        logger.info(f"Page {page_num}: Found {len(tables)} table(s) using find_tables()")
        return tables
    
    # Fallback to custom extraction
    logger.info(f"Page {page_num}: Trying fallback table extraction")
    tables = extract_tables_fallback(page, page_num, use_ocr, ocr_dpi, ocr_language)
    
    if tables:
        logger.info(f"Page {page_num}: Found {len(tables)} table(s) using fallback method")
    else:
        logger.info(f"Page {page_num}: No tables found")
    
    return tables


# ============================================
# Value Extraction Functions
# ============================================

def find_gstins(text: str) -> list[str]:
    """Find all valid GSTINs in text"""
    matches = GSTIN_PATTERN.findall(text)
    # Validate and deduplicate
    valid_gstins = []
    seen = set()
    for gstin in matches:
        if gstin not in seen and validate_gstin(gstin):
            valid_gstins.append(gstin)
            seen.add(gstin)
    return valid_gstins


def find_dates(text: str) -> list[str]:
    """Find all dates in text"""
    dates = []
    seen = set()
    for pattern in DATE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            if match not in seen:
                dates.append(match)
                seen.add(match)
    return dates


def find_amounts(text: str) -> list[str]:
    """
    Find monetary amounts with improved filtering to reduce noise.
    """
    amounts = []
    seen = set()
    
    # Pattern 1: Explicit currency markers (high confidence)
    currency_pattern = re.compile(r'(?:INR|Rs\.?|₹)\s*([\d,]+\.?\d*)', re.IGNORECASE)
    for match in currency_pattern.finditer(text):
        full_match = match.group(0)
        if full_match not in seen:
            amounts.append(full_match)
            seen.add(full_match)
    
    # Pattern 2: Numbers with exactly 2 decimal places (likely monetary)
    decimal_pattern = re.compile(r'\b([\d,]{1,15}\.\d{2})\b')
    for match in decimal_pattern.finditer(text):
        value_str = match.group(1)
        if value_str in seen:
            continue
        
        try:
            value = float(value_str.replace(',', ''))
            
            # Filter out unlikely amounts
            if not is_likely_amount(value_str, value):
                continue
            
            # Check context to avoid percentages (look at characters before match)
            start = max(0, match.start() - 10)
            context_before = text[start:match.start()].lower()
            
            # Skip if preceded by percentage indicators
            if '%' in context_before or 'percent' in context_before or '@' in context_before:
                continue
            
            amounts.append(value_str)
            seen.add(value_str)
            
        except ValueError:
            continue
    
    # Pattern 3: Large round numbers (likely totals)
    round_pattern = re.compile(r'\b([\d,]{4,12})(?:\.00)?\b')
    for match in round_pattern.finditer(text):
        value_str = match.group(1)
        if value_str in seen:
            continue
        
        try:
            value = float(value_str.replace(',', ''))
            if value >= 100 and value <= 10_000_000:  # 100 to 1 crore
                # Only add if it looks like a standalone amount
                amounts.append(value_str)
                seen.add(value_str)
        except ValueError:
            continue
    
    # Limit to top 30 amounts to avoid noise
    return amounts[:30]


# ============================================
# Confidence Scoring
# ============================================

def calculate_confidence(
    gstins: list[str],
    dates: list[str],
    amounts: list[str],
    tables: list[ExtractedTable],
    extraction_mode: str
) -> ConfidenceScores:
    """Calculate confidence scores for each extraction category"""
    
    # GSTIN confidence
    gstin_conf = "LOW"
    if len(gstins) >= 2:
        gstin_conf = "HIGH"
    elif len(gstins) == 1:
        gstin_conf = "MEDIUM"
    
    # Date confidence
    date_conf = "LOW"
    if len(dates) >= 1:
        date_conf = "HIGH" if extraction_mode == "FULL" else "MEDIUM"
    
    # Amount confidence  
    amount_conf = "MEDIUM"  # Always medium due to potential over-matching
    if len(amounts) == 0:
        amount_conf = "LOW"
    
    # Table confidence
    table_conf = "LOW"
    if tables:
        # Use the highest confidence from any table
        table_confs = [t.confidence for t in tables]
        if "HIGH" in table_confs:
            table_conf = "HIGH"
        elif "MEDIUM" in table_confs:
            table_conf = "MEDIUM"
    
    # Overall confidence
    scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    total = scores[gstin_conf] + scores[date_conf] + scores[amount_conf] + scores[table_conf]
    
    if total >= 10:
        overall = "HIGH"
    elif total >= 6:
        overall = "MEDIUM"
    else:
        overall = "LOW"
    
    return ConfidenceScores(
        gstins=gstin_conf,
        dates=date_conf,
        amounts=amount_conf,
        tables=table_conf,
        overall=overall
    )


# ============================================
# Main Extraction Function
# ============================================

def extract_from_pdf(
    pdf_bytes: bytes,
    ocr_if_needed: bool = True,
    ocr_dpi: int = 300,
    ocr_language: str = "eng"
) -> ExtractionResult:
    """Main extraction function"""
    start = time.time()
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return ExtractionResult(
            status="error",
            extraction_mode="FAILED",
            pages=[],
            total_pages=0,
            gstins=[],
            dates=[],
            amounts=[],
            tables=[],
            raw_text="",
            text_by_page=[],
            confidence=ConfidenceScores(
                gstins="LOW", dates="LOW", amounts="LOW", tables="LOW", overall="LOW"
            ),
            llm_guidance=f"PDF could not be opened: {e}",
            processing_time_ms=int((time.time() - start) * 1000)
        )
    
    pages_info = []
    all_text = []
    all_tables = []
    ocr_pages = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_type, char_count, has_vector_graphics = detect_page_type(page)
        
        # Decide extraction method
        use_ocr = page_type == "IMAGE_BASED" and ocr_if_needed
        if use_ocr:
            ocr_pages += 1
        
        extraction_method = "OCR" if use_ocr else "NATIVE"
        
        # Extract text
        if use_ocr:
            text = extract_text_ocr(page, ocr_dpi, ocr_language)
            if not text.strip():
                text = extract_text_native(page)
                extraction_method = "NATIVE_FALLBACK"
        else:
            text = extract_text_native(page)
        
        all_text.append(text)
        
        # Extract tables using the improved method
        tables = extract_tables(
            page=page,
            page_num=page_num + 1,
            page_type=page_type,
            has_vector_graphics=has_vector_graphics,
            use_ocr=use_ocr,
            ocr_dpi=ocr_dpi,
            ocr_language=ocr_language
        )
        all_tables.extend(tables)
        
        pages_info.append(PageInfo(
            page_number=page_num + 1,
            page_type=page_type,
            extraction_method=extraction_method,
            char_count=len(text.strip()),
            has_vector_graphics=has_vector_graphics
        ))
    
    doc.close()
    
    # Combine all text
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
    
    # Extract values
    gstins = find_gstins(full_text)
    dates = find_dates(full_text)
    amounts = find_amounts(full_text)
    
    # Determine extraction mode
    if ocr_pages == 0:
        mode = "FULL"
    elif ocr_pages == len(pages_info):
        mode = "OCR_ONLY"
    else:
        mode = "PARTIAL"
    
    # Calculate confidence scores
    confidence = calculate_confidence(gstins, dates, amounts, all_tables, mode)
    
    # Build LLM guidance
    guidance_parts = []
    
    if mode == "FULL":
        guidance_parts.append("All pages have native text layer - extraction is highly reliable.")
    elif mode == "OCR_ONLY":
        guidance_parts.append(f"All {len(pages_info)} pages required OCR - verify critical values against image.")
    else:
        guidance_parts.append(f"{ocr_pages} of {len(pages_info)} pages required OCR.")
    
    if all_tables:
        table_methods = set(t.extraction_method for t in all_tables)
        table_confs = set(t.confidence for t in all_tables)
        guidance_parts.append(
            f"Found {len(all_tables)} table(s) using {', '.join(table_methods)}. "
            f"Confidence: {', '.join(table_confs)}."
        )
    else:
        guidance_parts.append("No tables detected - line items may need extraction from raw text.")
    
    if confidence.overall == "LOW":
        guidance_parts.append("⚠️ Overall confidence is LOW - recommend manual verification.")
    
    llm_guidance = " ".join(guidance_parts)
    
    return ExtractionResult(
        status="success",
        extraction_mode=mode,
        pages=pages_info,
        total_pages=len(pages_info),
        gstins=gstins,
        dates=dates,
        amounts=amounts,
        tables=all_tables,
        raw_text=full_text,
        text_by_page=all_text,
        confidence=confidence,
        llm_guidance=llm_guidance,
        processing_time_ms=int((time.time() - start) * 1000)
    )


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {
        "service": "Invoice Extraction Service",
        "version": "3.0.0",
        "features": [
            "PyMuPDF find_tables() with automatic strategy selection",
            "Fallback to Y-coordinate grouping when needed",
            "Confidence scoring for extraction quality",
            "Improved amount filtering",
            "GSTIN validation"
        ],
        "note": "This service extracts raw values. LLM handles interpretation."
    }


@app.get("/health")
async def health():
    tesseract_ok = os.path.exists(TESSDATA_PREFIX)
    return {
        "status": "healthy",
        "pymupdf_version": fitz.version[0],
        "tesseract_available": tesseract_ok,
        "find_tables_available": hasattr(fitz.Page, "find_tables")
    }


@app.post("/extract", response_model=ExtractionResult)
async def extract_base64(request: ExtractRequest):
    """Extract from base64-encoded PDF"""
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    
    return extract_from_pdf(
        pdf_bytes,
        request.ocr_if_needed,
        request.ocr_dpi,
        request.ocr_language
    )


@app.post("/extract-file", response_model=ExtractionResult)
async def extract_file(
    file: UploadFile = File(...),
    ocr_if_needed: bool = True,
    ocr_dpi: int = 300,
    ocr_language: str = "eng"
):
    """Extract from uploaded PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await file.read()
    return extract_from_pdf(pdf_bytes, ocr_if_needed, ocr_dpi, ocr_language)


@app.post("/extract-for-llm")
async def extract_for_llm(request: ExtractRequest):
    """
    Extract and format for LLM consumption.
    Returns structured data + formatted prompt text.
    """
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    
    result = extract_from_pdf(
        pdf_bytes,
        request.ocr_if_needed,
        request.ocr_dpi,
        request.ocr_language
    )
    
    # Format for LLM
    lines = [
        "=" * 60,
        "OCR EXTRACTED DATA (PRIMARY SOURCE - use these exact values)",
        "=" * 60,
        "",
        f"Extraction Mode: {result.extraction_mode}",
        f"Overall Confidence: {result.confidence.overall}",
        f"Guidance: {result.llm_guidance}",
        "",
        "GSTINS FOUND (use exactly as shown):",
    ]
    
    if result.gstins:
        for i, gstin in enumerate(result.gstins, 1):
            lines.append(f"  {i}. {gstin}")
    else:
        lines.append("  (none found)")
    
    lines.extend(["", "DATES FOUND:"])
    if result.dates:
        for date in result.dates:
            lines.append(f"  - {date}")
    else:
        lines.append("  (none found)")
    
    lines.extend(["", f"AMOUNTS FOUND (confidence: {result.confidence.amounts}):"])
    if result.amounts:
        for amount in result.amounts[:20]:
            lines.append(f"  - {amount}")
        if len(result.amounts) > 20:
            lines.append(f"  ... and {len(result.amounts) - 20} more")
    else:
        lines.append("  (none found)")
    
    if result.tables:
        lines.extend(["", f"LINE ITEMS TABLES (confidence: {result.confidence.tables}):"])
        for table in result.tables:
            lines.append(f"\n  Table on page {table.page} ({table.extraction_method}, {table.confidence}):")
            lines.append(f"  Headers: {' | '.join(table.headers)}")
            for row in table.rows:
                lines.append(f"  ROW {row.row_index}: {' | '.join(row.cells)}")
    else:
        lines.extend(["", "NO TABLES DETECTED"])
    
    lines.extend(["", "=" * 60])
    
    return {
        "extraction_result": result,
        "llm_prompt_text": "\n".join(lines)
    }


# ============================================
# Debug Endpoint (Development Only)
# ============================================

@app.post("/debug/table-detection")
async def debug_table_detection(request: ExtractRequest):
    """
    Debug endpoint to see detailed table detection info.
    Shows what find_tables() returns at each step.
    """
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    debug_info = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_type, char_count, has_vector_graphics = detect_page_type(page)
        
        page_debug = {
            "page": page_num + 1,
            "page_type": page_type,
            "char_count": char_count,
            "has_vector_graphics": has_vector_graphics,
            "strategies_tried": []
        }
        
        # Try different strategies
        for strategy in ["lines_strict", "lines", "text"]:
            try:
                tabs = page.find_tables(strategy=strategy)
                tables_found = len(tabs.tables) if tabs and tabs.tables else 0
                
                table_info = []
                if tabs and tabs.tables:
                    for t in tabs.tables:
                        table_info.append({
                            "bbox": t.bbox,
                            "rows": t.row_count,
                            "cols": t.col_count,
                            "headers": t.header.names if t.header else None
                        })
                
                page_debug["strategies_tried"].append({
                    "strategy": strategy,
                    "tables_found": tables_found,
                    "tables": table_info
                })
            except Exception as e:
                page_debug["strategies_tried"].append({
                    "strategy": strategy,
                    "error": str(e)
                })
        
        debug_info.append(page_debug)
    
    doc.close()
    
    return {"debug_info": debug_info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)