"""
Invoice Extraction Service - PyMuPDF + Tesseract OCR

PURPOSE: Extract text accurately from PDFs. Leave interpretation to LLM.

This service extracts:
- Raw text (exact characters)
- All GSTINs found (values only)
- All dates found (values only)
- All amounts found (values only)
- Table structures (rows/columns)
- Page metadata (text-based vs image-based)

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
    description="Accurate text extraction from PDFs using PyMuPDF + Tesseract OCR",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Pydantic Models (Simplified)
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


class TableRow(BaseModel):
    row_index: int
    cells: list[str]
    y_position: float


class ExtractedTable(BaseModel):
    page: int
    headers: list[str]
    rows: list[TableRow]


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
    re.compile(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b'),  # DD/MM/YYYY
    re.compile(r'\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b'),  # YYYY/MM/DD
    re.compile(r'\b\d{1,2}[\s\-]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]?\d{4}\b', re.IGNORECASE),
]

# Amount patterns
AMOUNT_PATTERNS = [
    re.compile(r'(?:INR|Rs\.?|₹)\s*([\d,]+\.?\d*)', re.IGNORECASE),
    re.compile(r'([\d,]+\.\d{2})\b'),  # Numbers with 2 decimal places
]

# Table header keywords
TABLE_HEADERS = ['s.no', 'sr.no', 'description', 'hsn', 'sac', 'qty', 'quantity', 
                 'rate', 'amount', 'total', 'igst', 'cgst', 'sgst', 'tax', 'value']


# ============================================
# Extraction Functions
# ============================================

def detect_page_type(page: fitz.Page) -> tuple[str, int]:
    """Detect if page is text-based or image-based"""
    text = page.get_text("text").strip()
    char_count = len(text)
    
    if char_count < 50:
        return "IMAGE_BASED", char_count
    return "TEXT_BASED", char_count


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


def extract_text_with_positions(page: fitz.Page, use_ocr: bool = False,
                                  dpi: int = 300, language: str = "eng") -> list[dict]:
    """Extract text spans with bounding boxes"""
    try:
        if use_ocr:
            tp = page.get_textpage_ocr(dpi=dpi, language=language, full=True)
            blocks = page.get_text("dict", textpage=tp)["blocks"]
        else:
            blocks = page.get_text("dict")["blocks"]
    except Exception:
        blocks = page.get_text("dict")["blocks"]
    
    spans = []
    for block in blocks:
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        spans.append({
                            "text": text,
                            "x": span.get("bbox", [0])[0],
                            "y": span.get("bbox", [0, 0])[1],
                        })
    return spans


def find_gstins(text: str) -> list[str]:
    """Find all GSTINs in text"""
    matches = GSTIN_PATTERN.findall(text)
    return list(dict.fromkeys(matches))  # Remove duplicates, preserve order


def find_dates(text: str) -> list[str]:
    """Find all dates in text"""
    dates = []
    for pattern in DATE_PATTERNS:
        matches = pattern.findall(text)
        dates.extend(matches)
    return list(dict.fromkeys(dates))


def find_amounts(text: str) -> list[str]:
    """Find all monetary amounts in text"""
    amounts = []
    for pattern in AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            amounts.append(match.group(0))
    return list(dict.fromkeys(amounts))


def extract_tables(page: fitz.Page, spans: list[dict], page_num: int) -> list[ExtractedTable]:
    """Extract tables by analyzing text positions"""
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
        text = span["text"]
        
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
    
    # Find header row
    header_idx = None
    for idx, row in enumerate(rows_data):
        row_text = " ".join(row["cells"]).lower()
        matches = sum(1 for kw in TABLE_HEADERS if kw in row_text)
        if matches >= 3:
            header_idx = idx
            break
    
    if header_idx is None:
        return []
    
    # Build table
    headers = rows_data[header_idx]["cells"]
    table_rows = []
    
    for idx in range(header_idx + 1, len(rows_data)):
        row = rows_data[idx]
        if len(row["cells"]) >= 3:  # Minimum columns for a valid row
            table_rows.append(TableRow(
                row_index=len(table_rows) + 1,
                cells=row["cells"],
                y_position=row["y"]
            ))
            
            # Stop at totals row
            row_text = " ".join(row["cells"]).lower()
            if "total" in row_text and "taxable" not in row_text:
                break
    
    if table_rows:
        return [ExtractedTable(page=page_num, headers=headers, rows=table_rows)]
    return []


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
            llm_guidance=f"PDF could not be opened: {e}",
            processing_time_ms=int((time.time() - start) * 1000)
        )
    
    pages_info = []
    all_text = []
    all_tables = []
    ocr_pages = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_type, char_count = detect_page_type(page)
        
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
        
        # Extract tables
        spans = extract_text_with_positions(page, use_ocr, ocr_dpi, ocr_language)
        tables = extract_tables(page, spans, page_num + 1)
        all_tables.extend(tables)
        
        pages_info.append(PageInfo(
            page_number=page_num + 1,
            page_type=page_type,
            extraction_method=extraction_method,
            char_count=len(text.strip())
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
        guidance = "All pages have text layer. OCR data is highly accurate. Use extracted values as primary source."
    elif ocr_pages == len(pages_info):
        mode = "OCR_ONLY"
        guidance = "All pages are image-based (scanned). OCR was used. Verify critical values against image if needed."
    else:
        mode = "PARTIAL"
        guidance = f"{ocr_pages} of {len(pages_info)} pages required OCR. Most data is accurate."
    
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
        llm_guidance=guidance,
        processing_time_ms=int((time.time() - start) * 1000)
    )


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {
        "service": "Invoice Extraction Service",
        "version": "2.0.0",
        "purpose": "Accurate text extraction from PDFs",
        "note": "This service extracts raw values. LLM handles interpretation."
    }


@app.get("/health")
async def health():
    tesseract_ok = os.path.exists(TESSDATA_PREFIX)
    return {
        "status": "healthy",
        "pymupdf_version": fitz.version[0],
        "tesseract_available": tesseract_ok
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
        "OCR EXTRACTED DATA (use these exact values - do not re-read from image)",
        "=" * 60,
        "",
        f"Extraction Mode: {result.extraction_mode}",
        f"Guidance: {result.llm_guidance}",
        "",
        "GSTINS FOUND:",
    ]
    
    for gstin in result.gstins:
        lines.append(f"  - {gstin}")
    
    lines.extend(["", "DATES FOUND:"])
    for date in result.dates:
        lines.append(f"  - {date}")
    
    lines.extend(["", "AMOUNTS FOUND:"])
    for amount in result.amounts[:20]:  # Limit to avoid noise
        lines.append(f"  - {amount}")
    
    if result.tables:
        lines.extend(["", "LINE ITEMS TABLE:"])
        for table in result.tables:
            lines.append(f"  Headers: {table.headers}")
            for row in table.rows:
                lines.append(f"  ROW {row.row_index}: {row.cells}")
    
    lines.extend(["", "=" * 60])
    
    return {
        "extraction_result": result,
        "llm_prompt_text": "\n".join(lines)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)