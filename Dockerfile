# Invoice Extraction Service Dockerfile
# PyMuPDF + Tesseract OCR for text extraction from PDFs

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Tesseract OCR
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set tessdata path after installation
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]