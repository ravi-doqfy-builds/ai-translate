FROM python:3.11-slim-bullseye

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps:
#   wkhtmltopdf  → pdfkit (html_to_pdf)
#   libreoffice  → dynamic_fields DOCX-to-PDF conversion
#   libgl1       → PyMuPDF
#   fonts-*      → Unicode/Indic rendering inside wkhtmltopdf
RUN apt-get update && apt-get install -y --no-install-recommends \
        wkhtmltopdf \
        libreoffice \
        libgl1 \
        fonts-noto \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
