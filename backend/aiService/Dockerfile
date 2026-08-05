FROM python:3.11-slim

WORKDIR /app

# Copy aiService files
COPY aiService/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy aiService source
COPY aiService/ ./aiService/

# Copy each feature package main.py imports routers from
COPY translation/ ./translation/
COPY summary/ ./summary/
COPY extract/ ./extract/
COPY interact/ ./interact/
COPY comparison/ ./comparison/
COPY transliteration/ ./transliteration/
COPY dynamic_fields/ ./dynamic_fields/
COPY redact/ ./redact/

# Copy fonts
COPY fonts/ ./fonts/

WORKDIR /app/aiService

ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
