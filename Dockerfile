FROM python:3.11-slim

WORKDIR /app

# Silence debconf warnings during apt — no TTY available in Docker build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# --root-user-action=ignore suppresses pip-as-root warning (fine in containers)
# pip upgrade also silences the "new release available" notice on every build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip --quiet --root-user-action=ignore

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]