FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

WORKDIR /app

# Dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Default: HTTP API (Cloud Run injects $PORT). The CLI report still works with:
#   docker run --rm ops-report python -m ops_report --db /tmp/tickets.db --out /tmp/report.md
CMD ["sh", "-c", "uvicorn ops_report.api:app --host 0.0.0.0 --port ${PORT}"]
