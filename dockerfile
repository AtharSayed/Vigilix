FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Project
COPY . /app

# Defaults for in-container networking (overridable via compose)
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
    WAIT_FOR_KAFKA_TIMEOUT_SEC=120 \
    WAIT_FOR_KAFKA_INTERVAL_SEC=2 \
    METRICS_PORT=8001

EXPOSE 8001

# Use shell-form to avoid the "[python, not found]" issue
CMD python src/main.py
