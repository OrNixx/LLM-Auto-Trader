FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Устанавливаем зависимости и сам проект.
COPY . .

RUN pip install --upgrade pip \
    && pip install ".[local,telegram,finance-extra]" \
    && mkdir -p /app/data /app/logs

CMD ["python", "-m", "scripts.paper_trade"]
