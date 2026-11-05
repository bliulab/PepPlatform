FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends build-essential ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-web.txt requirements-pretrained.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1" "torchvision==0.20.1" \
    && python -m pip install --no-cache-dir -r requirements-web.txt -r requirements-pretrained.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec python -m uvicorn pepplatform.web.app:app --host 0.0.0.0 --port 8000"]
