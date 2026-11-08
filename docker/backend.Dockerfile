FROM python:3.10.19-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends build-essential ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-web.txt requirements-pretrained.txt ./
COPY docker/check_runtime.py ./docker/check_runtime.py
ARG TORCH_CHANNEL=cu121
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG PIP_DEFAULT_TIMEOUT=120
ENV PEP_TORCH_CHANNEL=${TORCH_CHANNEL}
RUN case "$TORCH_CHANNEL" in cu121|cpu) ;; *) exit 2 ;; esac \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip download --no-cache-dir --no-deps \
        --index-url "${TORCH_INDEX_BASE_URL%/}/${TORCH_CHANNEL}" --dest /tmp/pep-torch-wheels \
        "torch==2.5.1+${TORCH_CHANNEL}" "torchvision==0.20.1+${TORCH_CHANNEL}" \
    && python -m pip install --no-cache-dir /tmp/pep-torch-wheels/*.whl \
    && rm -rf /tmp/pep-torch-wheels \
    && python -m pip install --no-cache-dir -r requirements-web.txt -r requirements-pretrained.txt \
    && python -m pip check \
    && python docker/check_runtime.py --channel "$TORCH_CHANNEL"

COPY . .
RUN mkdir -p /var/lib/pepplatform/pretrained/models \
    && ln -s /var/lib/pepplatform/pretrained/models /app/models

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec python -m uvicorn pepplatform.web.app:app --host 0.0.0.0 --port 8000"]
