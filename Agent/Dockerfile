FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir numpy pulp==2.8.0

COPY assign.py auth.py main.py ./

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'cert="${SIGNING_CERT_PATH:-/keys/signing.pfx}"' \
    'while [ ! -s "$cert" ]; do' \
    '  sleep 1' \
    'done' \
    'exec uvicorn main:app --host 0.0.0.0 --port 8000' \
    | tr -d '\r' > /usr/local/bin/intelligence-entrypoint.sh \
 && chmod +x /usr/local/bin/intelligence-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/intelligence-entrypoint.sh"]
