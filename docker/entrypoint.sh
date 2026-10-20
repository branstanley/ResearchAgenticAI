#!/bin/sh
set -eu

ollama serve &
server_pid=$!

echo "Waiting for Ollama to accept connections..."
attempt=0
until ollama list >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 60 ]; then
    echo "Ollama did not become ready in time." >&2
    kill "$server_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if [ -z "${OLLAMA_MODEL:-}" ]; then
  echo "OLLAMA_MODEL is not set." >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi

echo "Pulling ${OLLAMA_MODEL}..."
ollama pull "$OLLAMA_MODEL"

wait "$server_pid"
