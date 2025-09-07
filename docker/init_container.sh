#!/bin/bash
# init_container.sh
# Downloads llama3.1 LLM via Ollama if it does not already exist.
# Runs supervisord which will handle running the Ollama and Uvicorn processes.

set -e

LLM_NAME="llama3.1:8b"

# Check if LLM already exists in Ollama's storage
# Note: 'ollama list' checks /root/.ollama (docker volume)
if ! ollama list | grep -q "$LLM_NAME"; then
    echo "LLM $LLM_NAME not found. Pulling now..."
    ollama serve &
    OLLAMA_PID=$!

    # Wait for Ollama API to be ready
    until curl -s http://localhost:11434/api/tags >/dev/null; do
        sleep 1
    done

    ollama pull "$LLM_NAME"

    # Kill background Ollama process after pulling
    kill $OLLAMA_PID
    wait $OLLAMA_PID || true
else
    echo "LLM $LLM_NAME already present. Skipping pull."
fi

# Start supervisord - Runs Ollama & Uvicorn
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
