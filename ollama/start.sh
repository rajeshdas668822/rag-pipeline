#!/bin/bash
# ============================================================
# RAG Reranking Lab — Start local Ollama (Docker)
# ============================================================
# Runs Ollama as a local Docker container so the RAG pipeline can
# reach it at http://localhost:11434 (the default OLLAMA_ENDPOINT).
#
# Model selection is OPTIONAL and configurable. Nothing is hardcoded:
#   ./start.sh                              # just start Ollama; pull nothing
#   MODEL=qwen2.5:3b ./start.sh             # also ensure a chat/instruct model
#   EMBED_MODEL=embeddinggemma ./start.sh   # also ensure an embedding model
#   MODEL=qwen2.5:3b EMBED_MODEL=embeddinggemma ./start.sh
#
# Any model you name must be pullable by / already present in Ollama.
# Stop with ./stop.sh
# ============================================================

set -e

# Optional, user-configurable models. Empty by default: no model is forced.
MODEL="${MODEL:-}"
EMBED_MODEL="${EMBED_MODEL:-}"

echo "=== RAG Reranking Lab — local Ollama (Docker) ==="
echo ""

# Step 1: Start the Ollama container
echo "[1/3] Starting Ollama container ..."
docker compose up -d
echo "  Container started."

# Step 2: Wait for Ollama to be ready
echo "[2/3] Waiting for Ollama to be ready ..."
MAX_RETRIES=60
RETRY=0
until docker exec ollama ollama list > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "  ERROR: Ollama failed to start after ${MAX_RETRIES} attempts."
        echo "  Check logs: docker logs ollama"
        exit 1
    fi
    sleep 2
done
echo "  Ollama is ready."

# Step 3: Optionally ensure requested models are present (pull only if named)
echo "[3/3] Checking requested models ..."
if [ -z "$MODEL" ] && [ -z "$EMBED_MODEL" ]; then
    echo "  No MODEL/EMBED_MODEL specified; skipping pull."
    echo "  Already-downloaded models:"
    docker exec ollama ollama list || true
else
    for M in "$MODEL" "$EMBED_MODEL"; do
        [ -z "$M" ] && continue
        if docker exec ollama ollama list | grep -q "$M"; then
            echo "  ✓ $M (already present)"
        else
            echo "  Pulling '$M' (may take a while) ..."
            docker exec ollama ollama pull "$M"
        fi
    done
fi

echo ""
echo "=== Ollama is running ==="
echo "  Endpoint:     http://localhost:11434"
echo "  RAG config:   set OLLAMA_ENDPOINT=http://localhost:11434 (this is the default)"
[ -n "$MODEL" ]       && echo "  Chat model:   $MODEL"
[ -n "$EMBED_MODEL" ] && echo "  Embed model:  $EMBED_MODEL"
echo ""
echo "  List models:  docker exec ollama ollama list"
echo "  Stop:         ./stop.sh"