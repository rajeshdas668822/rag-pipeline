#!/bin/bash
# ============================================================
# RAG Reranking Lab — Stop local Ollama (Docker)
# ============================================================
# Cleanly stops and removes the Ollama container. Downloaded models
# are preserved in the 'ollama_data' Docker volume.
# ============================================================

echo "=== Stopping local Ollama ==="
docker compose down
echo "Ollama stopped."
echo ""
echo "Note: Downloaded models are preserved in the 'ollama_data' Docker volume."
echo "To remove everything including downloaded models: docker compose down -v"