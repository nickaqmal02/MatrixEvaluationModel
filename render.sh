#!/bin/zsh
# Render the Sentiri evaluation site (both pages) using the Sentiri venv's
# Jupyter kernel.
#
# Usage:
#   ./render.sh              → full site with --execute (first run builds the
#                              cell cache, ~90 sec; later runs reuse it)
#   ./render.sh --no-execute → site from cache / prose only (instant)
#   ./render.sh pdf          → PDF of the model evaluation page (--execute)
#
# Caching notes:
#   - Heavy model cells (train-xlmr, train-baselines) carry `cache: true`;
#     Quarto reuses their output unless the CELL CODE changes.
#   - Data-file edits (gold_set_rag.json, *_results.json) do NOT invalidate
#     the cache — those cells are deliberately uncached and cheap (load-JSON).
#     To force a full re-run: quarto render --execute --cache=refresh
export PATH="/Users/apple/degree/finalyearengineredo/venv/bin:$PATH"
cd "$(dirname "$0")"

if [[ "$1" == "pdf" ]]; then
  echo "Rendering PDF (executing all cells for charts)..."
  quarto render model_evaluation.qmd --to pdf --execute
else
  exec quarto render "$@"
fi
