#!/bin/zsh
# Render the model evaluation notebook using the Sentiri venv's Jupyter kernel.
# Usage:
#   ./render.sh              → HTML with --execute (loads checkpoints, ~90 sec, all charts)
#   ./render.sh --no-execute → HTML from cache (instant, for prose-only changes)
#   ./render.sh pdf          → PDF with --execute (loads checkpoints, all charts, ~90 sec)
#   ./render.sh html         → same as default (HTML with --execute)
export PATH="/Users/apple/degree/finalyearengineredo/venv/bin:$PATH"
cd "$(dirname "$0")"

if [[ "$1" == "pdf" ]]; then
  echo "Rendering PDF (executing all cells for charts)..."
  quarto render model_evaluation.qmd --to pdf --execute
else
  exec quarto render model_evaluation.qmd "$@"
fi
