#!/bin/zsh
# Render the model evaluation notebook using the Sentiri venv's Jupyter kernel.
# Usage: ./render.sh
export PATH="/Users/apple/degree/finalyearengineredo/venv/bin:$PATH"
cd "$(dirname "$0")"
exec quarto render model_evaluation.qmd "$@"
