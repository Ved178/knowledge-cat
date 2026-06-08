#!/bin/bash
# Generate PCA and t-SNE embedding plots from ChromaDB.
set -e
cd "$(dirname "$0")"

if [ ! -f env/bin/python ]; then
  echo "Virtual environment not found. Run setup.command first."
  read -rp "Press Enter to close..."
  exit 1
fi

env/bin/python plot_embeddings.py --chroma-path ./chroma_db
echo ""
echo "Plots saved to embedding_plots/"
read -rp "Press Enter to close..."
