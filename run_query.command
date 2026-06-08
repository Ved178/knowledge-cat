#!/bin/bash
# Launch the interactive semantic search REPL.
set -e
cd "$(dirname "$0")"

if [ ! -f env/bin/python ]; then
  echo "Virtual environment not found. Run setup.command first."
  read -rp "Press Enter to close..."
  exit 1
fi

env/bin/python query.py --embedding-model models/e5-large-v2
