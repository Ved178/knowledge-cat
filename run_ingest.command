#!/bin/bash
# Ingest documents from ./data into ChromaDB.
# Edit the PATHS variable below to point at your folder(s).
set -e
cd "$(dirname "$0")"

PATHS="./data"
# PATHS="./data /path/to/another/folder"  # multiple paths example

if [ ! -f env/bin/python ]; then
  echo "Virtual environment not found. Run setup.command first."
  read -rp "Press Enter to close..."
  exit 1
fi

env/bin/python ingest.py --paths $PATHS --embedding-model models/e5-large-v2
echo ""
read -rp "Press Enter to close..."
