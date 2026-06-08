#!/bin/bash
# macOS launcher — double-click this file in Finder to run setup.
# First time only: right-click → Open to bypass Gatekeeper, or run:
#   chmod +x setup.command && open setup.command
set -e
cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
  echo "Python 3 not found. Download it from https://python.org/downloads"
  read -rp "Press Enter to close..."
  exit 1
fi

python3 setup.py
echo ""
read -rp "Press Enter to close this window..."
