#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment .venv not found. Please run ./install.sh first." >&2
    exit 1
fi

if [ ! -d "C3PA_Dataset" ]; then
    echo "Error: C3PA_Dataset directory not found. Please run ./install.sh first." >&2
    exit 1
fi

source .venv/bin/activate
python parse_c3pa_sentences.py "$@"
