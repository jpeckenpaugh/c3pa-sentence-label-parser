#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Setting up C3PA Sentence-Level Candidate Parser Environment"
echo "============================================================"

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed." >&2
    exit 1
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "--> Creating Python virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "--> Existing virtual environment found in .venv."
fi

# Activate virtual environment
source .venv/bin/activate

echo "--> Installing dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Clone C3PA Dataset if missing
if [ ! -d "C3PA_Dataset" ] || [ ! -d "C3PA_Dataset/Htmls" ]; then
    echo "--> Cloning C3PA Dataset repository (https://github.com/MaazBinMusa/C3PA_Dataset.git)..."
    git clone --depth 1 https://github.com/MaazBinMusa/C3PA_Dataset.git C3PA_Dataset
else
    echo "--> C3PA Dataset found in C3PA_Dataset/."
fi

echo "============================================================"
echo " Setup complete! Run ./parse.sh to execute the parser."
echo "============================================================"
