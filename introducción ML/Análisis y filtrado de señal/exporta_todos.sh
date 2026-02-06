#!/usr/bin/env bash
set -e

OUTDIR="html_py"

mkdir -p "$OUTDIR"

# HTML
jupyter nbconvert --to html *.ipynb --output-dir "$OUTDIR"

# Scripts Python
jupyter nbconvert --to script *.ipynb --output-dir "$OUTDIR"

echo "✅ HTML y .py generados en ./$OUTDIR"