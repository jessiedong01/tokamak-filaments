#!/bin/bash
# analysis -> figures only (aggregate outputs already on disk)
cd /Users/jessie/tokamak-filaments
set -eo pipefail
echo "[$(date +%H:%M:%S)] analysis";  .venv/bin/python src/analysis.py  2>&1 | grep -v -E "Warning|warnings.warn"
echo "[$(date +%H:%M:%S)] figures";   .venv/bin/python src/make_figures.py 2>&1 | grep -v -E "Warning|warnings.warn"
echo "[$(date +%H:%M:%S)] done"
