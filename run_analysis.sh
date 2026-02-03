#!/bin/bash
# Aggregate -> analysis -> figures, logging to results/log_analysis.txt
cd /Users/jessie/tokamak-filaments
set -eo pipefail
echo "[$(date +%H:%M:%S)] aggregate"; .venv/bin/python src/aggregate.py 2>&1 | grep -v -E "Warning|warnings.warn"
echo "[$(date +%H:%M:%S)] analysis";  .venv/bin/python src/analysis.py  2>&1 | grep -v -E "Warning|warnings.warn"
echo "[$(date +%H:%M:%S)] figures";   .venv/bin/python src/make_figures.py 2>&1 | grep -v -E "Warning|warnings.warn"
echo "[$(date +%H:%M:%S)] done"
