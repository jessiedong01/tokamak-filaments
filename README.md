# Tokamak filaments from 100 kHz imaging

Filaments are blobs of plasma that break off the edge of a tokamak and carry heat to the wall. This repository measures how fast and how large they are on MAST, from the 48 × 256 pixel, 100,000 frames-per-second strip recordings of the midplane fast camera in the public [FAIR-MAST](https://mastapp.site) archive. An automatic pipeline (background subtraction, thresholding, connected components, Hungarian tracking, optical flow) turns 368 discharges into 86,042 filament tracks, recovers the pixel scale from the equilibrium reconstruction because the archive ships none, and tests the result against the two-region blob model. The radial velocity is nearly independent of filament size, and the filaments sit at the model's predicted velocity maximum but move at a few percent of its velocity scale.

The paper is [`paper/main.pdf`](paper/main.pdf).

## Setup

```bash
uv venv && uv pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python src/catalog.py            # pick the discharges (reads the archive's shot table)
.venv/bin/python src/fetch.py              # download the frames, about 28 GB
.venv/bin/python src/process.py            # detect and track filaments
.venv/bin/python src/xcorr_check.py        # independent velocity check
./run_analysis.sh                          # calibrate, aggregate, analyse, draw the figures
cd paper && tectonic -X compile main.tex   # build the paper
```

The shot lists in `data/` and the aggregated results in `results/` are committed, so the selection and the numbers can be checked without downloading the frames.

## Data

All data come from the FAIR-MAST archive (Jackson et al., *SoftwareX* 27, 101869, 2024).
