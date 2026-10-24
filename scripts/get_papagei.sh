#!/usr/bin/env bash
# Download PaPaGei-S pretrained weights (Nokia Bell Labs, ICLR 2025, BSD-3-Clause-Clear).
# Source: https://zenodo.org/records/13983110  (papagei_s.pt, 23 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
if [ -f weights/papagei_s.pt ]; then
  echo "weights/papagei_s.pt already present"
  exit 0
fi
curl -L --fail -o weights/papagei_s.pt "https://zenodo.org/records/13983110/files/papagei_s.pt?download=1"
echo "saved weights/papagei_s.pt ($(du -h weights/papagei_s.pt | cut -f1))"
