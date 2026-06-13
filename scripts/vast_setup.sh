#!/bin/bash
# Fresh vast.ai Blackwell (RTX 5090) box setup for symmc-flow.
set -e
cd /root

echo "=== clone repo ==="
if [ -d symmc-flow ]; then cd symmc-flow && git pull && cd /root; else git clone https://github.com/fronkt/symmc-flow.git; fi
cd /root/symmc-flow

echo "=== torch cu128 (Blackwell) ==="
pip3 install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

echo "=== deps ==="
pip3 install -q --no-cache-dir numpy scipy pytest pymatgen pandas

echo "=== carbon-24 CSVs (CDVAE) ==="
mkdir -p data/raw
base=https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24
for split in train val test; do
  curl -fsSL "$base/$split.csv" -o "data/raw/carbon_$split.csv"
  echo "  carbon_$split.csv: $(wc -l < data/raw/carbon_$split.csv) lines"
done

echo "=== verify ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
df -h / | tail -1
echo "SETUP DONE"
