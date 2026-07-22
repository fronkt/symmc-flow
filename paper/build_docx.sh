#!/usr/bin/env bash
# Reproducible DOCX build for journal submission (RSC style).
# Word/submission portals cannot render embedded PDF images, so figure refs
# are swapped to the PNG copies before conversion. Requires pandoc >= 3.
set -euo pipefail
cd "$(dirname "$0")"   # paper/

sed 's|figures/\(fig[A-Za-z0-9_]*\)\.pdf|figures/\1.png|g' main.tex > _main_docx.tex
pandoc _main_docx.tex --citeproc --bibliography=references.bib --csl=rsc.csl -s -o symmc-flow_manuscript.docx
rm _main_docx.tex

sed 's|figures/\(fig[A-Za-z0-9_]*\)\.pdf|figures/\1.png|g' supplementary.tex > _si_docx.tex
pandoc _si_docx.tex --citeproc --bibliography=references.bib --csl=rsc.csl -s -o symmc-flow_SI.docx
rm _si_docx.tex

echo "Built $(pwd)/symmc-flow_manuscript.docx and symmc-flow_SI.docx"
