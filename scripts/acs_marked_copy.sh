#!/usr/bin/env bash
# Build the ACS "Supporting Information for Review Only" marked-up manuscript: latexdiff of the current main.tex
# (with sections/ and generated/ flattened in) against the pre-N3 original. Insertions underlined (blue), deletions
# struck out (red); tables and figures are treated as whole blocks (marked as added/removed, not diffed by cell).
# Needs latexdiff (MiKTeX) and the pure-Perl Algorithm::Diff on PERL5LIB_DIR (default D:/tools/perl5lib).
set -euo pipefail
cd "$(dirname "$0")/../paper/submissions/ACSOmega"
LD="${LATEXDIFF:-C:/Users/frank/AppData/Local/Programs/MiKTeX/scripts/latexdiff/latexdiff}"
LIB="${PERL5LIB_DIR:-/d/tools/perl5lib}"
# \itemsep spacing commands collide with latexdiff markup; strip them from temporary copies only
sed -E 's/\itemsep[0-9.]*em//g' archive/main-archive-2026-11-17-pre-N3.tex > _old_tmp.tex
sed -E 's/\itemsep[0-9.]*em//g' main.tex > _new_tmp.tex
/usr/bin/perl -I "$LIB" "$LD" --flatten --type=UNDERLINE --math-markup=whole --graphics-markup=none \
  --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|table|sidewaystable|figure)[\w\d*@]*" \
  _old_tmp.tex _new_tmp.tex > main_marked_vs_original.tex
python -I ../../../scripts/latexdiff_fix_lists.py main_marked_vs_original.tex
rm -f _old_tmp.tex _new_tmp.tex main_marked_vs_original.aux main_marked_vs_original.bbl
latexmk -pdf -interaction=nonstopmode -halt-on-error main_marked_vs_original.tex > /dev/null
echo "wrote paper/submissions/ACSOmega/main_marked_vs_original.pdf"
