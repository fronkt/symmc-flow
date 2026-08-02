#!/usr/bin/env python3
"""Build the ACS Omega Word manuscript from main.tex.

achemso's `tocentry`, `acknowledgement` and `suppinfo` environments have no
pandoc reader, so they are rewritten to plain sectioning commands first, and
vector figures are swapped to PNG (Word cannot render an embedded PDF image).

Uses literal string replacement, NOT sed: a sed-based version of this step is
what produced `oindent extbfSupporting Information` in the JCIM package, where
the backslashes of \\noindent and \\textbf were consumed as escape sequences.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "main.tex"
INTERMEDIATE = HERE / "main_docx.tex"
OUT = HERE / "Cai_SymMCFlow_ACSOmega_manuscript.docx"


def _field(tex: str, name: str) -> str:
    """Read a single-argument preamble command, whitespace-collapsed."""
    m = re.search(r"\\" + name + r"\{(.+?)\}", tex, flags=re.DOTALL)
    if not m:
        sys.exit(f"build_docx: \\{name}{{...}} not found in main.tex")
    return " ".join(m.group(1).split())


def transform(tex: str) -> str:
    # pandoc has no reader for achemso's \affiliation and \email, so the Word
    # author block would carry the name alone. Fold them into \author, and
    # restate \keywords after the abstract -- both read out of main.tex so this
    # file stays the single source of truth.
    author, affiliation = _field(tex, "author"), _field(tex, "affiliation")
    email, keywords = _field(tex, "email"), _field(tex, "keywords")
    tex = re.sub(
        r"\\author\{.+?\}",
        lambda _: "\\author{%s\\\\%s\\\\%s}" % (author, affiliation, email),
        tex,
        count=1,
        flags=re.DOTALL,
    )
    tex = tex.replace(
        r"\end{abstract}",
        r"\end{abstract}" + "\n\n" + r"\noindent\textbf{Keywords:} " + keywords,
        1,
    )

    # The TOC graphic is uploaded in its own ACS Paragon Plus slot; it is not
    # part of the manuscript body. Drop the environment and its \includegraphics.
    tex = re.sub(
        r"\\begin\{tocentry\}.*?\\end\{tocentry\}\s*",
        "",
        tex,
        flags=re.DOTALL,
    )

    # achemso environments -> ACS end-matter headings.
    replacements = [
        (r"\begin{acknowledgement}", r"\section*{Acknowledgments}" + "\n"),
        (r"\end{acknowledgement}", ""),
        (
            r"\begin{suppinfo}",
            r"\section*{Associated Content}"
            + "\n\n"
            + r"\noindent\textbf{Supporting Information.}"
            + "\n",
        ),
        (r"\end{suppinfo}", ""),
        # citeproc emits the bibliography where \bibliography sits; give it a heading.
        (r"\bibliography{references}", r"\section*{References}" + "\n" + r"\bibliography{references}"),
    ]
    for old, new in replacements:
        if old not in tex:
            sys.exit(f"build_docx: expected marker not found in main.tex: {old!r}")
        tex = tex.replace(old, new)

    # Word cannot display an embedded PDF image; use the 300 dpi PNG exports.
    tex = tex.replace("fig3_ladder.pdf", "fig3_ladder.png")
    tex = tex.replace("fig_toc.pdf", "fig_toc.png")

    # pandoc silently DROPS the caption of a starred (full-width) float, so
    # Tables 5-7 would arrive in Word uncaptioned. There are no columns to span
    # in a Word manuscript, so unstar them.
    tex = tex.replace(r"\begin{table*}", r"\begin{table}")
    tex = tex.replace(r"\end{table*}", r"\end{table}")
    tex = tex.replace(r"\begin{figure*}", r"\begin{figure}")
    tex = tex.replace(r"\end{figure*}", r"\end{figure}")

    return number_captions(tex)


def number_captions(tex: str) -> str:
    """Prefix every float caption with its LaTeX number.

    pandoc does not number floats, so without this the Word manuscript has
    captions with no "Table 3."/"Figure 1." label. Tables and figures carry
    independent counters, matching LaTeX, and captions are numbered in order of
    appearance in the source.
    """
    counters = {"table": 0, "figure": 0}
    labels = {"table": "Table", "figure": "Figure"}
    token = re.compile(r"\\begin\{(table|figure)\}|\\end\{(table|figure)\}|\\caption\{")

    open_floats: list[str] = []
    parts: list[str] = []
    pos = 0
    for m in token.finditer(tex):
        parts.append(tex[pos:m.start()])
        pos = m.end()
        begun, ended = m.group(1), m.group(2)
        if begun:
            open_floats.append(begun)
            parts.append(m.group(0))
        elif ended:
            if open_floats:
                open_floats.pop()
            parts.append(m.group(0))
        elif open_floats:
            env = open_floats[-1]
            counters[env] += 1
            parts.append("\\caption{%s %d. " % (labels[env], counters[env]))
        else:
            parts.append(m.group(0))
    parts.append(tex[pos:])
    return "".join(parts)


def post_process(path: Path) -> None:
    """Restyle pandoc's abstract to match the rest of the ACS end-matter.

    pandoc emits its own `Abstract Title`/`Abstract` styles, which are
    semantically right but carry pandoc's own look and are invisible to a
    Heading-based section audit. Normalize them so `apply_format.py` styles
    the abstract exactly like every other section.
    """
    import docx  # local import: only needed for this step

    doc = docx.Document(str(path))
    for para in doc.paragraphs:
        if para.style.name == "Abstract Title":
            para.style = doc.styles["Heading 1"]
        elif para.style.name == "Abstract":
            para.style = doc.styles["Normal"]
    doc.save(str(path))


def main() -> None:
    INTERMEDIATE.write_text(transform(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(INTERMEDIATE),
            "--citeproc",
            "--bibliography=references.bib",
            "--csl=acs.csl",
            "--resource-path=.:figures",
            "-o",
            str(OUT),
        ],
        cwd=HERE,
        check=True,
    )
    post_process(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
