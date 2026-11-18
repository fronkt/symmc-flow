"""Export the ACS Omega revision (paper/submissions/ACSOmega/main.tex) to Word.

pandoc cannot read this source directly: the numbers are macros from generated/n3_numbers.tex, table rows come in
through \\inputrows (a raw \\@@input), figures are PDFs wrapped in \\IfFileExists, and achemso supplies its own
environments. So the source is flattened first:
  - \\input / \\inputrows / \\IfFileExists are resolved and the number macros expanded to their generated values;
  - \\ref{...} and figure/table/equation numbers are taken from main.aux, so the Word file carries the same
    numbering as main.pdf (build main.tex first);
  - figures use the 600-dpi PNG next to each PDF; sidewaystable becomes table;
  - achemso's acknowledgement / suppinfo / tocentry become plain sections (the TOC graphic goes last, as in the
    ACS Word template).
Then pandoc LaTeX -> Markdown (to place the reference list before the TOC section) -> docx with citeproc,
references.bib and the ACS CSL style; the reference document sets Times New Roman 12 pt, double spaced.
Every number in the output is a generated macro value; nothing is typed here.

    python scripts/acs_docx_export.py [--csl D:/tools/csl/american-chemical-society.csl]
writes paper/submissions/ACSOmega/Cai_SymMCFlow_ACSOmega_revised.docx
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper", "submissions", "ACSOmega")
OUT = os.path.join(PAPER, "Cai_SymMCFlow_ACSOmega_revised.docx")


def read(rel):
    with open(os.path.join(PAPER, rel), encoding="utf-8") as f:
        return f.read()


def strip_comments(s):
    """Drop LaTeX comments (an unescaped % to end of line), keeping \\%."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))


def balanced(s, i):
    """s[i] == '{': return the index just past the matching '}'."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def args_of(s, i, n):
    """Parse n brace arguments starting at s[i] (whitespace allowed between); return (args, end)."""
    out = []
    for _ in range(n):
        while s[i].isspace():
            i += 1
        j = balanced(s, i)
        out.append(s[i + 1:j - 1])
        i = j
    return out, i


def replace_cmd(s, name, n, fn):
    """Replace every \\name{a1}...{an} by fn(args)."""
    pat = "\\" + name
    i = 0
    while True:
        k = s.find(pat, i)
        if k < 0:
            return s
        after = k + len(pat)
        if after < len(s) and s[after].isalpha():
            i = after
            continue
        a, end = args_of(s, after, n)
        rep = fn(a)
        s = s[:k] + rep + s[end:]
        i = k + len(rep)


def macros():
    m = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", read("generated/n3_numbers.tex"), re.M):
        m[name] = val
    assert len(m) > 50, len(m)
    return m


def expand_macros(s, m):
    for name in sorted(m, key=len, reverse=True):
        val = m[name]
        s = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda _: val, s)   # function: value used literally
    return s


def aux_labels():
    lab = {}
    for key, num in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", read("main.aux")):
        lab[key] = num
    for need in ("fig:worked", "tab:info", "tab:repr", "fig:forest", "tab:main", "tab:hyp", "tab:strata",
                 "eq:expand"):
        assert need in lab, f"{need} missing from main.aux: build main.tex first"
    return lab


def flatten():
    s = read("main.tex")
    body = s[s.index("\\begin{document}") + len("\\begin{document}"):s.index("\\end{document}")]
    body = strip_comments(body)
    # \IfFileExists{file}{yes}{no} -> yes ; then \input / \inputrows
    body = replace_cmd(body, "IfFileExists", 3, lambda a: a[1])
    body = replace_cmd(body, "input", 1, lambda a: strip_comments(read(a[0] if a[0].endswith(".tex")
                                                                       else a[0] + ".tex")))
    body = replace_cmd(body, "inputrows", 1, lambda a: strip_comments(read(a[0])))
    t = re.search(r"\\title\{([\s\S]*?)\}\s*\\keywords", s).group(1)
    author = re.search(r"\\author\{(.*?)\}", s).group(1)
    affil = re.search(r"\\affiliation\{(.*?)\}", s).group(1)
    email = re.search(r"\\email\{(.*?)\}", s).group(1)
    kw = re.search(r"\\keywords\{([\s\S]*?)\}", s).group(1)
    return body, " ".join(t.split()), author, affil, email, " ".join(kw.split())


def transform(body, lab, m):
    body = expand_macros(body, m)
    body = re.sub(r"\\itemsep[0-9.]*em", "", body)
    body = body.replace("sidewaystable", "table")
    body = replace_cmd(body, "paragraph", 1, lambda a: "\\textbf{" + a[0] + "}")   # run-in, not a numbered heading
    # figures: PDF -> the PNG beside it
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{(fig_n3_toc)\}", r"\\includegraphics[width=3.25in]{figures/\2.png}", body)
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{(fig_[^}.]+)\}", r"\\includegraphics[width=6.5in]{figures/\2.png}", body)

    # caption numbers from main.aux
    def number_env(mt):
        env, txt = mt.group(1), mt.group(0)
        lm = re.search(r"\\label\{([^}]+)\}", txt)
        if not lm:
            return txt
        word = "Figure" if env == "figure" else "Table"
        return txt.replace("\\caption{", "\\caption{\\textbf{" + word + " " + lab[lm.group(1)] + ".} ", 1)

    body = re.sub(r"\\begin\{(figure|table)\}[\s\S]*?\\end\{\1\}", number_env, body)
    # equation number shown in the equation itself
    body = re.sub(r"\\label\{(eq:[^}]+)\}", lambda mt: r"\qquad(" + lab[mt.group(1)] + ")", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda mt: lab[mt.group(1)], body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    # achemso environments
    toc = re.search(r"\\begin\{tocentry\}([\s\S]*?)\\end\{tocentry\}", body).group(1)
    body = re.sub(r"\\begin\{tocentry\}[\s\S]*?\\end\{tocentry\}", "", body)
    body = body.replace("\\begin{acknowledgement}", "\\section*{Acknowledgment}").replace("\\end{acknowledgement}", "")
    body = body.replace("\\begin{suppinfo}", "\\section*{Supporting Information}\nThe Supporting Information is "
                        "available free of charge: ").replace("\\end{suppinfo}", "")
    body = re.sub(r"\\bibliography\{[^}]+\}", "", body)
    body = body.replace("\\centering", "")
    return body, toc


def reference_docx(path):
    import docx
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Pt
    subprocess.run(["pandoc", "-o", path, "--print-default-data-file", "reference.docx"], check=True)
    d = docx.Document(path)
    for st in d.styles:
        if st.type != 1:                       # paragraph styles only
            continue
        try:
            st.font.name = "Times New Roman"
            st.font.color.rgb = None
            el = st.element.rPr
            if el is not None and el.rFonts is not None:
                for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                    el.rFonts.set(docx.oxml.ns.qn(k), "Times New Roman")
        except Exception:
            pass
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Bibliography"):
        if name in [s.name for s in d.styles]:
            st = d.styles[name]
            st.font.size = Pt(12)
            if name in ("Normal", "Body Text", "First Paragraph", "Abstract"):
                st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    for name, size in (("Title", 16), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12), ("Heading 4", 12)):
        if name in [s.name for s in d.styles]:
            d.styles[name].font.size = Pt(size)
            d.styles[name].font.bold = True
    d.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csl", default="D:/tools/csl/american-chemical-society.csl")
    a = ap.parse_args()
    assert os.path.exists(a.csl), f"ACS CSL style not found: {a.csl}"
    lab, m = aux_labels(), macros()
    body, title, author, affil, email, kw = flatten()
    body, toc = transform(body, lab, m)
    left = re.findall(r"\\n[A-Z][A-Za-z]+", body)
    assert not left, f"unexpanded number macros: {sorted(set(left))}"
    head = (f"\\documentclass{{article}}\n\\usepackage{{amsmath,amssymb,graphicx,booktabs}}\n"
            f"\\title{{{title}}}\n\\author{{{author}}}\n\\date{{}}\n\\begin{{document}}\n\\maketitle\n"
            f"{affil}. E-mail: {email}\n\n\\textbf{{Keywords:}} {kw}\n\n")
    tex = head + body + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as td:
        src, md, ref = (os.path.join(td, n) for n in ("flat.tex", "flat.md", "ref.docx"))
        with open(src, "w", encoding="utf-8") as f:
            f.write(tex)
        subprocess.run(["pandoc", src, "-f", "latex", "-t", "markdown", "-s", "-o", md], check=True, cwd=PAPER)
        with open(md, encoding="utf-8") as f:
            mdt = f.read()
        toc_md = subprocess.run(["pandoc", "-f", "latex", "-t", "markdown"],
                                input="\\begin{document}" + toc + "\\end{document}", capture_output=True,
                                text=True, check=True, cwd=PAPER).stdout
        mdt += ("\n\n# References {.unnumbered}\n\n::: {#refs}\n:::\n\n"
                "# For Table of Contents Only {.unnumbered}\n\n" + toc_md + "\n")
        with open(md, "w", encoding="utf-8") as f:
            f.write(mdt)
        reference_docx(ref)
        subprocess.run(["pandoc", md, "-f", "markdown", "-o", OUT, "--citeproc",
                        "--bibliography", os.path.join(PAPER, "references.bib"), "--csl", a.csl,
                        "--reference-doc", ref, "--resource-path", PAPER, "--number-sections"],
                       check=True, cwd=PAPER)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
