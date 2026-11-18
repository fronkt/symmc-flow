import re
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
n0 = len(s)
# latexdiff wraps a control space after \begin{description}/{itemize}/{enumerate} in \DIFadd{...}; LaTeX then
# sees text before the first \item. Drop that empty wrapper.
s, k = re.subn(r"(\\begin\{(?:description|itemize|enumerate)\})\\DIF(?:add|del)\{\\\s*\}", r"\1", s)
# a bare control space (backslash at end of line) right after the list opener has the same effect
s, k2 = re.subn(r"(\\begin\{(?:description|itemize|enumerate)\})\\[ \t]*\n", r"\1\n", s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("fixed", k, k2)
