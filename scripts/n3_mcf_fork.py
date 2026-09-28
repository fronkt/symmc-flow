"""N3 / G3: the MolCrystalFlow fork = MCF @3c493f8 + patches/mcf_n3.patch (protocol tasks/n3_protocol.md §3.2
"Code base": P1-P5, P6 behind opt-in keys, and our Hydra configs ours_*.yaml).

    python scripts/n3_mcf_fork.py setup [--src https://github.com/Liu-Group-UF/MolCrystalFlow]
        clone MCF into external/mcf (LF line endings), check out 3c493f8 on branch n3, apply the patch
    python scripts/n3_mcf_fork.py diff
        rewrite patches/mcf_n3.patch = diff of external/mcf against 3c493f8, tracked changes + new files
    python scripts/n3_mcf_fork.py verify [--src <MCF clone or URL>]
        apply the patch to a fresh 3c493f8 checkout under n3_mcf/_tmp and compare it with external/mcf

The patch is the published SI diff; external/mcf itself is gitignored.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = os.environ.get("N3_MCF", os.path.join(REPO, "external", "mcf"))
PATCH = os.path.join(REPO, "patches", "mcf_n3.patch")
BASE = "3c493f8"
URL = "https://github.com/Liu-Group-UF/MolCrystalFlow"


def git(*args, cwd=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    r.stdout, r.stderr = r.stdout.decode("utf-8"), r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r


def checkout(src, dest):
    git("-c", "core.autocrlf=false", "clone", "-q", src, dest)
    git("-c", "core.autocrlf=false", "checkout", "-q", "-b", "n3", BASE, cwd=dest)


def make_diff():
    tracked = git("diff", "--no-color", "--no-ext-diff", BASE, "--", ".", cwd=FORK).stdout
    new = git("ls-files", "--others", "--exclude-standard", cwd=FORK).stdout.split()
    parts = [tracked]
    for f in sorted(new):
        r = git("diff", "--no-color", "--no-index", "--", "/dev/null", f, cwd=FORK, check=False)
        assert r.returncode == 1 and r.stdout, f"no diff for new file {f}"
        parts.append(r.stdout)
    text = "".join(parts).replace("\r\n", "\n")
    os.makedirs(os.path.dirname(PATCH), exist_ok=True)
    with open(PATCH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {PATCH}: {len(new)} new files, {text.count(chr(10))} lines")
    return new


def rmtree(path):
    """shutil.rmtree that also removes git's read-only object files (Windows)."""
    import stat

    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)


def tree_files(root):
    out = set()
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
        out.update(os.path.relpath(os.path.join(d, f), root) for f in files if not f.endswith(".pyc"))
    return out


def same_text(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read().replace(b"\r\n", b"\n") == fb.read().replace(b"\r\n", b"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["setup", "diff", "verify"])
    ap.add_argument("--src", default=URL, help="MCF repository (URL or local clone) holding commit " + BASE)
    args = ap.parse_args()
    if args.cmd == "setup":
        if os.path.exists(FORK):
            raise SystemExit(f"{FORK} exists; remove it first")
        checkout(args.src, FORK)
        git("apply", "--whitespace=nowarn", PATCH, cwd=FORK)
        print(f"fork ready: {FORK} = {BASE} + {os.path.relpath(PATCH, REPO)}")
    elif args.cmd == "diff":
        make_diff()
    else:
        tmp = os.path.join(REPO, "n3_mcf", "_tmp", "fork_verify")
        rmtree(tmp)
        checkout(args.src, tmp)
        git("apply", "--whitespace=nowarn", PATCH, cwd=tmp)
        a, b = tree_files(FORK), tree_files(tmp)
        only = sorted(a ^ b)
        differ = sorted(f for f in a & b if not filecmp.cmp(os.path.join(FORK, f), os.path.join(tmp, f), False)
                        and not same_text(os.path.join(FORK, f), os.path.join(tmp, f)))
        rmtree(tmp)
        print(f"patched {BASE} vs {FORK}: {len(a & b)} common files, only-in-one {only}, differing {differ}")
        sys.exit(1 if (only or differ) else 0)


if __name__ == "__main__":
    main()
