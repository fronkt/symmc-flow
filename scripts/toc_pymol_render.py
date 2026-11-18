"""Ray-traced ball-and-stick render for the TOC graphic. Runs in the PyMOL venv, NOT the project env:

    <pymol venv python> scripts/toc_pymol_render.py out.png  < scene.json   (scene on stdin; never written to disk)

Scene (all lengths in A, coordinates already in the camera frame: x right, y up, z toward the viewer):
  atoms:   [{"el", "p": [x, y, z], "r": ball radius, "rgb": [r, g, b]}]
  bonds:   [[i, j], ...]          stick_r: stick radius
  cyl:     [{"a", "b", "r", "rgb"}]                      plain cylinders (cell edges, arrow shafts)
  cone:    [{"a", "b", "r", "rgb"}]                      arrow heads (base a, tip b)
  window:  [u0, u1, v0, v1]       the orthographic window; res_x / res_y must have the same aspect
  outline: true -> PyMOL's black-outline ray trace mode (ray_trace_mode 1), the STS report look
Adapted from sts-report/report/figures/src/pymol_atoms.py (same camera construction and lighting).
"""
import json
import math
import sys

import pymol2
from chempy import Atom, Bond, models
from pymol.cgo import BEGIN, COLOR, CONE, CYLINDER, END, SPHERE  # noqa: F401

spec = json.load(sys.stdin)
out_png = sys.argv[1]
u0, u1, v0, v1 = spec["window"]
cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)


def cam(p):
    return [float(p[0]) - cu, float(p[1]) - cv, float(p[2])]


p = pymol2.PyMOL()
p.start()
cmd = p.cmd
cmd.hide("everything")

if spec["atoms"]:
    m = models.Indexed()
    for k, a in enumerate(spec["atoms"]):
        at = Atom()
        at.symbol = a["el"]
        at.name = f"A{k}"                      # PyMOL may reorder atoms on load, so colour by unique name
        at.resn = "MOL"
        at.resi = "1"
        at.coord = cam(a["p"])
        at.vdw = a["r"]
        m.add_atom(at)
    for i, j in spec["bonds"]:
        bd = Bond()
        bd.index = [int(i), int(j)]
        bd.order = 1
        m.add_bond(bd)
    cmd.load_model(m, "st")
    for k, a in enumerate(spec["atoms"]):
        cmd.set_color(f"c{k}", a["rgb"])
        cmd.color(f"c{k}", f"st and name A{k}")
    cmd.hide("everything")
    cmd.show("spheres", "st")
    cmd.show("sticks", "st")
cmd.set("sphere_scale", 1.0)
cmd.set("stick_radius", spec.get("stick_r", 0.13))
cmd.set("valence", 0)
cmd.set("stick_ball", 0)
cmd.set("stick_color", -1)                     # sticks take the atom colours (half-bond colouring)

cg = []
for c in spec.get("cyl", []):
    cg += [CYLINDER, *cam(c["a"]), *cam(c["b"]), c["r"], *c["rgb"], *c["rgb"]]
for c in spec.get("cone", []):
    cg += [CONE, *cam(c["a"]), *cam(c["b"]), c["r"], 0.0, *c["rgb"], *c["rgb"], 1.0, 0.0]
for c in spec.get("sph", []):
    cg += [COLOR, *c["rgb"], SPHERE, *cam(c["p"]), c["r"]]
if cg:
    cmd.load_cgo(cg, "cg")

# camera: identity rotation, orthographic; the visible height equals the window height
H = v1 - v0
fov = 20.0
dist = H / (2 * math.tan(math.radians(fov) / 2))
zs = [cam(a["p"])[2] for a in spec["atoms"]]
zs += [cam(c[k])[2] for key in ("cyl", "cone") for c in spec.get(key, []) for k in ("a", "b")]
depth = max(abs(z) for z in zs) + 5.0
cmd.set("field_of_view", fov)
cmd.set("orthoscopic", 1)
cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -dist, 0, 0, 0, dist - depth, dist + depth, fov))
cmd.set("orthoscopic", 1)
assert cmd.get("orthoscopic") == "on"

cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.set("ray_shadows", 1)
cmd.set("ray_shadow_decay_factor", 0.1)
cmd.set("ambient", spec.get("ambient", 0.35))
cmd.set("direct", 0.55)
cmd.set("reflect", 0.35)
cmd.set("specular", 0.5)
cmd.set("shininess", 40)
cmd.set("light_count", 2)
cmd.set("depth_cue", 1 if spec.get("fog", True) else 0)
cmd.set("ray_trace_fog", 1 if spec.get("fog", True) else 0)
cmd.set("fog_start", spec.get("fog_start", 0.45))
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1 if spec.get("outline") else 0)
if spec.get("outline"):
    cmd.set("ray_trace_gain", spec.get("gain", 0.10))
    cmd.set("ray_trace_color", "black")
cmd.ray(int(spec["res_x"]), int(spec["res_y"]))
cmd.png(out_png, dpi=600)
p.stop()
print("RENDERED", out_png)
