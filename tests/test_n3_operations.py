"""Synthetic box-operation invariants: no CSD access, cloud work, or cleanup deletion."""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import n3_box_manifest as BM
import n3_timing_gate as TG


def manifest(payload=b"synthetic TRAIN input", step=5):
    return {"format": BM.FORMAT, "label": f"step{step}", "step": step, "deployable": True,
            "files": [{"group": "set_caches", "csd": True, "path": "data/n3_sets/train.pt",
                       "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}],
            "missing": [], "problems": [], "checks": {}, "sec": 0,
            "forbidden": [{"path": "data/n3_sets/testB.pt", "bytes": 4,
                           "sha256": hashlib.sha256(b"TEST").hexdigest()}]}


def archive(tmp, man=None, payload=b"synthetic TRAIN input", extra=(), omit=()):
    man = manifest() if man is None else man
    data = [("data/n3_sets/train.pt", payload),
            (f".n3_box/manifests/{man['label']}.manifest.json", json.dumps(man).encode())]
    for ext, content in zip((".sha256", ".tsv"), BM.inventory_text(man)):
        data.append((f".n3_box/manifests/{man['label']}" + ext, content.encode()))
    data += list(extra)
    path = tmp / "transfer.tar"
    with tarfile.open(path, "w") as tf:
        for name, content in data:
            if name in omit:
                continue
            ti = tarfile.TarInfo(name)
            ti.size = len(content)
            tf.addfile(ti, io.BytesIO(content))
    Path(str(path) + ".sha256").write_text(BM.sha256(path) + "  transfer.tar\n")
    return path


def unpack_args(tmp, path):
    return SimpleNamespace(tar=str(path), root=str(tmp / "box"), no_overwrite=False)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/absolute", "a\\b", "a//b", "a/./b", "a\tb"])
def test_inventory_paths_are_canonical_and_contained(tmp_path, path):
    with pytest.raises(ValueError):
        BM.safe_path(path, tmp_path)


@pytest.mark.parametrize("change", ["not_deployable", "missing", "problems", "duplicate", "heldout", "no_forbidden"])
def test_incomplete_or_blindness_violating_manifest_is_rejected(tmp_path, change):
    man = manifest()
    if change == "not_deployable":
        man["deployable"] = False
    elif change in ("missing", "problems"):
        man[change] = ["incomplete"]
    elif change == "duplicate":
        man["files"].append(copy.deepcopy(man["files"][0]))
    elif change == "heldout":
        man["files"][0]["path"] = "data/n3_sets/testB.pt"
    else:
        man["forbidden"] = []
    with pytest.raises(ValueError):
        BM.validate_manifest(man, tmp_path)


def test_valid_transfer_checks_members_then_records_receipt(tmp_path, monkeypatch):
    retained = tmp_path / "receipts"
    monkeypatch.setenv("N3_MANIFEST_DIR", str(retained))
    path = archive(tmp_path)
    with pytest.raises(SystemExit) as result:
        BM.cmd_unpack(unpack_args(tmp_path, path))
    assert result.value.code == 0
    assert (tmp_path / "box/data/n3_sets/train.pt").read_bytes() == b"synthetic TRAIN input"
    rec = json.loads((retained / "step5.unpack_receipt.json").read_text())
    assert rec["verified"] and rec["tar_sha256"] == BM.sha256(path)


@pytest.mark.parametrize("fault", ["unlisted", "duplicate", "tampered_member", "missing_inventory", "missing_transfer_hash"])
def test_archive_failure_does_not_extract_any_member(tmp_path, monkeypatch, fault):
    monkeypatch.setenv("N3_MANIFEST_DIR", str(tmp_path / "receipts"))
    kwargs = {}
    if fault == "unlisted":
        kwargs["extra"] = [("data/n3_sets/testB.pt", b"TEST")]
    elif fault == "duplicate":
        kwargs["extra"] = [("data/n3_sets/train.pt", b"synthetic TRAIN input")]
    elif fault == "tampered_member":
        kwargs["payload"] = b"changed"
    elif fault == "missing_inventory":
        kwargs["omit"] = [".n3_box/manifests/step5.tsv"]
    path = archive(tmp_path, **kwargs)
    if fault == "missing_transfer_hash":
        other = tmp_path / "without_checksum.tar"
        other.write_bytes(path.read_bytes())
        path = other
    with pytest.raises((ValueError, SystemExit)):
        BM.cmd_unpack(unpack_args(tmp_path, path))
    assert not (tmp_path / "box").exists()


def test_no_overwrite_preserves_differing_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("N3_MANIFEST_DIR", str(tmp_path / "receipts"))
    path = archive(tmp_path)
    saved = tmp_path / "box/data/n3_sets/train.pt"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"existing local data")
    args = unpack_args(tmp_path, path)
    args.no_overwrite = True
    with pytest.raises(SystemExit):
        BM.cmd_unpack(args)
    assert saved.read_bytes() == b"existing local data"


def test_hidden_heldout_copy_fails_box_verification(tmp_path):
    man = manifest()
    target = tmp_path / man["files"][0]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic TRAIN input")
    hidden = tmp_path / ".n3_box/private/renamed.bin"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"TEST")
    assert not BM.verify(man, tmp_path)


def test_diagnostic_pack_never_creates_archive(tmp_path, monkeypatch):
    man = manifest()
    man["deployable"] = False
    monkeypatch.setattr(BM, "build", lambda *a, **kw: man)
    args = SimpleNamespace(step=5, new_box=False, groups=["code"], allow_missing=True,
                           allow_unpatched=True, out=str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="pack refused"):
        BM.cmd_pack(args)
    assert not (tmp_path / "out").exists()


def test_timing_upload_omits_every_non_train_val_input(monkeypatch):
    monkeypatch.setattr(BM, "code_closure", lambda: (["scripts/n3_common.py"], []))
    monkeypatch.setattr(BM, "fork_files", lambda: ["external/mcf/model.py"])
    monkeypatch.setattr(BM, "n2_chosen", lambda: [f"n2_s{i}.pt" for i in range(3)])
    paths = [p for _, _, _, ps in BM.groups(4) for p in ps]
    assert "data/n3_sets/train.pt" in paths and "data/n3_sets/val.pt" in paths
    assert not any(BM.FORBIDDEN_PATH_4.search(p) for p in paths)
    assert not any("sel" in p or "devtest" in p.lower() or "testb" in p.lower() for p in paths)
    assert "tasks/n3_protocol.md" in paths


def test_all_upload_phases_include_the_public_scientific_protocol(monkeypatch):
    monkeypatch.setattr(BM, "code_closure", lambda: (["scripts/n3_common.py"], []))
    monkeypatch.setattr(BM, "fork_files", lambda: ["external/mcf/model.py"])
    monkeypatch.setattr(BM, "n2_chosen", lambda: [f"n2_s{i}.pt" for i in range(3)])
    for step in (4, 5, 7):
        assert "tasks/n3_protocol.md" in [p for _, _, _, ps in BM.groups(step) for p in ps]
    assert "results/n3/private/classical/ffstar_config.json" in [p for _, _, _, ps in BM.groups(5) for p in ps]


def cleanup_world(tmp):
    root, upload, retained = tmp / "box", tmp / "n3_upload", tmp / "receipts"
    BM.write_manifest_files(manifest(), retained)
    return root, upload, retained


def test_cleanup_plan_uses_only_explicit_private_targets_and_retains_receipts(tmp_path):
    root, upload, retained = cleanup_world(tmp_path)
    plan = BM.cleanup_plan(str(root), str(upload), str(retained), require_pullback=False)
    assert set(plan["targets"]) == {str(root), str(upload)}
    assert (retained / "step5.manifest.json").exists()
    assert not root.exists()


def test_cleanup_refuses_receipt_ancestor_or_unrelated_extra_target(tmp_path):
    root, upload, retained = cleanup_world(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        BM.cleanup_plan(str(tmp_path), str(upload), str(retained), require_pullback=False)
    with pytest.raises(ValueError, match="n3_ name"):
        BM.cleanup_plan(str(root), str(upload), str(retained), extras=[str(tmp_path / "other_project")],
                        require_pullback=False)


def test_cleanup_requires_successful_matching_laptop_pullback_receipt(tmp_path):
    root, upload, retained = cleanup_world(tmp_path)
    pull = manifest(step=8)
    pull["label"] = "pullback_step7"
    BM.write_manifest_files(pull, retained)
    (retained / "pullback_step7.archive_sha256").write_text("a" * 64)
    receipt = retained / "pullback_step7.unpack_receipt.json"
    receipt.write_text(json.dumps({"label": "pullback_step7", "verified": True, "tar_sha256": "b" * 64}))
    with pytest.raises(ValueError, match="verified copy"):
        BM.cleanup_plan(str(root), str(upload), str(retained), receipt=str(receipt))
    receipt.write_text(json.dumps({"label": "pullback_step7", "verified": True, "tar_sha256": "a" * 64}))
    assert BM.cleanup_plan(str(root), str(upload), str(retained), receipt=str(receipt))["pullback_receipt"] == str(receipt)


def projection_world():
    model = lambda v: {"type": "const", "v": v}
    rates = {"gpu": {"mcf_train_epoch_s": {a: {"p6_off": 1.0, "p6_on": 0.8} for a in ("R", "A")},
                     "mcf_infer": {a: {"c0": 1.0, "c1": 0.1} for a in ("R", "A")},
                     "ivp_train_epoch_s": {"pack": 2.0},
                     "ivp_sample_gpu_s_per_crystal": {"pack": {"8": 0.1}},
                     "ff": {"batch_float32": {"single": model(0.01), "batched": model(0.001),
                                                "startup_s": 10.0, "startup_scope": "job"}},
                     "ff_truth_evals_mean": {"float32": 10}},
             "cpu": {name: model(0.01) for name in ("energy_s", "press_s", "fit_primary_s", "rms_mcf05_s",
                         "rms_mcf08_s", "rms_mcf10_s", "screen_s", "ours_sample_s")}}
    stats = {name: {"n": n, "atoms": [40] * n, "K": [4] * n, "z_MCF": [4] * n}
             for name, n in (("val", 100), ("sel", 300), ("valsel", 400), ("testB", 1000), ("devtest", 200))}
    cfg = {"p6": "off", "ivp_encoder": "pack", "ivp_cpb": 8, "ff": "batch_float32", "ff_eval_frac": 1.0,
           "truth_evals": 0, "gplus_separate": False, "tier2": True, "g2_train_h": 0.5, "cores": 8,
           "gpu_job_cores": 2, "overhead": 1.15, "usd_per_hour": 0.3, "local_cores": 4, "cpubox_cores": 16,
           "cpubox_usd_per_hour": 0.2, "degrade": set()}
    return rates, stats, cfg


def test_degrade_cost_is_monotonic_and_preserves_required_basin_support():
    rates, stats, cfg = projection_world()
    previous = TG.totals(TG.build_jobs(rates, stats, cfg), cfg)
    for item in range(1, 8):
        cfg["degrade"].add(item)
        jobs = TG.build_jobs(rates, stats, cfg)
        total = TG.totals(jobs, cfg)
        assert total["usd_hi"] <= previous["usd_hi"] + 1e-12
        required = {j["name"]: j for j in jobs}
        assert required["MCF-R training"]["h"] == 3 * 1715 / 3600
        assert required["iv-P training"]["h"] == 5 * 60 * 2 / 3600
        assert required["OURS-N2 LJ picks, H2 BASIN"]["h"] == pytest.approx(1200 * 3 * 100 * 0.01 / 3600)
        assert required["iii-s FF* minimisations (64 starts x <=100 evaluations)"]["h"] == pytest.approx(1200 * 64 * 100 * 0.001 / 3600)
        if item >= 5:
            assert required["H2 BASIN of OURS LJ picks without i-E reuse"]["h"] >= 200 * 3 * 100 * 0.01 / 3600 - 1e-12
        previous = total


def test_default_ff_cost_includes_per_crystal_restart_and_compile():
    rates, stats, cfg = projection_world()
    batch = TG.totals(TG.build_jobs(rates, stats, cfg), cfg)
    rates["gpu"]["ff"]["default_float32"] = copy.deepcopy(rates["gpu"]["ff"]["batch_float32"])
    rates["gpu"]["ff"]["default_float32"]["startup_scope"] = "crystal"
    cfg["ff"] = "default_float32"
    default = TG.totals(TG.build_jobs(rates, stats, cfg), cfg)
    assert default["gpu_h"] > batch["gpu_h"] + 20


@pytest.mark.parametrize("scope", ["crystal", "job"])
def test_merged_ff_candidate_must_charge_a_server_start_per_crystal(scope):
    rates, stats, cfg = projection_world()
    rates["gpu"]["ff"]["merged_float32"] = copy.deepcopy(rates["gpu"]["ff"]["batch_float32"])
    rates["gpu"]["ff"]["merged_float32"]["startup_scope"] = scope
    cfg["ff"] = "merged_float32"
    if scope == "job":
        with pytest.raises(ValueError, match="startup scope"):
            TG.validate_projection_inputs(rates, stats, cfg)
    else:
        TG.validate_projection_inputs(rates, stats, cfg)
        batch = TG.totals(TG.build_jobs(projection_world()[0], stats, dict(cfg, ff="batch_float32")), cfg)
        assert TG.totals(TG.build_jobs(rates, stats, cfg), cfg)["gpu_h"] > batch["gpu_h"] + 20


@pytest.mark.parametrize("fault", ["missing_rate", "nan_rate", "truncated_set", "wrong_set_size", "invalid_degrade", "no_cpu_cores"])
def test_projection_refuses_incomplete_or_invalid_inputs(fault):
    rates, stats, cfg = projection_world()
    if fault == "missing_rate":
        del rates["gpu"]["ff"]["batch_float32"]["startup_s"]
    elif fault == "nan_rate":
        rates["cpu"]["energy_s"]["v"] = float("nan")
    elif fault == "truncated_set":
        stats["testB"]["atoms"] = [40]
    elif fault == "wrong_set_size":
        stats["testB"]["n"] = 999
    elif fault == "invalid_degrade":
        cfg["degrade"] = {7}
    else:
        cfg["cores"] = cfg["gpu_job_cores"]
    with pytest.raises(ValueError):
        TG.build_jobs(rates, stats, cfg)


def test_nonzero_child_exit_is_never_a_success(tmp_path):
    rc, _ = TG.run_concurrent([(str(tmp_path), [sys.executable, "-c", "import sys; sys.exit(3)"])],
                              str(tmp_path / "log"), dict(os.environ))
    assert rc != 0


@pytest.mark.parametrize("xs,ys", [([], []), ([1, 2], [1]), ([1], [float("nan")]), ([1], [0])])
def test_invalid_rate_observations_are_rejected(xs, ys):
    with pytest.raises(ValueError):
        TG.fit_power(xs, ys)


def test_timing_plan_uses_train_val_only_and_no_match_commands():
    args = SimpleNamespace(workers=2, gpu="0", concurrency=1, ivp_cpb=[1, 8], dry=True)
    for spec in TG.measurements(args).values():
        commands = spec.get("cmds", []) + spec.get("prep", []) + [c for _, _, c in spec.get("runs", [])]
        for _, argv in commands:
            assert "match" not in argv and "--step9" not in argv
            if "--set" in argv:
                selected = argv[argv.index("--set") + 1]
                assert selected in ("train", "val") or "/_smoke/val" in selected


def test_box_shell_scripts_have_valid_bash_syntax():
    import shutil
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    for name in ("n3_box_setup.sh", "n3_box_cleanup.sh"):
        result = subprocess.run([bash, "-n", str(REPO / "scripts" / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def measurement_world(tmp_path, monkeypatch):
    repo, out = tmp_path / "repo", tmp_path / "timing"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("n3_timing_gate", "n3_common", "n3_ours", "n3_old_symmc", "n3_classical", "n3_ffstar",
                 "n3_mcf_run", "n3_mcf_lib", "g2_asym_baselines", "g2_learned_flow", "g2_rank",
                 "n3_set_cache", "n3_harness", "n3_chirality"):
        (scripts / (name + ".py")).write_text("# original synthetic source\n")
    (scripts / "g2_asym_baselines.py").write_text("from timing_helper import energy\n")
    (scripts / "timing_helper.py").write_text("# original transitive energy implementation\n")
    config = repo / "external/mcf/molcrystalflow/configs/ours_inference_R.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("num_steps: 50\n")
    out.mkdir()
    monkeypatch.setattr(TG, "REPO", str(repo))
    monkeypatch.setattr(TG, "OUT", str(out))
    args = SimpleNamespace(workers=2, gpu="0", concurrency=1, ivp_cpb=[1, 8])
    spec = {"kind": "ivp_train", "env": "symmc", "gpu": False, "cmds": []}
    record = {"name": "ivp_train_pack", "kind": "ivp_train", "env": "symmc", "host": TG.platform.node(),
              "workers": 2, "ok": True, "epoch_s": 1.0, "measurement_spec": spec,
              "measurement_settings": TG.measurement_settings(args), "fingerprint": TG.measurement_fingerprint(spec, args)}
    return repo, out, config, record


@pytest.mark.parametrize("change", ["harness", "energy_implementation", "transitive_dependency", "hydra_config", "recorded_settings", "legacy_record", "other_host"])
def test_direct_rates_cannot_publish_a_stale_measurement(tmp_path, monkeypatch, change):
    repo, out, config, record = measurement_world(tmp_path, monkeypatch)
    measurement = out / "ivp_train_pack.json"
    measurement.write_text(json.dumps(record))
    TG.cmd_rates(SimpleNamespace())
    published = (out / "rates.json").read_bytes()
    assert json.loads(published)["gpu"]["ivp_train_epoch_s"]["pack"] == 1.0
    if change == "harness":
        (repo / "scripts/n3_ffstar.py").write_text("# changed after the measurement\n")
    elif change == "energy_implementation":
        (repo / "scripts/g2_asym_baselines.py").write_text("# changed LJ energy/press implementation\n")
    elif change == "transitive_dependency":
        (repo / "scripts/timing_helper.py").write_text("# changed indirect dependency\n")
    elif change == "hydra_config":
        config.write_text("num_steps: 100\n")
    elif change == "recorded_settings":
        record["measurement_settings"]["concurrency"] = 2
    elif change == "legacy_record":
        del record["measurement_spec"]
    else:
        record["host"] = "a different machine"
    measurement.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="rates refused"):
        TG.cmd_rates(SimpleNamespace())
    assert (out / "rates.json").read_bytes() == published


def wheel_pages():
    cu128, cu130 = "https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu130"
    link = lambda name: '<a href="' + name.replace("+", "%2B") + '#sha256=abc">wheel</a>'
    return {
        cu128 + "/torch/": "".join(link(f"torch-{v}+cu128-cp312-cp312-manylinux_2_28_x86_64.whl")
                                   for v in ("2.7.1", "2.11.0")),
        cu128 + "/torchvision/": link("torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"),
        cu130 + "/torch/": link("torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"),
        "https://data.pyg.org/whl/torch-2.7.0+cu128.html": "".join(
            link(f"{package}-{version}+pt27cu128-cp312-cp312-linux_x86_64.whl")
            for package, version in (("torch_scatter", "2.1.2"), ("torch_cluster", "1.6.3"))),
        "https://pypi.org/pypi/fairchem-core/2.23.0/json": json.dumps(
            {"info": {"requires_dist": ["torch~=2.13.0"]}, "urls": [
                {"packagetype": "bdist_wheel", "filename": "fairchem_core-2.23.0-py3-none-any.whl",
                 "url": "https://files.pythonhosted.org/fairchem_core-2.23.0-py3-none-any.whl"}]}),
    }


def test_wheel_preflight_reads_only_metadata_and_keeps_frozen_versions(monkeypatch):
    pages, called = wheel_pages(), []
    def read(url, timeout):
        called.append(url)
        return pages[url]
    monkeypatch.setattr(BM, "read_url", read)
    checked = BM.check_wheels()
    assert checked["available"] and checked["min_driver_cuda"] == "13.0"
    assert "2.13.0%2Bcu130" in checked["wheels"]["ff torch"][0]
    assert "2.11.0%2Bcu128" in checked["wheels"]["symmc torch"][0]
    assert checked["fairchem_torch_requirement"] == "torch~=2.13.0"
    assert len(called) == 5 and not any(url.endswith(".whl") for url in called)


@pytest.mark.parametrize("fault", ["missing_ff", "wrong_python", "wrong_architecture", "wrong_pyg", "incompatible_fairchem"])
def test_wheel_preflight_fails_closed_on_missing_or_incompatible_pins(monkeypatch, fault):
    pages = wheel_pages()
    ff = "https://download.pytorch.org/whl/cu130/torch/"
    if fault == "missing_ff":
        pages[ff] = ""
    elif fault == "wrong_python":
        pages[ff] = pages[ff].replace("cp312", "cp311")
    elif fault == "wrong_architecture":
        pages[ff] = pages[ff].replace("x86_64", "aarch64")
    elif fault == "wrong_pyg":
        pyg = "https://data.pyg.org/whl/torch-2.7.0+cu128.html"
        pages[pyg] = pages[pyg].replace("torch_cluster-1.6.3", "torch_cluster-1.6.4")
    else:
        p = "https://pypi.org/pypi/fairchem-core/2.23.0/json"
        pages[p] = pages[p].replace("2.13.0\"", "2.11.0\"")
    monkeypatch.setattr(BM, "read_url", lambda url, timeout: pages[url])
    with pytest.raises(ValueError):
        BM.check_wheels()


def test_setup_dry_plan_uses_ff_cu130_preflight_and_driver_requirement(tmp_path):
    import shutil
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    root = tmp_path / "empty_root"
    result = subprocess.run([bash, str(REPO / "scripts/n3_box_setup.sh"), "all"], capture_output=True, text=True,
                            env=dict(os.environ, N3_DRY_RUN="1", N3_BOX_ROOT=str(root), N3_SYMMC_TORCH="torch==2.11.0",
                                     N3_SYMMC_INDEX="https://download.pytorch.org/whl/cu128"))
    assert result.returncode == 0, result.stderr
    assert "driver CUDA >= 13.0" in result.stdout
    assert "n3_box_manifest.py wheels" in result.stdout
    assert "torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130" in result.stdout
    assert "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128" in result.stdout
    assert "torch-scatter==2.1.2+pt27cu128 torch-cluster==1.6.3+pt27cu128" in result.stdout
    assert not root.exists()


@pytest.mark.parametrize("banner, libcuda, glibc, ok, said", [
    ("CUDA Version: 13.0", "", "2.35", True, "driver CUDA: 13.0"),          # Vast boxes: banner present
    ("CUDA UMD Version: 13.4", "13.4", "2.35", True, "driver CUDA: 13.4"),  # Anvil driver 615: libcuda answers
    ("", "12.8", "2.35", False, "driver CUDA 12.8 < 13.0"),
    ("", "", "2.35", False, "driver CUDA version unreadable"),
    ("CUDA Version: 13.0", "", "2.28", False, "glibc 2.28 < 2.32"),         # Anvil Rocky 8 host (job 21318021)
])
def test_setup_check_reads_driver_cuda_without_the_nvidia_smi_banner(tmp_path, banner, libcuda, glibc, ok, said):
    import platform
    import shutil
    bash = shutil.which("bash")
    if not bash or platform.system() != "Linux":
        pytest.skip("check mode runs only on Linux")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text('#!/bin/sh\ncase "$1" in\n'
                                     '  --query-gpu=name) echo "NVIDIA A100-SXM4-40GB" ;;\n'
                                     '  --query-gpu=*) echo "0, NVIDIA A100-SXM4-40GB, 40960 MiB, 615.71.09" ;;\n'
                                     f'  *) echo "| NVIDIA-SMI 615.71.09   Driver Version: 615.71.09   {banner} |" ;;\n'
                                     'esac\n')
    (fake / "python3").write_text(f'#!/bin/sh\n[ -n "{libcuda}" ] && echo "{libcuda}"\n')
    (fake / "getconf").write_text(f'#!/bin/sh\necho "glibc {glibc}"\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    result = subprocess.run([bash, str(REPO / "scripts/n3_box_setup.sh"), "check"], capture_output=True, text=True,
                            env=dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", N3_BOX_ROOT=str(tmp_path),
                                     N3_MIN_DISK_GB="0"))
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    assert said in result.stdout + result.stderr


def project_args(tmp_path, rates, stats, cfg):
    rate_path, stats_path = tmp_path / "rates.json", tmp_path / "set_stats.json"
    rate_path.write_text(json.dumps(rates))
    stats_path.write_text(json.dumps(stats))
    return SimpleNamespace(**{k: v for k, v in cfg.items() if k != "degrade"}, rates=str(rate_path),
                           stats=str(stats_path), testB=1000, cap_usd=1000, out=str(tmp_path / "projection.json"))


def publish_complete_synthetic_measurements(out, record):
    cpu = {name: 0.01 for name in ("energy_s", "press_s", "fit_primary_s", "rms_mcf05_s", "rms_mcf08_s", "rms_mcf10_s")}
    entries = [(f"mcf_train_{a}_p0", "mcf_train", {"epoch_s": 1.0, "concurrency": 1}) for a in ("R", "A")]
    entries += [(f"mcf_infer_{a}_n{n}", "mcf_infer", {"call_s": 1.0 + 0.1 * n})
                for a, n in (("R", 10), ("R", 50), ("A", 50))]
    entries += [
        ("ivp_train_pack", "ivp_train", {"epoch_s": 2.0}),
        ("ivp_sample_pack_b8", "ivp_sample", {"gpu_s_per_crystal": 0.1}),
        ("ff_batch_float32", "ff_time", {"rows": [{"atoms": 40, "sec_per_struct_batched": 0.001,
             "sec_single": 0.01, "warm_sec": 10.0}] * 20}),
        ("ff_min_truth_float32", "ff_min", {"rows": [{"nev": [10], "conv": [True]}] * 20}),
        ("cpu", "cpu", {"rows": [{"atoms": 40, **cpu}] * 20}),
        ("lj_screen", "screen", {"rows": [{"refcode_idx": i, "sec": 0.01} for i in range(20)]}),
        ("ours_sample", "ours", {"cpu_s_per_crystal": 0.01}),
    ]
    for name, kind, values in entries:
        r = copy.deepcopy(record)
        env = "mcf" if name.startswith("mcf") else "symmc"
        spec = {"kind": kind, "env": env, "gpu": kind not in ("cpu", "screen", "ours"), "cmds": []}
        r.update(name=name, kind=kind, env=env, measurement_spec=spec, **values)
        r["fingerprint"] = TG.measurement_fingerprint(spec, SimpleNamespace(**r["measurement_settings"]))
        (out / (name + ".json")).write_text(json.dumps(r))
    TG.cmd_rates(SimpleNamespace())
    return json.loads((out / "rates.json").read_text())


@pytest.mark.parametrize("change", ["harness", "energy_implementation", "transitive_dependency", "hydra_config", "missing_identity"])
def test_project_rechecks_code_after_rate_publication_without_requiring_same_host(tmp_path, monkeypatch, change):
    repo, out, config, record = measurement_world(tmp_path, monkeypatch)
    rates = publish_complete_synthetic_measurements(out, record)
    _, stats, cfg = projection_world()
    args = project_args(tmp_path, rates, stats, cfg)
    # Projection runs on the laptop after verified pullback from the timing host.
    monkeypatch.setattr(TG.platform, "node", lambda: "different projection laptop")
    TG.cmd_project(args)
    published = Path(args.out).read_bytes()
    projection = json.loads(published)
    assert projection["measured"] and projection["approval_eligible"]
    assert not projection["assumed"] and not projection["approved"]
    if change == "harness":
        (repo / "scripts/n3_ours.py").write_text("# changed after rate publication\n")
    elif change == "energy_implementation":
        (repo / "scripts/g2_asym_baselines.py").write_text("# changed LJ energy/press implementation\n")
    elif change == "transitive_dependency":
        (repo / "scripts/timing_helper.py").write_text("# changed indirect dependency\n")
    elif change == "hydra_config":
        config.write_text("num_steps: 100\n")
    else:
        del rates["code_identity"]
        Path(args.rates).write_text(json.dumps(rates))
    with pytest.raises(SystemExit, match="projection refused"):
        TG.cmd_project(args)
    assert Path(args.out).read_bytes() == published


def test_explicit_prior_projection_is_labeled_assumed_and_never_approval_eligible(tmp_path):
    rates, stats, cfg = projection_world()
    rates.update(assumed=True, measured=["a prior report"], code_identity={"stale": "source"})
    args = project_args(tmp_path, rates, stats, cfg)
    TG.cmd_project(args)
    projection = json.loads(Path(args.out).read_text())
    assert projection["assumed"]
    assert not any(projection[k] for k in ("measured", "approval_eligible", "approved"))
    del rates["assumed"]
    rates["measured"] = []
    Path(args.rates).write_text(json.dumps(rates))
    with pytest.raises(SystemExit, match="explicit assumed=True"):
        TG.cmd_project(args)


def fake_infer_measurement(tmp_path, monkeypatch, failed, printed=True, rc=1):
    """Run TG.measure on a stand-in for `n3_mcf_run.py infer` that leaves a check.json and exits rc."""
    repo, out, _, _ = measurement_world(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "check.json").write_text(json.dumps({
        "pass": not failed, "failed": failed, "info": {"nonfinite_draws": 631, "crystals_without_finite_draw": 13}}))
    tail = {"run_dir": str(run_dir), "check_pass": not failed, "sampling_sec": 36.0, "resid_sec": 5.4}
    code = (f"print({json.dumps(json.dumps(tail))})\n" if printed else "") + f"raise SystemExit({rc})"
    spec = {"kind": "mcf_infer", "env": "mcf", "gpu": False, "n": 50, "cmds": [(str(repo), [sys.executable, "-c", code])]}
    args = SimpleNamespace(workers=2, gpu="0", concurrency=1, ivp_cpb=[1, 8])
    return TG.measure("mcf_infer_A_n50", spec, args)


def test_timing_infer_accepts_only_smoke_weight_resid_check_failures(tmp_path, monkeypatch):
    res = fake_infer_measurement(tmp_path, monkeypatch, sorted(TG.SMOKE_WEIGHT_CHECKS))
    assert res["ok"] and res["check_pass"] is False and res["call_s"] == 41.4
    assert set(res["check_failed"]) == TG.SMOKE_WEIGHT_CHECKS
    assert res["check_info"] == {"nonfinite_draws": 631, "crystals_without_finite_draw": 13}


def test_timing_infer_passing_check_is_recorded(tmp_path, monkeypatch):
    res = fake_infer_measurement(tmp_path, monkeypatch, [], rc=0)
    assert res["ok"] and res["check_pass"] is True and "check_failed" not in res


@pytest.mark.parametrize("failed", [["resid_finite_on_finite_draws", "config_num_timesteps"], ["clamp_lattice_atol_1e-5"]])
def test_timing_infer_rejects_other_check_failures(tmp_path, monkeypatch, failed):
    res = fake_infer_measurement(tmp_path, monkeypatch, failed)
    assert not res["ok"] and "infer check did not pass" in res["error"]


def test_timing_infer_rejects_a_crash_without_its_check_line(tmp_path, monkeypatch):
    res = fake_infer_measurement(tmp_path, monkeypatch, sorted(TG.SMOKE_WEIGHT_CHECKS), printed=False)
    assert not res["ok"] and res["error"].startswith("failed (1)")


def can_symlink(tmp):
    try:
        os.symlink(tmp / "absent", tmp / "probe_link")
        return True
    except (OSError, NotImplementedError):
        return False


def pullback_world(tmp_path, monkeypatch):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks need privileges on this host (run on Linux / the box)")
    root, man = tmp_path / "box", manifest(step=4)
    (root / "data/n3_sets").mkdir(parents=True)
    (root / "data/n3_sets/train.pt").write_bytes(b"synthetic TRAIN input")
    BM.write_manifest_files(man, root / BM.MAN_SUBDIR)
    wb = root / "n3_mcf/_timing/wandb/wandb"
    (wb / "offline-run-1/logs").mkdir(parents=True)
    (wb / "offline-run-1/logs/debug.log").write_text("run log")
    (root / ".n3_box/cache/wandb").mkdir(parents=True)
    (root / ".n3_box/cache/wandb/core.log").write_text("core log")
    monkeypatch.setenv("N3_MANIFEST_DIR", str(tmp_path / "manifests"))
    return root, wb


def test_pullback_lists_in_root_wandb_symlinks_without_packing_them(tmp_path, monkeypatch):
    root, wb = pullback_world(tmp_path, monkeypatch)
    os.symlink("offline-run-1", wb / "latest-run")
    os.symlink("offline-run-1/logs/debug.log", wb / "debug.log")
    os.symlink(root / ".n3_box/cache/wandb/core.log", wb / "offline-run-1/logs/debug-core.log")
    BM.cmd_pullback(SimpleNamespace(root=str(root), out=str(tmp_path / "return"), label="t"))
    with tarfile.open(tmp_path / "return/n3_pullback_t.tar") as tf:
        members = {m.name: m for m in tf.getmembers()}
    assert "n3_mcf/_timing/wandb/wandb/offline-run-1/logs/debug.log" in members
    assert all(m.isfile() for m in members.values())
    links = json.loads((tmp_path / "manifests/pullback_t.skipped_symlinks.json").read_text())
    assert {x["path"] for x in links} == {"n3_mcf/_timing/wandb/wandb/latest-run", "n3_mcf/_timing/wandb/wandb/debug.log",
                                         "n3_mcf/_timing/wandb/wandb/offline-run-1/logs/debug-core.log"}


def test_pullback_refuses_a_symlink_leaving_the_box_root(tmp_path, monkeypatch):
    root, wb = pullback_world(tmp_path, monkeypatch)
    (tmp_path / "outside.txt").write_text("not box data")
    os.symlink(tmp_path / "outside.txt", wb / "offline-run-1/logs/debug-core.log")
    with pytest.raises(SystemExit, match="leaves the box root"):
        BM.cmd_pullback(SimpleNamespace(root=str(root), out=str(tmp_path / "return"), label="t"))


def resid_world():
    import torch
    nan = float("nan")
    fin = torch.tensor([[True, True, False], [True, True, True]])        # [S=2, n=3]
    r = torch.tensor([[0.1, nan, nan], [0.2, 0.3, 0.4]], dtype=torch.float64)
    return torch, nan, fin, r


def test_resid_nonfinite_on_a_finite_draw_is_recorded_not_failed():
    import n3_mcf_run as MR
    torch, nan, fin, r = resid_world()
    checks, info = MR.resid_checks(r, fin, r.clone(), ["AAA", "BBB", "CCC"])
    assert all(checks.values())
    assert info == {"resid_nonfinite_on_finite_draws": 1, "resid_nonfinite_on_finite_refcodes": ["BBB"]}


def test_resid_checks_still_fail_on_real_defects():
    import n3_mcf_run as MR
    torch, nan, fin, r = resid_world()
    rec = r.clone()
    rec[1, 2] += 1e-6                                                      # stored RESID not reproducible
    assert not MR.resid_checks(r, fin, rec, ["A", "B", "C"])[0]["resid_recomputes_from_saved_rotmats"]
    rec = r.clone()
    rec[0, 1] = 0.5                                                        # recomputation finite where stored NaN
    assert not MR.resid_checks(r, fin, rec, ["A", "B", "C"])[0]["resid_recomputes_from_saved_rotmats"]
    bad = r.clone()
    bad[0, 2] = 0.7                                                        # non-finite draw with a finite RESID
    assert not MR.resid_checks(bad, fin, bad.clone(), ["A", "B", "C"])[0]["resid_nan_on_nonfinite_draws"]
    assert not MR.resid_checks(r, fin, None, ["A", "B", "C"])[0]["resid_recomputes_from_saved_rotmats"]
