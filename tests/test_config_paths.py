from __future__ import annotations

import json
from pathlib import Path

from newsbag.config import load_config


def test_load_config_resolves_repo_relative_script_paths_from_run_scoped_config(tmp_path: Path) -> None:
    """Torch flows write per-run configs under RUN_DIR/manifests.

    The config loader should *not* rewrite "src/..." into a bogus absolute path under RUN_DIR.
    It should instead resolve relative to the repo checkout (CWD during sbatch runs).
    """

    run_dir = tmp_path / "runs" / "layout_bagging_20260210_000000"
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True)

    cfg_path = manifests / "config.input.json"
    payload = {
        "manifest_path": str(manifests / "images.txt"),
        "run_root": str(tmp_path / "runs"),
        "run_name": "layout_bagging",
        "paddleocr_bin": "paddleocr",
        "device_order": ["cpu"],
        "cpu_threads": 2,
        "resume": True,
        "allow_gpu_to_cpu_fallback": False,
        "dell": {
            "enabled": True,
            "python_bin": "python3",
            "script_path": "src/newsbag/runners/run_dell_layout_only.py",
            "model_path": "/abs/model.onnx",
            "label_map_path": "/abs/label_map_layout.json",
            "provider": "cpu",
        },
        "mineru": {
            "enabled": True,
            "python_bin": "python3",
            "script_path": "src/newsbag/runners/run_mineru25_layout.py",
            "require_cuda": False,
        },
    }
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cfg = load_config(cfg_path)

    # These should resolve against repo root (CWD) because they are repo-relative "src/..." paths.
    assert Path(cfg.dell.script_path).exists(), f"dell.script_path did not resolve to an existing file: {cfg.dell.script_path}"
    assert Path(cfg.mineru.script_path).exists(), f"mineru.script_path did not resolve to an existing file: {cfg.mineru.script_path}"

