from __future__ import annotations

import tarfile
from pathlib import Path

from newsbag.input_staging import stage_inputs_to_manifest


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stage_inputs_single_file_and_directory(tmp_path: Path) -> None:
    img1 = tmp_path / "inputs" / "page01.png"
    img2 = tmp_path / "inputs" / "nested" / "page02.jpg"
    _touch(img1)
    _touch(img2)

    out_manifest = tmp_path / "run" / "manifests" / "images.txt"
    staging_dir = tmp_path / "run" / "staged_inputs"

    report = stage_inputs_to_manifest(
        inputs=[str(img1), str(img1.parent)],
        output_manifest=out_manifest,
        staging_dir=staging_dir,
        recursive=True,
        max_pages=0,
    )

    lines = [x for x in out_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert report.image_count == 2
    assert str(img1.resolve()) in lines
    assert str(img2.resolve()) in lines
    assert (staging_dir / "staging_summary.json").exists()


def test_stage_inputs_tar_archive_extracts_and_indexes_images(tmp_path: Path) -> None:
    src = tmp_path / "src_archive"
    _touch(src / "a.png")
    _touch(src / "b.txt", "not an image")
    _touch(src / "nested" / "c.tif")
    _touch(src / "._a.png")
    _touch(src / "__MACOSX" / "nested" / "d.png")

    archive = tmp_path / "batch_pages.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="batch_pages")

    out_manifest = tmp_path / "run" / "manifests" / "images.txt"
    staging_dir = tmp_path / "run" / "staged_inputs"
    report = stage_inputs_to_manifest(
        inputs=[str(archive)],
        output_manifest=out_manifest,
        staging_dir=staging_dir,
        recursive=True,
        max_pages=0,
    )

    lines = [x for x in out_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert report.archive_count == 1
    assert report.image_count == 2
    assert all("/archives/" in x for x in lines)
    assert not any("/._" in x or "/__MACOSX/" in x for x in lines)
    assert any(x.endswith("a.png") for x in lines)
    assert any(x.endswith("c.tif") for x in lines)


def test_stage_inputs_manifest_file_with_relative_paths(tmp_path: Path) -> None:
    img1 = tmp_path / "data" / "p1.png"
    img2 = tmp_path / "data" / "p2.jpeg"
    _touch(img1)
    _touch(img2)

    manifest_in = tmp_path / "seed" / "input_list.txt"
    manifest_in.parent.mkdir(parents=True, exist_ok=True)
    manifest_in.write_text("../data/p1.png\n../data/p2.jpeg\n", encoding="utf-8")

    out_manifest = tmp_path / "run" / "manifests" / "images.txt"
    staging_dir = tmp_path / "run" / "staged_inputs"
    report = stage_inputs_to_manifest(
        inputs=[str(manifest_in)],
        output_manifest=out_manifest,
        staging_dir=staging_dir,
        recursive=False,
        max_pages=1,
    )

    lines = [x for x in out_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert report.manifest_input_count == 1
    assert report.image_count == 1
    assert lines[0] in {str(img1.resolve()), str(img2.resolve())}
