from __future__ import annotations

import hashlib
import json
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable, List, Set

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MANIFEST_EXTS = {".txt", ".manifest", ".lst", ".tsv", ".csv"}


def is_archive_path(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar.bz2")
        or name.endswith(".tbz")
        or name.endswith(".tbz2")
        or name.endswith(".tar.xz")
        or name.endswith(".txz")
        or name.endswith(".zip")
    )


def _safe_child(base_dir: Path, member_name: str) -> Path:
    target = (base_dir / member_name).resolve()
    base_resolved = base_dir.resolve()
    if os.path.commonpath([str(base_resolved), str(target)]) != str(base_resolved):
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return target


def _is_ignored_archive_member(member_name: str) -> bool:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    if not parts:
        return True
    for part in parts:
        if part == "__MACOSX":
            return True
        if part.startswith("."):
            return True
        if part.startswith("._"):
            return True
    return False


def _is_ignored_path_under_root(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
    except Exception:
        parts = path.parts
    if not parts:
        return True
    for part in parts:
        if part == "__MACOSX":
            return True
        if part.startswith("."):
            return True
        if part.startswith("._"):
            return True
    return False


def _iter_image_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [
            p.resolve()
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and not _is_ignored_path_under_root(p, root)
        ]
    else:
        files = [
            p.resolve()
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and not _is_ignored_path_under_root(p, root)
        ]
    files.sort(key=lambda p: str(p).lower())
    return files


def _archive_extract_dir(archive: Path, staging_dir: Path) -> Path:
    digest = hashlib.sha1(str(archive.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = archive.name.lower()
    for bad in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".tbz", ".txz", ".tar", ".zip"):
        if stem.endswith(bad):
            stem = stem[: -len(bad)]
            break
    stem = stem.strip().replace(" ", "_") or "archive"
    out_dir = (staging_dir / "archives" / f"{stem}_{digest}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _extract_archive(archive: Path, staging_dir: Path) -> Path:
    out_dir = _archive_extract_dir(archive=archive, staging_dir=staging_dir)
    name = archive.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if _is_ignored_archive_member(info.filename):
                    continue
                target = _safe_child(out_dir, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    dst.write(src.read())
        return out_dir

    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if _is_ignored_archive_member(member.name):
                continue
            target = _safe_child(out_dir, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as dst:
                dst.write(src.read())
    return out_dir


@dataclass
class StageReport:
    inputs: List[str]
    manifest_path: str
    staging_dir: str
    recursive: bool
    max_pages: int
    image_count: int
    archive_count: int
    manifest_input_count: int
    directory_input_count: int
    file_input_count: int
    image_files: List[str] = field(default_factory=list)


def _expand_one(
    item: Path,
    *,
    recursive: bool,
    staging_dir: Path,
    seen_manifest_files: Set[Path],
    report: StageReport,
) -> List[Path]:
    p = item.expanduser()
    if not p.is_absolute():
        p = p.resolve()
    else:
        p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    if p.is_dir():
        report.directory_input_count += 1
        return _iter_image_files(p, recursive=recursive)

    if p.suffix.lower() in IMAGE_EXTS:
        report.file_input_count += 1
        return [p]

    if is_archive_path(p):
        report.archive_count += 1
        extracted = _extract_archive(archive=p, staging_dir=staging_dir)
        return _iter_image_files(extracted, recursive=True)

    if p.suffix.lower() in MANIFEST_EXTS:
        if p in seen_manifest_files:
            return []
        seen_manifest_files.add(p)
        report.manifest_input_count += 1
        out: List[Path] = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            child = Path(line)
            if not child.is_absolute():
                child = (p.parent / child).resolve()
            out.extend(
                _expand_one(
                    child,
                    recursive=recursive,
                    staging_dir=staging_dir,
                    seen_manifest_files=seen_manifest_files,
                    report=report,
                )
            )
        return out

    raise ValueError(
        f"Unsupported input path type: {p}. "
        "Provide image file, directory, manifest(.txt/.manifest/.lst/.tsv/.csv), or archive(.tar/.tar.gz/.tgz/.zip/...)."
    )


def stage_inputs_to_manifest(
    *,
    inputs: Iterable[str],
    output_manifest: Path,
    staging_dir: Path,
    recursive: bool,
    max_pages: int = 0,
) -> StageReport:
    in_list = [str(x).strip() for x in inputs if str(x).strip()]
    if not in_list:
        raise ValueError("No input paths were provided.")

    output_manifest = output_manifest.expanduser().resolve()
    staging_dir = staging_dir.expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    report = StageReport(
        inputs=in_list,
        manifest_path=str(output_manifest),
        staging_dir=str(staging_dir),
        recursive=bool(recursive),
        max_pages=int(max_pages),
        image_count=0,
        archive_count=0,
        manifest_input_count=0,
        directory_input_count=0,
        file_input_count=0,
    )

    files: List[Path] = []
    seen_manifest_files: Set[Path] = set()
    for entry in in_list:
        parts = [x.strip() for x in entry.split(",") if x.strip()]
        for part in parts:
            files.extend(
                _expand_one(
                    Path(part),
                    recursive=recursive,
                    staging_dir=staging_dir,
                    seen_manifest_files=seen_manifest_files,
                    report=report,
                )
            )

    deduped: List[Path] = []
    seen = set()
    for p in files:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    deduped.sort(key=lambda p: str(p).lower())
    if max_pages > 0:
        deduped = deduped[:max_pages]

    output_manifest.write_text(
        "\n".join(str(p) for p in deduped) + ("\n" if deduped else ""),
        encoding="utf-8",
    )
    report.image_count = len(deduped)
    report.image_files = [str(p) for p in deduped]

    summary_path = staging_dir / "staging_summary.json"
    summary_path.write_text(json.dumps(report.__dict__, indent=2) + "\n", encoding="utf-8")
    return report
