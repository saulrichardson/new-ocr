#!/bin/bash
# Submit newspaper inputs through the full pipeline on Torch.
#
# This script:
# 1) Creates a run directory (RUN_DIR) under /scratch/$USER/paddleocr_vl15/runs
# 2) Stages mixed input types and generates an image manifest
# 3) Writes a per-run config.json (based on configs/pipeline.torch.json)
# 4) Submits GPU inference + CPU fusion/review + GPU transcription (afterok chain)
#
# Usage (on Torch login):
#   bash torch/slurm/submit_newsbag_from_dir.sh --input /path/to/scans --recursive --gpu l40s
#
# Flags:
#   --input <path>      Required; repeatable. Each may be:
#                      - a single image file
#                      - a directory of images
#                      - a manifest text file (.txt/.manifest/.lst/.tsv/.csv)
#                      - an archive (.tar/.tar.gz/.tgz/.zip)
#                      (also supports comma-separated paths)
#   --input-dir <dir>   Deprecated alias for --input.
#   --recursive         Optional. Recurse subdirectories.
#   --gpu l40s|h200|split  Optional. Default: l40s.
#   --max-pages N       Optional. Limit the manifest to N images (for smoke tests).
#   --run-dir <dir>     Optional. Explicit run dir (default: /scratch/.../runs/layout_bagging_<ts>)
#   --stages <csv>      Optional. Override infer stages for the single GPU job mode (l40s/h200).
#
# Optional env:
#   BASE, PROJECT_ROOT, TEMPLATE_CONFIG_JSON

set -euo pipefail

INPUTS=()
RECURSIVE=0
GPU_KIND="l40s"
RUN_DIR=""
MAX_PAGES=0
STAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUTS+=("${2:-}")
      shift 2
      ;;
    --input-dir)
      INPUTS+=("${2:-}")
      shift 2
      ;;
    --recursive)
      RECURSIVE=1
      shift 1
      ;;
    --gpu)
      GPU_KIND="${2:-l40s}"
      shift 2
      ;;
    --stages)
      STAGES="${2:-}"
      shift 2
      ;;
    --max-pages)
      MAX_PAGES="${2:-0}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,180p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "ERROR: at least one --input path is required" >&2
  exit 2
fi

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
TEMPLATE_CONFIG_JSON="${TEMPLATE_CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$BASE/runs/layout_bagging_$STAMP"
fi

mkdir -p "$RUN_DIR/manifests"

MANIFEST="$RUN_DIR/manifests/images.input.txt"
OUT_CONFIG="$RUN_DIR/manifests/config.input.json"
STAGING_DIR="$RUN_DIR/staged_inputs"

echo "[submit] input_count=${#INPUTS[@]}"
for i in "${!INPUTS[@]}"; do
  echo "[submit] input[$((i+1))]=${INPUTS[$i]}"
done
echo "[submit] recursive=$RECURSIVE"
echo "[submit] gpu_kind=$GPU_KIND"
echo "[submit] max_pages=$MAX_PAGES"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] staging_dir=$STAGING_DIR"
if [[ -n "$STAGES" ]]; then
  echo "[submit] stages=$STAGES"
fi

STAGE_INPUT_ARGS=(--output "$MANIFEST" --staging-dir "$STAGING_DIR" --max-pages "$MAX_PAGES")
if [[ "$RECURSIVE" -eq 1 ]]; then
  STAGE_INPUT_ARGS+=(--recursive)
fi
for in_path in "${INPUTS[@]}"; do
  STAGE_INPUT_ARGS+=(--input "$in_path")
done

# scripts/stage_inputs.py is stdlib-only; use python3 from login node.
python3 "$PROJECT_ROOT/scripts/stage_inputs.py" "${STAGE_INPUT_ARGS[@]}"

COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
echo "[submit] manifest_count=$COUNT"
if [[ "$COUNT" -eq 0 ]]; then
  echo "ERROR: manifest is empty after staging (no images found)" >&2
  exit 2
fi

# Create run-scoped config so this run is self-contained and reproducible.
python3 - <<PY
import json
from pathlib import Path

tmpl=Path("$TEMPLATE_CONFIG_JSON")
payload=json.loads(tmpl.read_text(encoding="utf-8"))
payload["manifest_path"]=str(Path("$MANIFEST"))
payload["run_root"]=str(Path("$BASE")/"runs")
payload["run_name"]="layout_bagging"
Path("$OUT_CONFIG").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
print("[submit] wrote config:", "$OUT_CONFIG")
PY

GPU_SCRIPT="$PROJECT_ROOT/torch/slurm/newsbag_infer_l40s.sbatch"
if [[ "$GPU_KIND" == "h200" ]]; then
  GPU_SCRIPT="$PROJECT_ROOT/torch/slurm/newsbag_infer_h200.sbatch"
elif [[ "$GPU_KIND" == "split" ]]; then
  GPU_SCRIPT=""
fi

export RUN_DIR
export CONFIG_JSON="$OUT_CONFIG"
export GPU_SCRIPT

cd "$PROJECT_ROOT"

if [[ "$GPU_KIND" == "split" ]]; then
  # Split-GPU submission: Paddle on L40S, Dell+MinerU on H200, then CPU fuse/review.
  bash torch/slurm/submit_newsbag_split_gpu.sh
  exit 0
fi

if [[ -z "$STAGES" ]]; then
  if [[ "$GPU_KIND" == "h200" ]]; then
    STAGES="dell,mineru"
  else
    STAGES="paddle_layout,paddle_vl15,dell,mineru"
  fi
fi

if [[ "$GPU_KIND" == "h200" ]]; then
  if echo ",$STAGES," | grep -qE ",paddle_layout,|,paddle_vl15,"; then
    echo "[submit] ERROR: Paddle stages requested with --gpu h200 ($STAGES)." >&2
    echo "[submit] Use --gpu split (recommended) or --gpu l40s." >&2
    exit 3
  fi
fi

export STAGES
bash torch/slurm/submit_newsbag_full.sh
