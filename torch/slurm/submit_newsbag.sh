#!/bin/bash
# Generic entrypoint for submitting newspaper parsing runs from mixed inputs.
#
# Delegates to submit_newsbag_from_dir.sh, which now accepts:
# - single image files
# - directories
# - manifest files
# - tar/zip archives
#
# Example:
#   bash torch/slurm/submit_newsbag.sh --input /path/to/page.png --gpu split

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/submit_newsbag_from_dir.sh" "$@"
