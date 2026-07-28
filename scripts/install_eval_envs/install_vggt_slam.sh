#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
ensure_conda_env vggt_slam 3.11
ENV_PREFIX="$(env_path vggt_slam)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.3.1+cu121" "0.18.1+cu121" \
  "https://download.pytorch.org/whl/cu121"
"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/vggt-slam-h100.txt"

# Only these two upstream packages are used by the benchmark solver. SAM3 and
# Perception Encoder are optional object-detection features and are omitted.
"${ENV_PREFIX}/bin/python" -m pip install --no-deps \
  "salad @ git+https://github.com/Dominic101/salad.git@33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d" \
  "vggt @ git+https://github.com/MIT-SPARK/VGGT_SPARK.git@6e6e16107b88e8e76c751826af10d4295d87ecd2"

assert_torch_cuda "${ENV_PREFIX}" "12.1"
verify_method "${ENV_PREFIX}" vggt_slam

