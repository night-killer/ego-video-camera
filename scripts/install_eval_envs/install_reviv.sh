#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
require_toolchain "${CUDA124_ROOT}"
ensure_conda_env reviv 3.12
ENV_PREFIX="$(env_path reviv)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.6.0+cu124" "0.21.0+cu124" \
  "https://download.pytorch.org/whl/cu124"
"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/reviv-h100.txt"
build_torch_scatter "${ENV_PREFIX}" "${CUDA124_ROOT}"

assert_torch_cuda "${ENV_PREFIX}" "12.4"
verify_method "${ENV_PREFIX}" reviv

