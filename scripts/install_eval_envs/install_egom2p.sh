#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100
ensure_conda_env egom2p 3.12
ENV_PREFIX="$(env_path egom2p)"
install_packaging_tools "${ENV_PREFIX}"
install_torch_wheels "${ENV_PREFIX}" "2.6.0+cu124" "0.21.0+cu124" \
  "https://download.pytorch.org/whl/cu124"
"${ENV_PREFIX}/bin/python" -m pip install \
  -r "${INSTALL_DIR}/requirements/egom2p-h100.txt"

assert_torch_cuda "${ENV_PREFIX}" "12.4"
verify_method "${ENV_PREFIX}" egom2p

