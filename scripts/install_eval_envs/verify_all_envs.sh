#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100

[[ -f "${REPO_ROOT}/thirdparty/dinov2/hubconf.py" ]] || \
  die "missing thirdparty/dinov2/hubconf.py"
[[ -x "${REPO_ROOT}/build/benchmark/orb_slam3_monocular" ]] || \
  die "missing ORB-SLAM3 benchmark runner"

methods=(worldsearcher vggt_slam reviv egom2p droid_slam megasam hawor)
for method in "${methods[@]}"; do
  prefix="$(env_path "${method}")"
  [[ -x "${prefix}/bin/python" ]] || die "missing environment ${prefix}"
  verify_method "${prefix}" "${method}"
done

log "all benchmark environments passed import and H100 checks"
