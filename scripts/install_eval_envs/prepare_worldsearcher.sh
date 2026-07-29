#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_repo
require_h100

WORLDSEARCHER="$(env_path worldsearcher)"
[[ -x "${WORLDSEARCHER}/bin/python" ]] || die "missing ${WORLDSEARCHER}"
[[ -x "${WORLDSEARCHER}/bin/nvcc" ]] || die "worldsearcher does not contain nvcc"

before_versions="$(${WORLDSEARCHER}/bin/python - <<'PY'
import numpy as np
import torch
print(f"{torch.__version__}|{np.__version__}")
PY
)"
log "worldsearcher core before install: ${before_versions}"

# ViPE imports rerun, pycg and gdown even when their optional paths are unused.
# Install only pinned packages and do not let pip resolve the shared environment.
"${WORLDSEARCHER}/bin/python" -m pip install --no-deps \
  "pyarrow==24.0.0" "python-pycg==1.0.3" "rerun-sdk==0.32.0" \
  "PySocks==1.7.1" "soupsieve==2.6" "beautifulsoup4==4.12.3" \
  "gdown==5.2.0"

activate_cuda_toolchain "${WORLDSEARCHER}" "${WORLDSEARCHER}"
log "building ViPE v1.2.0 for worldsearcher's Torch/CUDA and sm_90"
USE_SYSTEM_EIGEN=1 "${WORLDSEARCHER}/bin/python" -m pip install \
  --force-reinstall --no-deps --no-build-isolation \
  "${REPO_ROOT}/thirdparty/ViPE"

after_versions="$(${WORLDSEARCHER}/bin/python - <<'PY'
import numpy as np
import torch
print(f"{torch.__version__}|{np.__version__}")
PY
)"
[[ "${after_versions}" == "${before_versions}" ]] || \
  die "worldsearcher Torch/NumPy changed: ${before_versions} -> ${after_versions}"

verify_method "${WORLDSEARCHER}" worldsearcher
log "worldsearcher is ready for DA3, VGGT-Omega, LingBot-Map, ViPE and ORB worker IO"
