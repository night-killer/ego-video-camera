#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB_ROOT="${REPO_ROOT}/thirdparty/ORB_SLAM3"
BUILD_DIR="${REPO_ROOT}/build/orb_slam3_benchmark"
OUTPUT_DIR="${REPO_ROOT}/build/benchmark"

if [[ ! -f "${ORB_ROOT}/lib/libORB_SLAM3.so" ]]; then
  echo "ORB-SLAM3 must be built first: ${ORB_ROOT}/lib/libORB_SLAM3.so" >&2
  exit 1
fi

cmake -S "${REPO_ROOT}/tools/orb_slam3_benchmark" -B "${BUILD_DIR}" \
  -DORB_SLAM3_ROOT="${ORB_ROOT}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel
mkdir -p "${OUTPUT_DIR}"
cp "${BUILD_DIR}/orb_slam3_monocular" "${OUTPUT_DIR}/orb_slam3_monocular"
echo "Built ${OUTPUT_DIR}/orb_slam3_monocular"
