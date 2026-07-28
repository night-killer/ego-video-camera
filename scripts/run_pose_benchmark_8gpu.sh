#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-/data/aigc/cyb/zxgu/env/worldsearcher/bin/python}"
CONFIG="${BENCHMARK_CONFIG:-${REPO_ROOT}/configs/ego_pose_benchmark.yaml}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/ego_pose_benchmark}"
EVAL_ENV_ROOT="${EVAL_ENV_ROOT:-/data/aigc/cyb/zxgu/env}"
GPU_CSV="${BENCHMARK_GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${REPO_ROOT}/scripts/run_pose_benchmark.py"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

log() {
  printf '[benchmark-8gpu] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

[[ -x "${PYTHON}" ]] || die "benchmark Python is not executable: ${PYTHON}"
[[ -f "${CONFIG}" ]] || die "benchmark config is missing: ${CONFIG}"
[[ -f "${RUNNER}" ]] || die "benchmark runner is missing: ${RUNNER}"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
[[ ${#GPUS[@]} -gt 0 ]] || die "BENCHMARK_GPUS is empty"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU id: ${gpu}"
  SEEN_GPUS[${gpu}]=1
done

if [[ "${BENCHMARK_PLAN_ONLY:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available"
  declare -A AVAILABLE_GPUS=()
  while IFS= read -r gpu; do
    AVAILABLE_GPUS[${gpu}]=1
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  for gpu in "${GPUS[@]}"; do
    [[ -n "${AVAILABLE_GPUS[${gpu}]:-}" ]] || die "GPU ${gpu} is not available"
  done
fi

export CONDA_ENVS_PATH="${EVAL_ENV_ROOT}"
unset PYTHONHOME
mkdir -p "${OUTPUT_ROOT}/launcher_logs"

SHARD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ego-pose-benchmark-shards.XXXXXX")"
PIDS=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -d "${SHARD_DIR}" ]]; then
    rm -r -- "${SHARD_DIR}"
  fi
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${BENCHMARK_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  log "checking datasets, checkpoints, repositories, tools, and Conda environments"
  if ! "${PYTHON}" "${RUNNER}" \
    --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" preflight \
    > "${OUTPUT_ROOT}/preflight.console.json"; then
    die "preflight failed; inspect ${OUTPUT_ROOT}/preflight.console.json"
  fi
fi

log "building the complete deterministic run plan"
"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" plan \
  > "${OUTPUT_ROOT}/plan.console.json"

"${PYTHON}" - \
  "${REPO_ROOT}" "${CONFIG}" "${OUTPUT_ROOT}" "${SHARD_DIR}" "${GPU_CSV}" <<'PY'
import collections
import copy
import json
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
shard_dir = Path(sys.argv[4]).resolve()
gpu_ids = [int(value) for value in sys.argv[5].split(",")]

inventory = json.loads((output_root / "inventory.json").read_text(encoding="utf-8"))
plan = json.loads((output_root / "plan.json").read_text(encoding="utf-8"))
durations = {
    f"{item['dataset_id']}/{item['sequence_id']}": float(item["duration_sec"])
    for item in inventory["sequences"]
}
run_counts = collections.Counter(
    f"{item['dataset_id']}/{item['sequence_id']}" for item in plan["runs"]
)
missing = sorted(set(durations) - set(run_counts))
if missing:
    raise RuntimeError(f"sequences without planned runs: {', '.join(missing)}")
if len(gpu_ids) > len(run_counts):
    raise RuntimeError(
        f"cannot split {len(run_counts)} sequences across {len(gpu_ids)} GPUs"
    )

shards = [[] for _ in gpu_ids]
loads = [0.0 for _ in gpu_ids]
weighted_sequences = sorted(
    run_counts,
    key=lambda key: (durations[key] * run_counts[key], key),
    reverse=True,
)
for sequence_key in weighted_sequences:
    shard_index = min(range(len(gpu_ids)), key=lambda index: loads[index])
    weight = durations[sequence_key] * run_counts[sequence_key]
    shards[shard_index].append(sequence_key)
    loads[shard_index] += weight

base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
base_config["repo_root"] = str(repo_root)
base_config["benchmark"]["output_root"] = str(output_root)
shard_plan = {
    "schema_version": 1,
    "planned_run_count": int(plan["run_count"]),
    "sequence_count": len(run_counts),
    "weight": "duration_sec * planned_run_count",
    "shards": [],
}
for shard_index, gpu_id in enumerate(gpu_ids):
    config = copy.deepcopy(base_config)
    config["benchmark"]["gpu"] = gpu_id
    (shard_dir / f"config-{shard_index}.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    sequences = sorted(shards[shard_index])
    (shard_dir / f"sequences-{shard_index}.txt").write_text(
        "\n".join(sequences) + "\n", encoding="utf-8"
    )
    shard_plan["shards"].append(
        {
            "shard_index": shard_index,
            "gpu": gpu_id,
            "sequence_count": len(sequences),
            "weight": loads[shard_index],
            "sequences": sequences,
        }
    )
    print(
        f"GPU {gpu_id}: {len(sequences)} sequences, "
        f"weighted seconds={loads[shard_index]:.0f}"
    )

(output_root / "multi_gpu_plan.json").write_text(
    json.dumps(shard_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY

if [[ "${BENCHMARK_PLAN_ONLY:-0}" == "1" ]]; then
  log "plan-only mode complete: ${OUTPUT_ROOT}/multi_gpu_plan.json"
  exit 0
fi

for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[${shard_index}]}"
  mapfile -t sequences < "${SHARD_DIR}/sequences-${shard_index}.txt"
  [[ ${#sequences[@]} -gt 0 ]] || die "GPU ${gpu} received an empty shard"
  selection_args=()
  for sequence in "${sequences[@]}"; do
    selection_args+=(--sequences "${sequence}")
  done
  log "starting GPU ${gpu} shard with ${#sequences[@]} sequences"
  "${PYTHON}" "${RUNNER}" \
    --config "${SHARD_DIR}/config-${shard_index}.yaml" \
    run --resume "${selection_args[@]}" \
    > "${OUTPUT_ROOT}/launcher_logs/gpu${gpu}.log" 2>&1 &
  PIDS[${shard_index}]=$!
done

SHARD_FAILURES=0
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[${shard_index}]}"
  if wait "${PIDS[${shard_index}]}"; then
    log "GPU ${gpu} shard completed"
  else
    log "GPU ${gpu} shard failed; reconciliation will retry incomplete runs"
    SHARD_FAILURES=$((SHARD_FAILURES + 1))
  fi
done
PIDS=()

log "reconciling the full matrix on GPU ${GPUS[0]} (${SHARD_FAILURES} failed shards)"
"${PYTHON}" "${RUNNER}" \
  --config "${SHARD_DIR}/config-0.yaml" run --resume \
  > "${OUTPUT_ROOT}/reconcile.console.json"

log "evaluating completed predictions"
"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" evaluate --resume \
  > "${OUTPUT_ROOT}/evaluate.console.json"

log "generating reports"
"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" report

log "benchmark complete: ${OUTPUT_ROOT}/report"
