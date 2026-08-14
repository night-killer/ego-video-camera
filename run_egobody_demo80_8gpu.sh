#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${DEMO80_PYTHON:-/data/aigc/cyb/zxgu/env/worldsearcher/bin/python}
RUNNER=${DEMO80_RUNNER:-$ROOT/scripts/run_egobody_demo80.py}
CONFIG=${DEMO80_CONFIG:-$ROOT/configs/egobody_demo80_actimind.yaml}
MANIFEST=${DEMO80_MANIFEST:-$ROOT/configs/egobody_demo_80/egobody_80_manifest.json}
OUTPUT_ROOT=${DEMO80_OUTPUT_ROOT:-$ROOT/outputs/egobody_demo80_actimind}
GPU_CSV=${DEMO80_GPUS:-0,1,2,3,4,5,6,7}
DATA_ROOT=${DEMO80_DATA_ROOT:-}
METADATA_ROOT=${DEMO80_METADATA_ROOT:-}

log() {
  printf '[demo80-8gpu] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

resolve_repo_path() {
  local value=$1
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$ROOT" "$value"
  fi
}

CONFIG=$(resolve_repo_path "$CONFIG")
MANIFEST=$(resolve_repo_path "$MANIFEST")
OUTPUT_ROOT=$(resolve_repo_path "$OUTPUT_ROOT")
RUNNER=$(resolve_repo_path "$RUNNER")

[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"
[[ -f "$CONFIG" ]] || die "config is missing: $CONFIG"
[[ -f "$MANIFEST" ]] || die "manifest is missing: $MANIFEST"
command -v flock >/dev/null 2>&1 || die "flock is required"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ ${#GPUS[@]} -eq 8 ]] || die "exactly 8 GPU ids are required; got ${#GPUS[@]}"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || die "duplicate GPU id: $gpu"
  SEEN_GPUS[$gpu]=1
done

mkdir -p "$OUTPUT_ROOT/launcher_logs"
exec 9>"$OUTPUT_ROOT/.demo80_8gpu.lock"
flock -n 9 || die "another demo80 8-GPU launcher is using $OUTPUT_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

COMMON_ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT")
if [[ -n "$DATA_ROOT" ]]; then
  COMMON_ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ -n "$METADATA_ROOT" ]]; then
  COMMON_ARGS+=(--metadata-root "$METADATA_ROOT")
fi

log "building deterministic 8-GPU plan"
"$PYTHON" - "$MANIFEST" "$OUTPUT_ROOT" "$GPU_CSV" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
gpu_ids = [int(value) for value in sys.argv[3].split(",")]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
available = {
    str(clip["clip_id"])
    for category in ("desktop_head_motion", "walking_person")
    for clip in manifest["categories"][category]["clips"]
}
expected = {
    f"{prefix}_{index:03d}"
    for prefix in ("DESK", "WALK")
    for index in range(1, 41)
}
if available != expected:
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    raise RuntimeError(f"manifest is not the frozen demo80 selection: missing={missing}, extra={extra}")

shards = []
for shard_index, gpu in enumerate(gpu_ids):
    numbers = [shard_index + 1 + 8 * offset for offset in range(5)]
    clips = [
        clip_id
        for number in numbers
        for clip_id in (f"DESK_{number:03d}", f"WALK_{number:03d}")
    ]
    shards.append(
        {
            "shard_index": shard_index,
            "gpu": gpu,
            "clip_count": len(clips),
            "expected_sampled_frame_count": 1040,
            "clips": clips,
        }
    )

plan = {
    "schema_version": "egobody_demo80_8gpu_plan_v1",
    "manifest": str(manifest_path),
    "output_root": str(output_root),
    "gpu_count": len(gpu_ids),
    "clip_count": sum(item["clip_count"] for item in shards),
    "shards": shards,
}
output_root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=output_root, prefix=".multi_gpu_plan.",
    suffix=".tmp", delete=False
) as handle:
    temporary = Path(handle.name)
    json.dump(plan, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(output_root / "multi_gpu_plan.json")
for shard in shards:
    print(
        f"GPU {shard['gpu']}: {shard['clip_count']} clips, "
        f"{shard['expected_sampled_frame_count']} sampled frames"
    )
PY

if [[ ${DEMO80_PLAN_ONLY:-0} == 1 ]]; then
  log "plan-only complete: $OUTPUT_ROOT/multi_gpu_plan.json"
  exit 0
fi

rm -f -- "$OUTPUT_ROOT/run_summary.json"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available"
declare -A AVAILABLE_GPUS=()
while IFS= read -r gpu; do
  AVAILABLE_GPUS[$gpu]=1
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
for gpu in "${GPUS[@]}"; do
  [[ -n "${AVAILABLE_GPUS[$gpu]:-}" ]] || die "GPU $gpu is not available"
done

if [[ ${DEMO80_SKIP_VALIDATE:-0} != 1 ]]; then
  log "validating all 80 ego/exo inputs before launching GPUs"
  "$PYTHON" "$RUNNER" validate "${COMMON_ARGS[@]}" --all --require-exo \
    > "$OUTPUT_ROOT/launcher_logs/input_validation.json"
fi

PIDS=()
EXIT_CODES=()

cleanup() {
  local status=$?
  trap - EXIT INT HUP TERM
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'exit 143' TERM

for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  selection_args=()
  clips=()
  for offset in 0 1 2 3 4; do
    number=$((shard_index + 1 + 8 * offset))
    desk=$(printf 'DESK_%03d' "$number")
    walk=$(printf 'WALK_%03d' "$number")
    clips+=("$desk" "$walk")
    selection_args+=(--clip-id "$desk" --clip-id "$walk")
  done
  worker_log="$OUTPUT_ROOT/launcher_logs/gpu${gpu}.log"
  worker_summary="$OUTPUT_ROOT/launcher_logs/gpu${gpu}.run_summary.json"
  rm -f -- "$worker_summary"
  {
    printf '[demo80-8gpu] GPU %s starting clips:' "$gpu"
    printf ' %s' "${clips[@]}"
    printf '\n'
  } > "$worker_log"
  log "starting GPU $gpu with 10 clips; log: $worker_log"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$RUNNER" run \
    "${COMMON_ARGS[@]}" \
    --summary-path "$worker_summary" \
    --run-da3 --render --evaluate --resume --continue-on-error \
    "${selection_args[@]}" \
    >> "$worker_log" 2>&1 &
  PIDS[$shard_index]=$!
done

WORKER_FAILURES=0
for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  if wait "${PIDS[$shard_index]}"; then
    EXIT_CODES[$shard_index]=0
    log "GPU $gpu completed"
  else
    code=$?
    EXIT_CODES[$shard_index]=$code
    WORKER_FAILURES=$((WORKER_FAILURES + 1))
    log "GPU $gpu failed with exit code $code; inspect launcher_logs/gpu${gpu}.log"
  fi
done
PIDS=()

exit_code_csv=$(IFS=,; printf '%s' "${EXIT_CODES[*]}")
AGGREGATE_STATUS=0
"$PYTHON" - "$OUTPUT_ROOT" "$exit_code_csv" <<'PY' || AGGREGATE_STATUS=$?
import json
import os
import sys
import tempfile
from pathlib import Path

output_root = Path(sys.argv[1]).resolve()
exit_codes = [int(value) for value in sys.argv[2].split(",")]
plan = json.loads((output_root / "multi_gpu_plan.json").read_text(encoding="utf-8"))
all_results = []
worker_reports = []
seen = set()

for shard, exit_code in zip(plan["shards"], exit_codes):
    gpu = shard["gpu"]
    expected = set(shard["clips"])
    summary_path = output_root / "launcher_logs" / f"gpu{gpu}.run_summary.json"
    results = []
    load_error = None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = payload.get("clips", [])
    except (OSError, ValueError, TypeError) as error:
        load_error = f"{type(error).__name__}: {error}"
    observed = {str(item.get("clip_id")) for item in results}
    unexpected = sorted(observed - expected)
    duplicates = sorted(clip_id for clip_id in observed if clip_id in seen)
    duplicate_worker_results = len(observed) != len(results)
    if unexpected or duplicates or duplicate_worker_results:
        raise RuntimeError(
            f"invalid worker summary for GPU {gpu}: unexpected={unexpected}, "
            f"cross_worker_duplicates={duplicates}, "
            f"duplicate_worker_results={duplicate_worker_results}"
        )
    for item in results:
        enriched = dict(item)
        enriched["gpu"] = gpu
        enriched["shard_index"] = shard["shard_index"]
        all_results.append(enriched)
        seen.add(str(item["clip_id"]))
    missing = sorted(expected - observed)
    for clip_id in missing:
        all_results.append(
            {
                "clip_id": clip_id,
                "status": "failed",
                "gpu": gpu,
                "shard_index": shard["shard_index"],
                "error_type": "MissingWorkerResult",
                "error": load_error or f"worker exited {exit_code} without a clip result",
            }
        )
    worker_reports.append(
        {
            "gpu": gpu,
            "shard_index": shard["shard_index"],
            "exit_code": exit_code,
            "summary_path": str(summary_path),
            "reported_clip_count": len(results),
            "missing_clips": missing,
            "load_error": load_error,
        }
    )

all_results.sort(key=lambda item: item["clip_id"])
failed = sum(item.get("status") != "ok" for item in all_results)
worker_failures = sum(exit_code != 0 for exit_code in exit_codes)
overall_failed = bool(failed or worker_failures)
summary = {
    "schema_version": "egobody_demo80_8gpu_summary_v1",
    "status": "failed" if overall_failed else "ok",
    "clip_count": len(all_results),
    "succeeded": len(all_results) - failed,
    "failed": failed,
    "worker_failures": worker_failures,
    "workers": worker_reports,
    "clips": all_results,
}
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=output_root, prefix=".run_summary.",
    suffix=".tmp", delete=False
) as handle:
    temporary = Path(handle.name)
    json.dump(summary, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(output_root / "run_summary.json")
print(json.dumps({
    key: summary[key]
    for key in ("status", "clip_count", "succeeded", "failed", "worker_failures")
}))
raise SystemExit(1 if overall_failed else 0)
PY

if (( WORKER_FAILURES != 0 || AGGREGATE_STATUS != 0 )); then
  log "run incomplete: worker_failures=$WORKER_FAILURES aggregate_status=$AGGREGATE_STATUS"
  log "resume with the same command after addressing errors"
  exit 1
fi

log "all 80 clips completed: $OUTPUT_ROOT/run_summary.json"
