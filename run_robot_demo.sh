#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec /data/aigc/cyb/zxgu/env/worldsearcher/bin/python "$ROOT/scripts/run_robot_demo.py" "$@"
