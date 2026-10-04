#!/usr/bin/env bash
set -euo pipefail

python app.py --host 0.0.0.0 --port 5070 "$@"
