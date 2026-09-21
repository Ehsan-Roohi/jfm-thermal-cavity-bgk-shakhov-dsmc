#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m py_compile solver/*.py tools/*.py tests/*.py
python tests/test_bundle.py
python tests/test_restart_contract.py
python tests/test_restart_roundtrip.py
python tests/test_summary_e2e.py
python tools/audit_publication.py
echo "[OK] restartable checkpoint-fast local preflight passed"
