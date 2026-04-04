#!/bin/bash
# Run pytest with parallel execution using pytest-xdist

set -e

# Number of workers (default: auto-detect CPU count)
WORKERS="${1:-auto}"

uv run pytest -n "$WORKERS" "${@:2}"
