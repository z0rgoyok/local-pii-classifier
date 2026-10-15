#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
cd "$script_dir"

test -f data/development.jsonl

runtime_parent=${TMPDIR:-/tmp}
task_runtime=$(mktemp -d "$runtime_parent/local-pii.XXXXXX")
cleanup() {
  case "$task_runtime" in
    "$runtime_parent"/local-pii.??????)
      if [ -d "$task_runtime" ] && [ ! -L "$task_runtime" ]; then
        rm -r "$task_runtime"
      fi
      ;;
  esac
}
trap cleanup 0
trap 'exit 130' HUP INT TERM

export UV_CACHE_DIR="$task_runtime/uv-cache"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p build/results
task_results=$(mktemp -d "$script_dir/build/results/run.XXXXXX")

uv venv --python 3.12 "$task_runtime/venv"
uv pip install --python "$task_runtime/venv/bin/python" -r requirements.txt
uv pip freeze --python "$task_runtime/venv/bin/python" > "$task_results/requirements.freeze.txt"
task_python="$task_runtime/venv/bin/python"

"$task_python" -m unittest discover -s . -p 'test_*.py'
"$task_python" corpus.py --output "$task_results/challenge.jsonl"
"$task_python" models.py "$task_runtime/models" > "$task_results/model-manifest.json"
"$task_python" evaluate.py --models "$task_runtime/models" \
  --corpus data/development.jsonl --output "$task_results/development.json"
"$task_python" evaluate.py --models "$task_runtime/models" \
  --corpus "$task_results/challenge.jsonl" --output "$task_results/challenge.json" --warm-calls 600

printf 'Results: %s\n' "$task_results"
