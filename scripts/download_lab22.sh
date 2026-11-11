#!/usr/bin/env bash
# Downloads metadata and annotations only, through the Lab22 CLI (https://lab22.ai/docs/cli). No video.
# Auth: LAB22_API_KEY. Read from the repo's .env when it is not in the environment.
#
# Usage: scripts/download_lab22.sh <ego4d|egoexo> [output_dir]   (default <repo>/data/raw/<dataset>)
# Env:   LAB22_ENV_FILE                                           (default <repo>/.env)
set -euo pipefail

DATASET="${1:-}"
case "$DATASET" in
  ego4d | egoexo) ;;
  *)
    echo "Usage: scripts/download_lab22.sh <ego4d|egoexo> [output_dir]" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${2:-$ROOT/data/raw/$DATASET}"
ENV_FILE="${LAB22_ENV_FILE:-$ROOT/.env}"

# Reads only the LAB22_API_KEY line, so the rest of the file is never run as shell code.
if [ -z "${LAB22_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then
  LAB22_API_KEY="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}LAB22_API_KEY=//p' "$ENV_FILE" | tail -n 1)"
  LAB22_API_KEY="${LAB22_API_KEY%$'\r'}"
fi
if [ -z "${LAB22_API_KEY:-}" ]; then
  echo "LAB22_API_KEY is empty: add a key from the Lab22 API Keys page to $ENV_FILE." >&2
  exit 1
fi
export LAB22_API_KEY

LAB22=(uvx --isolated --python 3.11 --from lab22 lab22)
if [ "$DATASET" = ego4d ]; then
  # No --version: the platform serves its current release only, and the CLI picks it.
  "${LAB22[@]}" ego4d --output_directory "$OUT" --datasets annotations --metadata
  # The release folder name (v2_1, ...) follows the release, so look the files up.
  METADATA="$(find "$OUT" -name ego4d.json -type f 2>/dev/null | sort | tail -n 1)"
  NARRATIONS="$(find "$OUT" -path '*/annotations/narration.json' -type f 2>/dev/null | sort | tail -n 1)"
  if [ -z "$METADATA" ] || [ -z "$NARRATIONS" ]; then
    echo "Could not find ego4d.json or annotations/narration.json under $OUT." >&2
    exit 1
  fi
  echo
  echo "Next steps:"
  echo "  npm run ingest -- ego4d --metadata $METADATA --list-scenarios"
  echo "  python3 scripts/convert_ego4d_narrations.py --metadata $METADATA --narrations $NARRATIONS --out $OUT/narration_selected.json"
  echo "  npm run ingest -- ego4d --metadata $METADATA --narrations $OUT/narration_selected.json"
else
  "${LAB22[@]}" egoexo -o "$OUT" --parts annotations metadata
fi
