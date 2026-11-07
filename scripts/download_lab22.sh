#!/usr/bin/env bash
# Downloads metadata and annotations only, through the Lab22 CLI (https://lab22.ai/docs/cli). No video.
# Auth: LAB22_API_KEY. Read from ~/.config/lab22/env when it is not in the environment.
#
# Usage: scripts/download_lab22.sh <ego4d|egoexo> [output_dir]   (default <repo>/data/raw/<dataset>)
# Env:   LAB22_ENV_FILE                                           (default ~/.config/lab22/env)
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
ENV_FILE="${LAB22_ENV_FILE:-$HOME/.config/lab22/env}"

if [ -z "${LAB22_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi
if [ -z "${LAB22_API_KEY:-}" ]; then
  echo "LAB22_API_KEY is empty: add a key from the Lab22 API Keys page to $ENV_FILE." >&2
  exit 1
fi
export LAB22_API_KEY

LAB22=(uvx --isolated --python 3.11 --from lab22 lab22)
if [ "$DATASET" = ego4d ]; then
  "${LAB22[@]}" ego4d --output_directory "$OUT" --datasets annotations --metadata --version v2
  echo
  echo "Next steps:"
  echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --list-scenarios"
  echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --narrations $OUT/v2/annotations/narration.json"
else
  "${LAB22[@]}" egoexo -o "$OUT" --parts annotations metadata
fi
