#!/usr/bin/env bash
# Lab22 CLI(https://lab22.ai/docs/cli)로 메타데이터와 annotations만 받는다. 영상은 받지 않는다.
# 인증: LAB22_API_KEY. 환경 변수에 없으면 ~/.config/lab22/env 를 읽는다.
#
# Usage: scripts/download_lab22.sh <ego4d|egoexo> [output_dir]   (기본값 <repo>/data/raw/<dataset>)
# Env:   LAB22_ENV_FILE                                           (기본값 ~/.config/lab22/env)
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
  echo "LAB22_API_KEY가 비어 있습니다: $ENV_FILE 에 Lab22 API Keys 페이지에서 만든 키를 채우세요." >&2
  exit 1
fi
export LAB22_API_KEY

LAB22=(uvx --isolated --python 3.11 --from lab22 lab22)
if [ "$DATASET" = ego4d ]; then
  "${LAB22[@]}" ego4d --output_directory "$OUT" --datasets annotations --metadata --version v2
  echo
  echo "다음 단계:"
  echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --list-scenarios"
  echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --narrations $OUT/v2/annotations/narration.json"
else
  "${LAB22[@]}" egoexo -o "$OUT" --parts annotations metadata
fi
