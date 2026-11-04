#!/usr/bin/env bash
# Ego4D v2 메타데이터(ego4d.json)와 annotations(narration.json 등)만 받는다. 영상은 받지 않는다.
# 자격 증명: ~/.aws/credentials 의 [ego4d] 프로필에 라이선스 메일의 access key / secret key 를 채워 둔다.
#
# Usage: scripts/download_ego4d.sh [output_dir]   (기본값 data/raw/ego4d)
# Env:   EGO4D_AWS_PROFILE                         (기본값 ego4d)
set -euo pipefail

OUT="${1:-data/raw/ego4d}"
PROFILE="${EGO4D_AWS_PROFILE:-ego4d}"
CRED="${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}"

# 프로필에 두 키가 모두 비어 있지 않은지 확인한다. 값은 출력하지 않는다.
filled=$(awk -v p="[$PROFILE]" '
  /^\[/ { in_p = ($0 == p); next }
  in_p && /^[[:space:]]*aws_(access_key_id|secret_access_key)[[:space:]]*=/ {
    v = $0; sub(/^[^=]*=[[:space:]]*/, "", v); if (v != "") n++
  }
  END { print n + 0 }' "$CRED" 2>/dev/null || echo 0)
if [ "$filled" -lt 2 ]; then
  echo "자격 증명이 비어 있습니다: $CRED 의 [$PROFILE] 프로필에 aws_access_key_id / aws_secret_access_key 를 채우세요." >&2
  exit 1
fi

# 다운로드 용량을 확인하는 프롬프트가 뜬다(-y 를 일부러 주지 않음).
uvx --isolated --python 3.11 --from ego4d ego4d \
  --output_directory "$OUT" \
  --datasets annotations \
  --metadata \
  --version v2 \
  --aws_profile_name "$PROFILE"

echo
echo "다음 단계:"
echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --list-scenarios"
echo "  npm run ingest -- ego4d --metadata $OUT/v2/ego4d.json --narrations $OUT/v2/annotations/narration.json"
