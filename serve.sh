#!/usr/bin/env bash
# 로컬 서버 실행
#
# 왜 필요한가:
#   Chrome은 getUserMedia(웹캠)에 secure context를 요구합니다. file:// 로 열면
#   카메라가 절대 열리지 않습니다. localhost 는 secure context로 인정됩니다.
#   ES 모듈 import 도 file:// 에서는 CORS로 차단됩니다.
#
# 사용:  ./serve.sh          (기본 8000번 포트)
#        ./serve.sh 3000

set -euo pipefail

PORT="${1:-8000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/web"

if [ ! -d "$DIR" ]; then
  echo "web/ 폴더를 찾을 수 없습니다: $DIR" >&2
  exit 1
fi

echo "────────────────────────────────────────────────────────"
echo "  http://localhost:${PORT}"
echo ""
echo "  · 웹캠은 localhost 에서만 동작합니다 (file:// 불가)"
echo "  · 종료: Ctrl+C"
echo "────────────────────────────────────────────────────────"
echo ""

# 브라우저 자동 실행 (macOS / Linux)
if command -v open >/dev/null 2>&1; then
  (sleep 1 && open "http://localhost:${PORT}") &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 1 && xdg-open "http://localhost:${PORT}") &
fi

cd "$DIR"
exec python3 -m http.server "$PORT"
