#!/usr/bin/env bash
# SessionStart hook: kéo thông tin mới nhất của origin/main và BÁO CÁO thay đổi.
# Chỉ fetch + báo cáo, KHÔNG tự merge. Không bao giờ chặn phiên (luôn exit 0).

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

echo "=== Kiểm tra nhánh main (tự động khi mở phiên) ==="
echo "Nhánh hiện tại: $(git branch --show-current 2>/dev/null)"

if ! timeout 30 git fetch --quiet origin main 2>/dev/null; then
  echo "⚠️ Không kéo được origin/main (lỗi mạng?). Hãy chạy lại: git fetch origin main"
  exit 0
fi

NEW_COMMITS=$(git log --oneline HEAD..origin/main 2>/dev/null)
if [ -z "$NEW_COMMITS" ]; then
  echo "✅ Nhánh hiện tại đã có đủ mọi thay đổi của main."
  exit 0
fi

COUNT=$(printf '%s\n' "$NEW_COMMITS" | wc -l)
echo "🔔 main có $COUNT commit mới chưa gộp vào nhánh hiện tại:"
printf '%s\n' "$NEW_COMMITS" | head -20
[ "$COUNT" -gt 20 ] && echo "  ... (còn $((COUNT - 20)) commit nữa)"
echo
echo "File thay đổi trên main (A=thêm, M=sửa, D=xóa, R=đổi tên):"
git diff --name-status HEAD...origin/main 2>/dev/null | head -40
echo
echo "→ Theo CLAUDE.md: báo cho người dùng các thay đổi trên, rồi 'git merge origin/main' TRƯỚC khi sửa. Gặp xung đột thì dừng lại hỏi."
exit 0
