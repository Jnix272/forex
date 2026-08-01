#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/update_session_report.sh
# Manually update docs/SESSION_REPORT.md with a custom message.
# Usage:  ./scripts/update_session_report.sh "Your note here"
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REPORT="$REPO_ROOT/docs/SESSION_REPORT.md"
mkdir -p "$REPO_ROOT/docs"

TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M UTC')
NOTE="${1:-Manual update}"

# Try to get git context (may not exist if repo not initialised)
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "no-commit")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
AUTHOR=$(git config user.name 2>/dev/null || echo "$(whoami)")

TMPFILE=$(mktemp)
cat > "$TMPFILE" << ENTRY
---

## Manual Update — $TIMESTAMP
**Author:** $AUTHOR  
**Branch:** $BRANCH ($COMMIT_HASH)  
**Note:** $NOTE

ENTRY

if [ -f "$REPORT" ]; then
    cat "$REPORT" >> "$TMPFILE"
fi
mv "$TMPFILE" "$REPORT"

echo "[update_session_report] Updated → $REPORT"
echo "Entry: $NOTE ($TIMESTAMP)"
