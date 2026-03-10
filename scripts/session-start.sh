#!/bin/bash
# session-start.sh — Claude Code SessionStart hook
# Reads the most recent session report and injects a brief summary
# so Claude can flag pending suggestions without the user asking.

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
REPORTS_DIR="$PROJECT_DIR/.session-reports"

if [ ! -d "$REPORTS_DIR" ]; then
    exit 0
fi

# Find the most recent session report
LATEST_REPORT=$(ls -t "$REPORTS_DIR"/*.md 2>/dev/null | head -1)

if [ -z "$LATEST_REPORT" ]; then
    exit 0
fi

REPORT_NAME=$(basename "$LATEST_REPORT")

# Check if the report contains CLAUDE.md suggestions
if grep -qi "claude\.md suggest" "$LATEST_REPORT" 2>/dev/null; then
    HAS_SUGGESTIONS=true
else
    HAS_SUGGESTIONS=false
fi

# Check if there are open items
if grep -qi "open items\|unfinished\|todo\|blocked" "$LATEST_REPORT" 2>/dev/null; then
    HAS_OPEN_ITEMS=true
else
    HAS_OPEN_ITEMS=false
fi

# Build the context injection via stdout
# (SessionStart hooks can inject additionalContext)
echo "[Session Context] Last session report: $REPORT_NAME"

if [ "$HAS_SUGGESTIONS" = true ]; then
    echo "[Session Context] ⚡ Pending CLAUDE.md suggestions found in last report — review with user."
fi

if [ "$HAS_OPEN_ITEMS" = true ]; then
    echo "[Session Context] 📋 Open items from last session — check if user wants to continue."
fi
