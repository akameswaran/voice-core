#!/bin/bash
# session-cleanup.sh — Claude Code Stop hook
# Handles mechanical cleanup tasks that don't require AI reasoning.
# Designed to be fast and safe — runs after every Claude response.

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
ARCHIVE_DIR="$PROJECT_DIR/.project-archive"
REPORTS_DIR="$PROJECT_DIR/.session-reports"
ARCHIVE_QUEUE="$PROJECT_DIR/.archive-queue"
STALE_DAYS=14

# Ensure directories exist
mkdir -p "$ARCHIVE_DIR" "$REPORTS_DIR"

# --- Process archive queue ---
# During a session, Claude adds file paths to .archive-queue.
# This hook moves them to .project-archive/ preserving directory structure.
if [ -f "$ARCHIVE_QUEUE" ]; then
    while IFS= read -r filepath; do
        # Skip empty lines and comments
        [[ -z "$filepath" || "$filepath" == \#* ]] && continue
        
        full_path="$PROJECT_DIR/$filepath"
        if [ -e "$full_path" ]; then
            dest="$ARCHIVE_DIR/$filepath"
            mkdir -p "$(dirname "$dest")"
            mv "$full_path" "$dest"
            echo "[session-cleanup] Archived: $filepath"
        fi
    done < "$ARCHIVE_QUEUE"
    rm "$ARCHIVE_QUEUE"
fi

# --- Auto-archive old screenshots ---
# Debug screenshots older than STALE_DAYS are auto-archived.
SCREENSHOTS_DIR="$PROJECT_DIR/docs/screenshots"
if [ -d "$SCREENSHOTS_DIR" ]; then
    find "$SCREENSHOTS_DIR" -type f -mtime +$STALE_DAYS -print0 2>/dev/null | while IFS= read -r -d '' file; do
        rel_path="${file#$PROJECT_DIR/}"
        dest="$ARCHIVE_DIR/$rel_path"
        mkdir -p "$(dirname "$dest")"
        mv "$file" "$dest"
        echo "[session-cleanup] Auto-archived old screenshot: $rel_path"
    done
fi
