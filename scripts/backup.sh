#!/usr/bin/env bash
# Back up results, checkpoints and the registry:  scripts/backup.sh [storage_dir] [dest_dir]
set -e
SRC=${1:-storage}; DEST=${2:-backups}; mkdir -p "$DEST"
tar czf "$DEST/fairtraffic-$(date +%Y%m%d-%H%M%S).tgz" -C "$(dirname "$SRC")" "$(basename "$SRC")"
echo "backup written to $DEST"
