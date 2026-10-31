#!/bin/bash
# ARC supervised run: stages 10->12 with per-stage codegen override.
# Usage: tmux new-session -d -s arc './scripts/run_stage10_to_12.sh'
# tmux is required: plain nohup children died with the launching session.
set -u
cd "$(dirname "$0")/.." || exit 1

# Load secrets (API keys) from an untracked env file.
set -a
. "$HOME/.config/researchclaw/secrets.env"
set +a
export PYTHONUNBUFFERED=1

LOG=/tmp/arc-run.log
STATUS=/tmp/arc-run.status
echo "START $(date -u +%FT%TZ) supervisor_pid=$$" >"$STATUS"

.venv/bin/researchclaw run \
  --config config.yaml \
  --output artifacts/<RUN_DIR> \
  --from-stage CODE_GENERATION \
  --to-stage EXPERIMENT_RUN \
  --auto-approve \
  >>"$LOG" 2>&1 &
CHILD=$!
echo "CHILD_PID=$CHILD $(date -u +%FT%TZ)" >>"$STATUS"
wait "$CHILD"
RC=$?
echo "EXIT=$RC $(date -u +%FT%TZ)" >>"$STATUS"
echo "EXIT=$RC" >>"$LOG"
exit "$RC"
