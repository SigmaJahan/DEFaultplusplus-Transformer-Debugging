#!/usr/bin/env bash
# Record the DEFault++ committee demo as an MP4 suitable for slide embedding.
#
# Flow:
#   1. Start macOS ``screencapture -v`` in the background.
#   2. Run the demo script; its terminal output is what gets recorded.
#   3. Stop the recording, convert .mov -> .mp4 (H.264 + AAC) so
#      PowerPoint and Keynote can embed it without re-encoding.
#
# Outputs (next to the script):
#   demo_committee_recording.mov   raw HEVC from screencapture
#   demo_committee_recording.mp4   PowerPoint-friendly H.264
#
# Requirements (already verified on this machine):
#   - macOS screencapture has Screen Recording permission for the
#     terminal you launch this from.
#   - ffmpeg is on PATH (Homebrew: ``brew install ffmpeg``).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW_MOV="$SCRIPT_DIR/demo_committee_recording.mov"
OUT_MP4="$SCRIPT_DIR/demo_committee_recording.mp4"

# Activate the project venv so `python` finds defaultplusplus 0.4.1.
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Demo runs in ~9s; we pad to 20s to cover the leading countdown and
# the final "Done" banner.
DURATION_SECONDS=20

# Remove any prior outputs so the file modes are clean.
rm -f "$RAW_MOV" "$OUT_MP4"

echo "Recording starts in 3 seconds. Bring this terminal to the foreground."
sleep 1; echo "  3..."
sleep 1; echo "  2..."
sleep 1; echo "  1..."

# -v       record video
# -V N     stop after N seconds
# -D 1     main display
# -C       capture the cursor too (helpful when scrolling)
# -x       no sound effect (otherwise macOS plays the screenshot click)
screencapture -v -V "$DURATION_SECONDS" -D 1 -C -x "$RAW_MOV" &
RECORD_PID=$!

# Tiny lead-in so the recording is rolling before the first line prints.
sleep 1

python "$SCRIPT_DIR/demo_committee.py"

# Let the recorder finish its window (it stops on its own at V seconds).
wait "$RECORD_PID" 2>/dev/null || true

if [[ ! -f "$RAW_MOV" ]]; then
    echo "ERROR: recording did not produce $RAW_MOV." >&2
    echo "Check System Settings -> Privacy & Security -> Screen & System Audio" >&2
    echo "Recording, and ensure this terminal app has permission." >&2
    exit 1
fi

echo
echo "Raw recording: $RAW_MOV ($(du -h "$RAW_MOV" | cut -f1))"
echo "Converting to MP4 (H.264) for slide embedding..."

# -crf 23 is visually lossless for a terminal screencast.
# -movflags +faststart lets PowerPoint/Keynote start playback before
# the whole file is read.
ffmpeg -y -i "$RAW_MOV" \
    -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p \
    -movflags +faststart \
    -an \
    "$OUT_MP4" 2>&1 | tail -5

echo
echo "Done."
echo "  MP4 for slides: $OUT_MP4 ($(du -h "$OUT_MP4" | cut -f1))"
echo "  Raw MOV       : $RAW_MOV ($(du -h "$RAW_MOV" | cut -f1))"
echo
echo "Embed the MP4 in PowerPoint via Insert -> Video -> This Device."
