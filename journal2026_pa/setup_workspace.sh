#!/usr/bin/env bash
# Lay out a workspace in which the journal analyses run unchanged (they address runs/, cadence_journal_1119/
# and grapemots-protocol/ relative to one root).  Usage, from the repository root:
#     bash journal2026_pa/setup_workspace.sh /path/to/workspace
# Needs bash, tar and xz; about 2 GB of free space for the extracted per-frame outputs.
set -euo pipefail
W="${1:?usage: setup_workspace.sh WORKSPACE}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
J="$REPO/journal2026_pa"
mkdir -p "$W/cadence_journal_1119"
cp -R "$J/tools" "$W/cadence_journal_1119/tools"
cp -R "$J/results" "$W/cadence_journal_1119/results"
ln -sfn "$REPO" "$W/grapemots-protocol"
for name in oracle_cadence_1119 wholeframe_strongsort_1119 fullframe_pponly_0922 error_sources_1120 external_1122_apple \
            vine_calibration_1122 pa_followup_1124; do
  if [ -f "$J/raw/$name.tar.xz" ]; then
    xz -dc "$J/raw/$name.tar.xz" | tar -x -C "$W"
  else
    cat "$J/raw/$name.tar.xz.part-"* | xz -dc | tar -x -C "$W"
  fi
done
echo "workspace ready: $W"
