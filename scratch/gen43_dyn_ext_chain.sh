#!/bin/zsh
# Chain wrapper (2026-08-08, Kilian's in-conversation instruction: "once all this stuff is done
# and the mac is free can you chain the following command"). Waits until the aerial night work
# has fully drained, then launches the roads gen43 dynamic-arm extension batch VERBATIM as he
# gave it, and verifies at first-print level plus the zsh nice trap the batch's own header warns
# about. Launch authority for the batch itself: Kilian's, granted in-conversation.
set -u
AER=/Users/kilian/Kilian/ICL/Thesis/code/sacred-aerial
LOG=$AER/models/runs/gen39_step5/night_mac.log

echo "=== chain armed $(date '+%F %H:%M:%S'); waiting for the aerial night work to drain ==="

# 1. the Mac night driver signals its own completion
while ! grep -q "MAC NIGHT DRIVER COMPLETE" $LOG 2>/dev/null; do sleep 60; done
echo "=== night driver complete $(date '+%H:%M:%S') ==="

# 2. belt and braces: no aerial training / scoring / sweep process still holding cores
while pgrep -f "train_gen39_conceal|zeroshot2.py|gen44_budget_sweep" >/dev/null; do sleep 60; done
echo "=== aerial Mac work drained $(date '+%H:%M:%S'); launching the roads batch ==="

# 3. Kilian's command, verbatim
cd ~/Kilian/ICL/Thesis/code/sacred && nohup bash scratch/gen43_dyn_ext_batch.sh >> models/runs/gen43_unified/batch.log 2>&1 &
disown
sleep 90

# 4. verification: first prints, live PIDs, and the nice values (zsh can nice silently)
echo "=== batch launched; first-print check $(date '+%H:%M:%S') ==="
tail -12 ~/Kilian/ICL/Thesis/code/sacred/models/runs/gen43_unified/batch.log 2>/dev/null
PIDS=$(pgrep -f "train_.*71-33|gen43_dyn_ext|train_b1lite1|train_dyn_generalist" | tr '\n' ' ')
echo "training PIDs: ${PIDS:-none visible yet}"
[ -n "${PIDS// /}" ] && ps -o pid,nice,etime -p ${PIDS// /,}
echo "=== NOTE: if nice is non-zero, only Kilian can fix it, and on macOS renice -n is RELATIVE:"
echo "===   sudo renice -n -5 -p <pids>   (from nice 5 back to 0)"
echo "=== chain done $(date '+%H:%M:%S') ==="
