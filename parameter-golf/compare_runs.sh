#!/bin/bash
# Compare BPB across local test runs
# Usage: bash compare_runs.sh

echo "============================================================"
echo "  Local A/B Test Results"
echo "============================================================"
echo ""
printf "%-30s %10s %10s %10s %10s\n" "Run" "Roundtrip" "Sliding" "SLOT" "Artifact"
printf "%-30s %10s %10s %10s %10s\n" "---" "--------" "-------" "----" "--------"

for log in logs/local/*.log; do
    [ -f "$log" ] || continue
    NAME=$(basename "$log" .log)
    RT=$(grep "final_int6_roundtrip_exact" "$log" 2>/dev/null | grep -o "val_bpb:[0-9.]*" | grep -o "[0-9.]*" | tail -1)
    SW=$(grep "final_int6_sliding_window_exact" "$log" 2>/dev/null | grep -o "val_bpb:[0-9.]*" | grep -o "[0-9.]*" | tail -1)
    SL=$(grep "final_slot_exact" "$log" 2>/dev/null | grep -o "val_bpb:[0-9.]*" | grep -o "[0-9.]*" | tail -1)
    ART=$(grep "Total submission size" "$log" 2>/dev/null | grep -o "[0-9]* bytes" | grep -o "[0-9]*" | tail -1)
    [ -n "$ART" ] && ART=$(echo "scale=2; $ART / 1000000" | bc 2>/dev/null || echo "$ART")
    printf "%-30s %10s %10s %10s %8s MB\n" "$NAME" "${RT:-n/a}" "${SW:-n/a}" "${SL:-n/a}" "${ART:-n/a}"
done
echo ""
