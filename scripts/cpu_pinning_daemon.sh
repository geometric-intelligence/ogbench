#!/bin/bash
# CPU Pinning Daemon for ogbench-train processes
# Monitors for new processes and pins them to dedicated CPU cores
#
# Usage:
#   ./scripts/cpu_pinning_daemon.sh [CPUS_PER_JOB] [MAX_CPU] [INTERVAL]
#
# Examples:
#   ./scripts/cpu_pinning_daemon.sh           # Default: 2 CPUs per job, max CPU 127, check every 5s
#   ./scripts/cpu_pinning_daemon.sh 4         # 4 CPUs per job
#   ./scripts/cpu_pinning_daemon.sh 2 63      # 2 CPUs per job, only use cores 0-63
#   ./scripts/cpu_pinning_daemon.sh 2 127 10  # Check every 10 seconds
#
# To run in background:
#   nohup ./scripts/cpu_pinning_daemon.sh > /tmp/cpu_pinning.log 2>&1 &
#
# To stop:
#   pkill -f cpu_pinning_daemon.sh

CPUS_PER_JOB=${1:-2}
MAX_CPU=${2:-$(($(nproc) - 1))}  # Auto-detect CPU count
INTERVAL=${3:-5}

# File to track assigned PIDs and their cores
STATE_FILE="/tmp/cpu_pinning_state_$(whoami).txt"

# Calculate total number of slots
TOTAL_SLOTS=$(( (MAX_CPU + 1) / CPUS_PER_JOB ))

echo "=============================================="
echo "CPU Pinning Daemon Started"
echo "=============================================="
echo "CPUs per job: $CPUS_PER_JOB"
echo "CPU range: 0-$MAX_CPU"
echo "Total slots: $TOTAL_SLOTS"
echo "Check interval: ${INTERVAL}s"
echo "State file: $STATE_FILE"
echo "=============================================="

# Initialize state file if it doesn't exist
touch "$STATE_FILE"

# Function to get cores for a given slot
slot_to_cores() {
    local slot=$1
    local start=$((slot * CPUS_PER_JOB))
    local end=$((start + CPUS_PER_JOB - 1))
    echo "${start}-${end}"
}

# Function to find a free slot (not used by any running process)
find_free_slot() {
    local slot=0
    while [ $slot -lt $TOTAL_SLOTS ]; do
        local cores=$(slot_to_cores $slot)
        local in_use=false
        
        # Check if this slot's cores are assigned to any running process
        while IFS=: read -r pid assigned_cores; do
            if [ "$assigned_cores" = "$cores" ]; then
                # Check if this process is still running
                if kill -0 "$pid" 2>/dev/null; then
                    in_use=true
                    break
                fi
            fi
        done < "$STATE_FILE"
        
        if [ "$in_use" = false ]; then
            echo $slot
            return
        fi
        slot=$((slot + 1))
    done
    
    # All slots in use, wrap around to slot 0 (will share)
    echo "0"
}

# Function to pin a process and all its threads
pin_process() {
    local pid=$1
    local cores=$2
    
    # Pin main process
    sudo taskset -cp "$cores" "$pid" 2>/dev/null
    
    # Pin all threads
    if [ -d "/proc/$pid/task" ]; then
        for tid in $(ls /proc/$pid/task/ 2>/dev/null); do
            sudo taskset -cp "$cores" "$tid" 2>/dev/null
        done
    fi
}

# Main loop
while true; do
    # First, clean up finished processes from state file
    if [ -f "$STATE_FILE" ]; then
        > "${STATE_FILE}.tmp"
        while IFS=: read -r old_pid old_cores; do
            if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
                # Process still running, keep it
                echo "$old_pid:$old_cores" >> "${STATE_FILE}.tmp"
            else
                [ -n "$old_pid" ] && echo "[$(date '+%H:%M:%S')] Process PID=$old_pid finished, freeing cores $old_cores"
            fi
        done < "$STATE_FILE"
        mv "${STATE_FILE}.tmp" "$STATE_FILE" 2>/dev/null
    fi
    
    # Get all ogbench-train PIDs
    current_pids=$(pgrep -f "ogbench-train" 2>/dev/null | sort -n)
    
    if [ -n "$current_pids" ]; then
        for pid in $current_pids; do
            # Check if this PID is already tracked
            if ! grep -q "^$pid:" "$STATE_FILE" 2>/dev/null; then
                # New process found - find a free slot and assign cores
                free_slot=$(find_free_slot)
                cores=$(slot_to_cores $free_slot)
                
                echo "[$(date '+%H:%M:%S')] New process PID=$pid -> slot $free_slot (cores $cores)"
                
                # Pin the process
                pin_process "$pid" "$cores"
                
                # Record in state file
                echo "$pid:$cores" >> "$STATE_FILE"
            else
                # Existing process - re-pin threads (in case new threads spawned)
                cores=$(grep "^$pid:" "$STATE_FILE" | cut -d: -f2)
                if [ -n "$cores" ]; then
                    # Silently re-pin all threads
                    pin_process "$pid" "$cores" 2>/dev/null
                fi
            fi
        done
    fi
    
    sleep "$INTERVAL"
done
