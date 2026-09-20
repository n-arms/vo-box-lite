#!/usr/bin/env bash
# cargo run runner for the ESP32-S3 (see .cargo/config.toml): cargo invokes
# this as  bash scripts/cargo_run.sh <path-to-elf>  after building.
#
# Flow: 1) usbipd-attach the board to WSL (if not already) so espflash can
#          reach it, 2) espflash flash the ELF (no --monitor), 3) usbipd
#          detach so the CH343 COM port is free on the Windows side, 4) stream
#          the console from the Windows COM port via powershell -> python
#          scripts/serial_monitor.py.
#
# Why COM5-on-Windows instead of a WSL-side monitor: usbipd's USB forwarding
# to WSL dies the moment the laptop joins the vo-box SoftAP (see AGENTS.md,
# "Console dies at station join = usbipd"), so a usbipd-attached console cuts
# out ~7 s after every boot when Windows auto-joins. The real COM port keeps
# streaming. If the COM port is already in use on Windows (e.g. a stale
# serial_monitor.py from an interrupted run), usbipd attach fails with
# "Device busy (exported)" — kill the python holding COM5 first.
#
# Requires: espflash on PATH (WSL), pyserial on the Windows python.
set -u

ELF="$1"
# The CH343's usbipd busid on the Windows host (check: usbipd list).
USBIPD_BUSID="${USBIPD_BUSID:-1-1}"
cd "$(dirname "$0")/.." || exit 1   # repo root

echo "== vo-box: flash + console (Windows COM port) =="

# --- 0) kill any stale serial_monitor.py from an interrupted run ----------
# A Ctrl-C'd monitor can leave the Windows python orphaned, holding COM5 so
# usbipd attach fails with "Device busy (exported)". Self-heal here.
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -like '*serial_monitor.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1 || true

# --- 0b) refuse to run while a previous espflash is wedged ------------------
# A D-state espflash parks in usb_kill_urb and pins the ttyACM node, so a new
# run just hangs at "Connecting..." (signals can't clear it: replug / `wsl
# --shutdown`). Abort clearly instead of hanging.
if ps -eo stat=,cmd= | awk '$1 ~ /^D/ && /espflash/ {f=1} END {exit !f}'; then
    echo "ERROR: a previous espflash is stuck in D state (usbipd/vhci wedge):" >&2
    ps -eo pid,stat,etimes,wchan:20,cmd | grep '[e]spflash' >&2 || true
    echo "  Recovery: unplug the ESP32-S3 -> 'wsl --shutdown' from Windows ->" >&2
    echo "  reopen WSL -> replug -> 'usbipd attach --wsl --busid $USBIPD_BUSID' -> re-run." >&2
    exit 1
fi
pkill -9 -x espflash 2>/dev/null || true   # interrupted, non-wedged leftovers

# --- 1) ensure the board is attached to WSL for flashing --------------------
if [ -z "$(ls /dev/ttyACM* 2>/dev/null | head -1)" ]; then
    echo "Attaching the board to WSL (usbipd attach --busid $USBIPD_BUSID)..."
    powershell.exe -NoProfile -Command "usbipd attach --wsl --busid $USBIPD_BUSID" || true
    sleep 2
fi
port="$(ls -t /dev/ttyACM* 2>/dev/null | head -1)"
if [ -z "$port" ]; then
    echo "ERROR: board not attached to WSL. Is it plugged in? Is COM5 free on" >&2
    echo "       Windows (no stale serial_monitor.py holding it)?" >&2
    exit 1
fi
# More than one node means a wedged session left a ghost; `ls -t` picks the
# newest (the live one), but say so loudly in case the ghost is all that's left.
if [ "$(ls /dev/ttyACM* 2>/dev/null | wc -l)" -gt 1 ]; then
    echo "WARNING: multiple ttyACM nodes ($(ls /dev/ttyACM* 2>/dev/null | tr '\n' ' ')); using newest $port" >&2
fi
echo "port: $port"

# --- 2) flash ---------------------------------------------------------------
espflash flash --port "$port" "$ELF" || { echo "ERROR: flash failed" >&2; exit 1; }

# --- 2a) custom partition table ---------------------------------------------
# `espflash flash <ELF>` has no partition table to go on, so it writes its
# built-in single-app table (no `model` row). It can't parse ours either:
# esp-idf-part 0.6.0 panics on the numeric `data` subtype 0x40 (both CSV and
# binary). The build copies the correct IDF-built binary table next to the ELF,
# and `write-bin` doesn't parse it, so overwrite what espflash just wrote.
PT_BIN="$(dirname "$ELF")/partition-table.bin"
if [ -f "$PT_BIN" ]; then
    echo "writing custom partition table -> 0x8000..."
    espflash write-bin --port "$port" 0x8000 "$PT_BIN" || { echo "ERROR: partition table write failed" >&2; exit 1; }
else
    echo "ERROR: $PT_BIN not found; the 'model' partition will be missing" >&2
fi

# --- 2b) flash the semantic model blob (int8 calc8) to its partition --------
# Offset must match the `model` row in partitions.csv. Absent -> the semantic
# task idles (logs an error).
MODEL_OFFSET="${MODEL_OFFSET:-0x400000}"
MODEL_TFLITE="${MODEL_TFLITE:-models/calc8.tflite}"
if [ -f "$MODEL_TFLITE" ]; then
    echo "writing $MODEL_TFLITE -> $MODEL_OFFSET (model partition)..."
    if ! espflash write-bin --port "$port" "$MODEL_OFFSET" "$MODEL_TFLITE"; then
        echo "WARNING: model write failed; the semantic task will idle" >&2
    fi
else
    echo "note: $MODEL_TFLITE not found — copy it from calc-quant/quant/calc_int8.tflite (semantic task will idle)"
fi

# --- 3) detach usbipd so the COM port is free on the Windows side ----------
echo "Detaching usbipd so COM5 is readable on Windows..."
powershell.exe -NoProfile -Command "usbipd detach --busid $USBIPD_BUSID" || true
sleep 2
if [ -n "$(ls /dev/ttyACM* 2>/dev/null | head -1)" ]; then
    echo "WARNING: usbipd still attached — the Windows COM port is not available." >&2
fi

# --- 4) console monitor on the Windows COM port (survives the vo-box join) --
# Note: WSL env vars do NOT reach Windows processes, so pass the UNC script
# path literally to python (wslpath -w renders \\wsl.localhost\Ubuntu\...).
WIN_SCRIPT="$(wslpath -w "$PWD/scripts/serial_monitor.py")"
echo "== console (Ctrl-C to exit) =="
exec powershell.exe -NoProfile -Command "python \"$WIN_SCRIPT\""
