#!/usr/bin/env python3
"""ESP32-S3 console reader for the Windows side (run via powershell from the
cargo-run runner in WSL). Reads the board's console on the CH343 COM port
directly, bypassing usbipd — usbipd's WSL forwarding dies when the laptop
joins the vo-box SoftAP (see AGENTS.md), the real COM port does not.

Finds the CH343 by VID/PID, pulses the chip reset (EN via RTS) so the board
emits a fresh boot log, then streams bytes to stdout with a heartbeat line
every ~5 s of silence so "logger alive, no data" is distinguishable from a
dead stream. Ctrl-C to stop.

Usage: python serial_monitor.py            # auto-find the CH343 COM port
"""
import datetime
import sys
import time

import serial
import serial.tools.list_ports

BAUD = 115200
# QinHeng CH340/CH341/CH342/CH343/CH9102 family VIDs/PIDs (the onboard
# USB-UART bridge on the Freenove ESP32-S3 board).
CH34X_PIDS = {0x7523, 0x5523, 0x55D3, 0x5584, 0x55D4}


def find_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x1A86 and p.pid in CH34X_PIDS:
            return p.device
    return None


def say(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


def main():
    port = find_port()
    if port is None:
        say("[serial_monitor] no CH34x COM port found. Board plugged in and "
            "usbipd detached? (run 'usbipd detach --busid 1-1' in an admin "
            "PowerShell)\n")
        return 1
    say(f"[serial_monitor] opening {port} at {BAUD}\n")
    try:
        s = serial.Serial(port, BAUD, timeout=0.2)
    except Exception as e:
        say(f"[serial_monitor] failed to open {port}: {e}\n")
        return 1

    # Pulse the chip reset (EN via RTS; IO0 held high = run mode) so we get a
    # fresh boot log as a baseline.
    s.dtr = False
    s.rts = True
    time.sleep(0.3)
    s.rts = False

    last_data = time.time()
    last_beat = 0.0
    while True:
        try:
            chunk = s.read(4096)
        except Exception as e:
            say(f"[serial_monitor] serial error: {e}\n")
            break
        now = time.time()
        if chunk:
            last_data = now
            say(chunk.decode(errors="replace"))
        elif now - last_data > 5.0 and now - last_beat > 5.0:
            last_beat = now
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            say(f"[{ts}] [alive, no data for {int(now - last_data)}s]\n")
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
