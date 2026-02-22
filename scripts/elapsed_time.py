#!/usr/bin/env python3
"""
Print current (actual) time and/or run a command and print elapsed time.

Usage:
  python scripts/elapsed_time.py
      → print current local and UTC time (for "actual time" at start/end).

  python scripts/elapsed_time.py -- <command> [args...]
      → run command, then print elapsed wall-clock time.
  Example: python scripts/elapsed_time.py -- python scripts/train_model.py --max-epochs 1
"""

import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter


def main():
    now = datetime.now()
    now_utc = datetime.now(timezone.utc)
    print(f"Local: {now.isoformat()}")
    print(f"UTC:   {now_utc.isoformat()}")

    if len(sys.argv) >= 2 and sys.argv[1] == "--":
        cmd = sys.argv[2:]
        if not cmd:
            print("No command after '--'. Exiting.")
            sys.exit(1)
        t0 = perf_counter()
        result = subprocess.run(cmd)
        elapsed = perf_counter() - t0
        m, s = divmod(int(round(elapsed)), 60)
        h, m = divmod(m, 60)
        if h:
            elapsed_str = f"{h}:{m:02d}:{s:02d}"
        else:
            elapsed_str = f"{m}:{s:02d}"
        print(f"Elapsed: {elapsed_str} ({elapsed:.1f}s)")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
