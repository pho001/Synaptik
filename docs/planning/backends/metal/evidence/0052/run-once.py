#!/usr/bin/env python3
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
completed=subprocess.run([str(root/"oracle"),str(root/"oracle.metal")],capture_output=True,check=False)
(root/"raw-output.txt").write_bytes(completed.stdout)
(root/"raw-stderr.txt").write_bytes(completed.stderr)
(root/"process-exit.txt").write_text(str(completed.returncode)+"\n")
print(f"ORACLE_INVOCATIONS=1 EXIT={completed.returncode} STDOUT_BYTES={len(completed.stdout)} STDERR_BYTES={len(completed.stderr)}")
raise SystemExit(completed.returncode)
