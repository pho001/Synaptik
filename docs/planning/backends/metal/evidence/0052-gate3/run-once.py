#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: run-once.py <executable> <retained-oracle.metal> <raw-output>")

executable = pathlib.Path(sys.argv[1]).resolve(strict=True)
kernel = pathlib.Path(sys.argv[2]).resolve(strict=True)
raw_path = pathlib.Path(sys.argv[3])
if raw_path.exists():
    raise SystemExit(f"refusing second invocation: {raw_path} already exists")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())

result = subprocess.run([str(executable), str(kernel)], capture_output=True, check=False)
record = (
    f"EXECUTABLE_SHA256={sha256(executable)}\n"
    f"KERNEL_SHA256={sha256(kernel)}\n"
    f"PROCESS_EXIT={result.returncode}\n"
    f"STDERR_SHA256={sha256_bytes(result.stderr)}\n"
).encode() + result.stdout
if result.stderr:
    record += b"STDERR_BEGIN\n" + result.stderr + b"STDERR_END\n"
raw_path.open("xb").write(record)
if result.returncode != 0:
    raise SystemExit(f"Gate-3 process failed with exit {result.returncode}; raw evidence retained")
if result.stderr:
    raise SystemExit("Gate-3 process wrote stderr; raw evidence retained")
text = result.stdout.decode("utf-8")
required = (
    "TASK=0052_GATE3\n", "PROTOCOL=custom-only-v1\n", "DEVICE=Apple M3 Max\n",
    "CANDIDATE_COUNT=16\n", "WARMUP_ROUNDS=4\n", "RETAINED_ROUNDS=8\n",
    "FAST_MATH=0\n", "VALIDATION=PASS\n", "INPUTS_PRESERVED=PASS\n",
    "CLAMP_WINNER=", "GATE3_VERDICT=PASS\n",
)
missing = [item.rstrip() for item in required if item not in text]
if missing:
    raise SystemExit("Gate-3 output missing controls: " + ", ".join(missing))
if text.count("SAMPLE candidate=") != 128:
    raise SystemExit("Gate-3 output did not contain 16 x 8 retained samples")
if text.count("CANDIDATE=custom_") + text.count("CANDIDATE=composed_custom_") != 16:
    raise SystemExit("Gate-3 output did not contain 16 candidate summaries")
print(f"RAW_SHA256={sha256(raw_path)}")
print("WRAPPER=PASS")
