#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[3]
MANIFEST = EVIDENCE / "compiled-msl-audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert len(sys.argv) == 2, "usage: verify-compiled-msl-audit.py COMPILE_DIRECTORY"
    generated = Path(sys.argv[1])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    source = generated / "runtime-source.metal"
    air = generated / "runtime-source.air"
    metallib = generated / "runtime-source.metallib"
    ir = (generated / "runtime-source.air.ll").read_text(encoding="utf-8")
    toolchain = manifest["toolchain"]
    assert (generated / "xcode-version.txt").read_text(encoding="utf-8").splitlines() == [
        f'Xcode {toolchain["xcodeVersion"]}',
        f'Build version {toolchain["xcodeBuild"]}',
    ]
    assert (generated / "metal-version.txt").read_text(encoding="utf-8").strip().splitlines()[0] == (
        toolchain["compilerVersion"]
    )
    assert (generated / "metallib-version.txt").read_text(encoding="utf-8").strip() == (
        toolchain["metallibVersion"]
    )
    assert (generated / "sdk-version.txt").read_text(encoding="utf-8").strip() == (
        toolchain["sdkVersion"]
    )
    assert (generated / "sdk-build.txt").read_text(encoding="utf-8").strip() == (
        toolchain["sdkBuild"]
    )
    assert (generated / "sdk-path.txt").read_text(encoding="utf-8").strip() == (
        toolchain["sdkPath"]
    )

    symbols = (generated / "runtime-source.air.nm").read_text(encoding="utf-8")

    assert digest(source) == artifacts["source"]["sha256"]
    assert digest(air) == artifacts["air"]["sha256"]
    assert digest(metallib) == artifacts["metallib"]["sha256"]
    assert symbols.count(" T l1_norm_f32_0069") == 1

    function_match = re.search(
        r"define void @l1_norm_f32_0069\([^\n]+\) [^{]+\{(?P<body>.*?)\n\}",
        ir,
        re.DOTALL,
    )
    assert function_match is not None
    body = function_match.group("body")
    fadds = [line.strip() for line in body.splitlines() if " fadd " in line]
    assert len(fadds) == 1
    assert re.search(r"= fadd float %\d+, %\d+$", fadds[0])
    assert all(flag not in fadds[0] for flag in ("fast", "contract", "reassoc", "afn"))
    assert body.count("2147483647") == 2
    assert body.count(" bitcast i32 ") == 2
    assert body.count(" bitcast float ") == 1
    assert "phi i64" in body and "[ 1," in body
    assert body.count("store i8") == 1
    for forbidden in (" fmul ", " fsub ", " fdiv ", "air.fma", "llvm.fma"):
        assert forbidden not in body

    target = manifest["compiledAirAudit"]["targetTriple"]
    assert f'target triple = "{target}"' in ir
    assert manifest["compiledAirAudit"]["kernelSymbol"] == "l1_norm_f32_0069"
    assert manifest["compiledAirAudit"]["floatingAddInstructions"] == 1
    assert manifest["compiledAirAudit"]["unsafeFloatingFlags"] == []
    print("Task0069 Xcode compiled-MSL/AIR audit verified")


if __name__ == "__main__":
    main()
