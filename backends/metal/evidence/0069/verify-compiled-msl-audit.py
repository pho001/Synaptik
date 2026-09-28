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
    ir_path = generated / "runtime-source.air.ll"
    symbols_path = generated / "runtime-source.air.nm"
    ir = ir_path.read_text(encoding="utf-8")
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

    symbols = symbols_path.read_text(encoding="utf-8")

    for name, path in (
        ("source", source),
        ("air", air),
        ("metallib", metallib),
        ("ir", ir_path),
        ("symbols", symbols_path),
    ):
        assert path.stat().st_size == artifacts[name]["bytes"], (
            name, path.stat().st_size
        )
        assert digest(path) == artifacts[name]["sha256"], (name, digest(path))
    assert symbols.count(" T l1_norm_f32_0069") == 1
    assert symbols.count(" T variance_f32_0069") == 1
    assert symbols.count(" T scatter_add_f32_0069") == 1

    def kernel_body(symbol: str) -> str:
        match = re.search(
            rf"define void @{symbol}\([^\n]+\) [^{{]+\{{(?P<body>.*?)\n\}}",
            ir,
            re.DOTALL,
        )
        assert match is not None
        return match.group("body")

    l1_body = kernel_body("l1_norm_f32_0069")
    l1_fadds = [line.strip() for line in l1_body.splitlines() if " fadd " in line]
    assert len(l1_fadds) == 1
    assert re.search(r"= fadd float %\d+, %\d+$", l1_fadds[0])
    assert all(flag not in l1_fadds[0] for flag in ("fast", "contract", "reassoc", "afn"))
    assert l1_body.count("2147483647") == 2
    assert l1_body.count(" bitcast i32 ") == 2
    assert l1_body.count(" bitcast float ") == 1
    assert "phi i64" in l1_body and "[ 1," in l1_body
    assert l1_body.count("store i8") == 1

    variance_body = kernel_body("variance_f32_0069")
    variance_fdivs = [
        line.strip() for line in variance_body.splitlines() if " fdiv " in line
    ]
    variance_fsubs = [
        line.strip() for line in variance_body.splitlines() if " fsub " in line
    ]
    variance_fmuls = [
        line.strip() for line in variance_body.splitlines() if " fmul " in line
    ]
    assert len(variance_fdivs) == 2
    assert len(variance_fsubs) == 1
    assert len(variance_fmuls) == 1
    variance_instructions = variance_fdivs + variance_fsubs + variance_fmuls
    assert all(
        re.search(r"= f(div|sub|mul) float ", instruction)
        for instruction in variance_instructions
    )
    assert all(
        all(flag not in instruction for flag in ("fast", "contract", "reassoc", "afn"))
        for instruction in variance_instructions
    )
    assert " fadd " not in variance_body
    assert "air.fma" not in variance_body and "llvm.fma" not in variance_body
    assert variance_body.count("store i8") == 1

    scatter_body = kernel_body("scatter_add_f32_0069")
    scatter_fadds = [
        line.strip() for line in scatter_body.splitlines() if " fadd " in line
    ]
    assert len(scatter_fadds) == 1
    assert re.search(r"= fadd float %\d+, %\d+$", scatter_fadds[0])
    assert all(
        flag not in scatter_fadds[0] for flag in ("fast", "contract", "reassoc", "afn")
    )
    assert scatter_body.count("store i8") == 1
    assert "phi i64" in scatter_body and "[ 0," in scatter_body
    assert "icmp eq i64" in scatter_body
    assert "add nuw i64" in scatter_body
    assert re.search(r"select i1 %\d+, i32 %\d+, i32 %\d+", scatter_body)
    assert "atomic" not in scatter_body

    for body in (l1_body, scatter_body):
        for forbidden in (" fmul ", " fsub ", " fdiv ", "air.fma", "llvm.fma"):
            assert forbidden not in body

    audit = manifest["compiledAirAudit"]
    target = audit["targetTriple"]
    assert f'target triple = "{target}"' in ir
    assert audit["l1Norm"]["kernelSymbol"] == "l1_norm_f32_0069"
    assert audit["l1Norm"]["floatingAddInstructions"] == 1
    assert audit["l1Norm"]["unsafeFloatingFlags"] == []
    assert audit["variance"]["kernelSymbol"] == "variance_f32_0069"
    assert audit["variance"]["floatingDivideInstructions"] == 2
    assert audit["variance"]["floatingSubtractInstructions"] == 1
    assert audit["variance"]["floatingMultiplyInstructions"] == 1
    assert audit["variance"]["floatingAddInstructions"] == 0
    assert audit["variance"]["unsafeFloatingFlags"] == []
    assert audit["scatterAdd"]["kernelSymbol"] == "scatter_add_f32_0069"
    assert audit["scatterAdd"]["floatingAddInstructions"] == 1
    assert audit["scatterAdd"]["unsafeFloatingFlags"] == []
    assert audit["scatterAdd"]["atomicInstructions"] == 0
    print("Task0069 Xcode compiled-MSL/AIR audit verified")


if __name__ == "__main__":
    main()
