#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[3]
SYMBOLS = (
    "matmul_f32_f32_f32_epilogue_0071",
    "matmul_f32_bf16_f32_epilogue_0071",
    "matmul_bf16_f32_f32_epilogue_0071",
    "matmul_f32_f32_f32_add_epilogue_0071",
    "matmul_f32_bf16_f32_add_epilogue_0071",
    "matmul_bf16_f32_f32_add_epilogue_0071",
    "conv2d_epilogue_0071",
    "conv2d_add_epilogue_0071",
)
UNSAFE_FLAGS = ("fast", "contract", "reassoc", "afn", "arcp", "nnan", "ninf")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_body(ir: str, symbol: str) -> str:
    match = re.search(
        rf"define [^\n]* @{re.escape(symbol)}\([^\n]+\) [^{{]+\{{(?P<body>.*?)\n\}}",
        ir,
        re.DOTALL,
    )
    assert match is not None, symbol
    return match.group("body")

def floating_instructions(body: str) -> list[str]:
    result = []
    for line in body.splitlines():
        stripped = line.strip()
        if re.search(r"\bf(add|mul|sub|div)\b", stripped):
            result.append(stripped)
        elif "@air.fma." in stripped or "@llvm.fma." in stripped:
            result.append(stripped)
    return result


def observation(work: Path) -> dict:
    source = work / "anchor-runtime.metal"
    air = work / "anchor-runtime.air"
    metallib = work / "anchor-runtime.metallib"
    ir_path = work / "anchor-runtime.air.ll"
    nm_path = work / "anchor-runtime.air.nm"
    ir = ir_path.read_text(encoding="utf-8")
    nm = nm_path.read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")
    foundation = (ROOT / "native/metal-macos-arm64/src/synaptik_metal_foundation.m").read_text(
        encoding="utf-8"
    )
    header = (ROOT / "native/metal-macos-arm64/src/synaptik_task0071_anchor_epilogue_kernels.h").read_text(
        encoding="utf-8"
    )

    assert "options.mathMode = MTLMathModeSafe" in foundation
    assert "options.languageVersion = MTLLanguageVersion3_2" in foundation
    assert "-ffast-math" not in header and "fast::" not in header
    for site in (
        "anchor_mul_site_0071(value, as_type<float>(m.scalarBits))",
        "anchor_add_site_0071(external, value)",
        "anchor_add_site_0071(value, external)",
        "anchor_terminal_0071(value, m)",
    ):
        assert site in header
    assert header.count("store_word(output, linear, 4u") == len(SYMBOLS)

    bodies = []
    for symbol in SYMBOLS:
        body = function_body(ir, symbol)
        floating = floating_instructions(body)
        if symbol != "conv2d_epilogue_0071":
            assert floating, symbol
        for instruction in floating:
            assert not any(re.search(rf"\b{flag}\b", instruction) for flag in UNSAFE_FLAGS), (
                symbol,
                instruction,
            )
        assert body.count("store i8") == 1, symbol
        bodies.append(
            {
                "symbol": symbol,
                "fadd": sum(" fadd " in line for line in floating),
                "fmul": sum(" fmul " in line for line in floating),
                "fsub": sum(" fsub " in line for line in floating),
                "fdiv": sum(" fdiv " in line for line in floating),
                "fma": sum("@air.fma." in line or "@llvm.fma." in line for line in floating),
                "floatingInstructionCount": len(floating),
                "unsafeFlags": [],
                "staticByteStores": 1,
            }
        )
    helper_match = re.search(
        r"define linkonce_odr float @(?P<symbol>_Z17conv2d_value_0071[^(]+)\(",
        ir,
    )
    assert helper_match is not None
    helper_symbol = helper_match.group("symbol")
    helper_body = function_body(ir, helper_symbol)
    helper_floating = floating_instructions(helper_body)
    assert helper_floating
    for instruction in helper_floating:
        assert not any(re.search(rf"\b{flag}\b", instruction) for flag in UNSAFE_FLAGS), instruction
    bodies.append(
        {
            "symbol": "conv2d_value_0071.helper",
            "fadd": sum(" fadd " in line for line in helper_floating),
            "fmul": sum(" fmul " in line for line in helper_floating),
            "fsub": sum(" fsub " in line for line in helper_floating),
            "fdiv": sum(" fdiv " in line for line in helper_floating),
            "fma": sum("@air.fma." in line or "@llvm.fma." in line for line in helper_floating),
            "floatingInstructionCount": len(helper_floating),
            "unsafeFlags": [],
            "staticByteStores": 0,
        }
    )

    triple = re.search(r'target triple = "([^"]+)"', ir)
    assert triple is not None
    return {
        "schema": "synaptik.metal.anchor-epilogue-air-audit.v1",
        "compilerFlags": [
            "-std=metal3.2",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-isysroot",
            "<MacOSX27.0.sdk>",
        ],
        "runtimeMathMode": "MTLMathModeSafe",
        "targetTriple": triple.group(1),
        "artifacts": {
            "source": {"bytes": source.stat().st_size, "sha256": digest(source)},
            "air": {"bytes": air.stat().st_size, "sha256": digest(air)},
            "metallib": {"bytes": metallib.stat().st_size, "sha256": digest(metallib)},
            "ir": {"bytes": ir_path.stat().st_size, "sha256": digest(ir_path)},
            "symbols": {"bytes": nm_path.stat().st_size, "sha256": digest(nm_path)},
            "kernelHeader": {
                "bytes": (ROOT / "native/metal-macos-arm64/src/synaptik_task0071_anchor_epilogue_kernels.h").stat().st_size,
                "sha256": digest(ROOT / "native/metal-macos-arm64/src/synaptik_task0071_anchor_epilogue_kernels.h"),
            },
        },
        "kernelBodies": bodies,
    }


def main() -> None:
    assert len(sys.argv) == 3, "usage: audit-anchor-air.py WORK OUTPUT"
    actual = observation(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(
        json.dumps(actual, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
