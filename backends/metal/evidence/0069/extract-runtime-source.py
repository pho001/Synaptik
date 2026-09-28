#!/usr/bin/env python3
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMPONENTS = (
    ("native/metal-macos-arm64/src/synaptik_exact_kernels.h", "SynaptikExactKernelSource"),
    (
        "native/metal-macos-arm64/src/synaptik_task0059_data_kernels.h",
        "SynaptikTask0059DataKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0060_reduction_kernels.h",
        "SynaptikTask0060ReductionKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0061_matmul_kernels.h",
        "SynaptikTask0061MatmulKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0063_ordering_kernels.h",
        "SynaptikTask0063OrderingKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0064_convolution_pooling_kernels.h",
        "SynaptikTask0064ConvolutionPoolingKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0069_aggregate_kernels.h",
        "SynaptikTask0069AggregateKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0065_rng_dropout_kernels.h",
        "SynaptikTask0065RngDropoutKernelSource",
    ),
    (
        "native/metal-macos-arm64/src/synaptik_task0066_dtype_layout_kernels.h",
        "SynaptikTask0066DtypeLayoutKernelSource",
    ),
)


def extract_component(relative_path: str, symbol: str) -> str:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    marker = f"static NSString *const {symbol} ="
    assert text.count(marker) == 1
    block = text.split(marker, 1)[1].split(";\n", 1)[0]
    fragments = []
    for line in block.splitlines():
        literal = line.strip()
        if literal.startswith('@"'):
            literal = literal[1:]
        if literal.startswith('"'):
            fragments.append(ast.literal_eval(literal))
    assert fragments
    return "".join(fragments)


def runtime_source() -> str:
    foundation = (
        ROOT / "native/metal-macos-arm64/src/synaptik_metal_foundation.m"
    ).read_text(encoding="utf-8")
    cursor = -1
    for _, symbol in COMPONENTS:
        position = foundation.find(f"stringByAppendingString:{symbol}")
        if symbol == "SynaptikExactKernelSource":
            position = foundation.find(symbol)
        assert position > cursor
        cursor = position
    assert "SynaptikTask0053CandidateKernelSource" not in foundation
    assert "task0053_domain_approved" not in foundation
    return "".join(extract_component(path, symbol) for path, symbol in COMPONENTS)


def main() -> None:
    assert len(sys.argv) == 2, "usage: extract-runtime-source.py OUTPUT"
    Path(sys.argv[1]).write_text(runtime_source(), encoding="utf-8")


if __name__ == "__main__":
    main()
