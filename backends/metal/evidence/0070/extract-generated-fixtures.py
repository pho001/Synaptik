#!/usr/bin/env python3
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMPONENTS = (
    ("native/metal-macos-arm64/src/synaptik_exact_kernels.h", "SynaptikExactKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0059_data_kernels.h", "SynaptikTask0059DataKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0060_reduction_kernels.h", "SynaptikTask0060ReductionKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0061_matmul_kernels.h", "SynaptikTask0061MatmulKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0063_ordering_kernels.h", "SynaptikTask0063OrderingKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0064_convolution_pooling_kernels.h", "SynaptikTask0064ConvolutionPoolingKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0069_aggregate_kernels.h", "SynaptikTask0069AggregateKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0065_rng_dropout_kernels.h", "SynaptikTask0065RngDropoutKernelSource"),
    ("native/metal-macos-arm64/src/synaptik_task0066_dtype_layout_kernels.h", "SynaptikTask0066DtypeLayoutKernelSource"),
)
HELPERS = {"floor": "floor_bits", "ceil": "ceil_bits", "sign": "sign_bits", "relu": "relu_bits"}
HELPER_BYTES = {"floor": 10, "ceil": 9, "sign": 9, "relu": 9}
FIXTURES = (
    (0, "singleton", ("floor",)),
    (1, "singleton", ("ceil",)),
    (2, "singleton", ("sign",)),
    (3, "singleton", ("relu",)),
    (4, "chain-2", ("floor", "ceil")),
    (5, "chain-3", ("ceil", "sign", "relu")),
    (6, "chain-4", ("floor", "ceil", "sign", "relu")),
    (7, "chain-5", ("floor", "ceil", "sign", "relu", "floor")),
    (8, "chain-6", ("floor", "ceil", "sign", "relu", "floor", "ceil")),
    (9, "chain-7", ("floor", "ceil", "sign", "relu", "floor", "ceil", "sign")),
    (10, "chain-8", ("floor", "ceil", "sign", "relu", "floor", "ceil", "sign", "relu")),
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


def expected_function_bytes(step: int, opcodes: tuple[str, ...]) -> int:
    assert 1 <= len(opcodes) <= 8
    return 345 + len(str(step)) + len(str(len(opcodes))) + sum(
        16 + len(str(index)) + len(str(index + 1)) + HELPER_BYTES[opcode]
        for index, opcode in enumerate(opcodes)
    )


def generated_function(step: int, opcodes: tuple[str, ...]) -> str:
    lines = [
        f"kernel void synaptik_pw_g1_s{step}(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]], constant PointMeta &meta [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {{\n",
        "  ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight);\n",
        "  if (linear >= meta.elementCount) return;\n",
        "  uint v0 = input[linear];\n",
    ]
    lines.extend(
        f"  uint v{index + 1} = {HELPERS[opcode]}(v{index});\n"
        for index, opcode in enumerate(opcodes)
    )
    lines.extend((f"  output[linear] = v{len(opcodes)};\n", "}\n"))
    result = "".join(lines)
    assert len(result.encode("utf-8")) == expected_function_bytes(step, opcodes)
    return result


def main() -> None:
    assert len(sys.argv) == 3, "usage: extract-generated-fixtures.py SOURCE METADATA"
    component_bytes = [
        (symbol, extract_component(path, symbol).encode("utf-8"))
        for path, symbol in COMPONENTS
    ]
    fixed = b"".join(content for _, content in component_bytes).decode("utf-8")
    assert len(fixed.encode("utf-8")) == 77_411
    generated = "\n// synaptik pointwise fusion generator schema 1\n" + "".join(
        generated_function(step, opcodes) for step, _, opcodes in FIXTURES
    )
    source = fixed + generated
    Path(sys.argv[1]).write_text(source, encoding="utf-8", newline="\n")
    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "fixedBytes": len(fixed.encode("utf-8")),
                "generatedBytes": len(generated.encode("utf-8")),
                "totalBytes": len(source.encode("utf-8")),
                "fixedSha256": hashlib.sha256(fixed.encode("utf-8")).hexdigest(),
                "generatedSha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
                "assembledSha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "components": [
                    {
                        "symbol": symbol,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                    for symbol, content in component_bytes
                ],
                "fixtures": [
                    {
                        "step": step,
                        "siteKind": site_kind,
                        "opcodes": list(opcodes),
                        "functionBytes": expected_function_bytes(step, opcodes),
                    }
                    for step, site_kind, opcodes in FIXTURES
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
