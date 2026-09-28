#!/usr/bin/env python3
import ast
import hashlib
import json
import re
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
    ("native/metal-macos-arm64/src/synaptik_task0071_anchor_epilogue_kernels.h", "SynaptikTask0071AnchorEpilogueKernelSource"),
)
HELPERS = {
    b"floor_bits": "floor",
    b"ceil_bits": "ceil",
    b"sign_bits": "sign",
    b"relu_bits": "relu",
}
FIXED_BYTES = 84_541


def extract_component(relative_path: str, symbol: str) -> bytes:
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
    return "".join(fragments).encode("utf-8")


def main() -> None:
    assert len(sys.argv) == 3, "usage: extract-generated-fixtures.py SOURCE METADATA"
    source = Path(sys.argv[1]).read_bytes()
    fixed = source[:FIXED_BYTES]
    generated = source[FIXED_BYTES:]
    component_bytes = [
        (symbol, extract_component(path, symbol)) for path, symbol in COMPONENTS
    ]
    assert fixed == b"".join(content for _, content in component_bytes)

    symbols = list(re.finditer(rb"synaptik_pw_g1_s([0-9]+)\(", generated))
    assert [int(match.group(1)) for match in symbols] == list(range(11))
    fixtures = []
    function_start = 49
    for match in symbols:
        function_end = generated.find(b"\n}\n", match.end()) + 3
        assert function_end >= 3
        function = generated[function_start:function_end]
        step = int(match.group(1))
        opcodes = [HELPERS[name] for name in re.findall(
            rb"= (floor_bits|ceil_bits|sign_bits|relu_bits)\(", function
        )]
        expected_count = 1 if step < 4 else step - 2
        assert len(opcodes) == expected_count
        fixtures.append({
            "step": step,
            "siteKind": "singleton" if step < 4 else f"chain-{expected_count}",
            "opcodes": opcodes,
            "functionBytes": len(function),
        })
        function_start = function_end
    assert function_start == len(generated)

    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "fixedBytes": len(fixed),
                "generatedBytes": len(generated),
                "totalBytes": len(source),
                "fixedSha256": hashlib.sha256(fixed).hexdigest(),
                "generatedSha256": hashlib.sha256(generated).hexdigest(),
                "assembledSha256": hashlib.sha256(source).hexdigest(),
                "components": [
                    {
                        "symbol": symbol,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                    for symbol, content in component_bytes
                ],
                "fixtures": fixtures,
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
