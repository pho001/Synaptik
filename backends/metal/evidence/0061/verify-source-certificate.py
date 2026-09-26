#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CERTIFICATE = Path(__file__).with_name("source-compiler-site-certificate.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    kernel = ROOT / certificate["kernelSource"]["path"]
    foundation = ROOT / certificate["foundationSource"]["path"]
    assert digest(kernel) == certificate["kernelSource"]["sha256"]
    assert digest(foundation) == certificate["foundationSource"]["sha256"]
    kernel_text = kernel.read_text(encoding="utf-8")
    foundation_text = foundation.read_text(encoding="utf-8")
    assert kernel_text.count("fma(") == 1
    assert "accumulator = matmul_fma_0061(" in kernel_text
    assert "float accumulator = 0.0f" in kernel_text
    assert "leftWord << 16u" in kernel_text
    assert "rightWord << 16u" in kernel_text
    assert "options.mathMode = MTLMathModeSafe" in foundation_text
    assert "stringByAppendingString:SynaptikTask0061MatmulKernelSource" in foundation_text
    print("Task0061 source/compiler-site certificate verified")


if __name__ == "__main__":
    main()
