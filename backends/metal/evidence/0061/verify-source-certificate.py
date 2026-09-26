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

    compiler = certificate["compilerSite"]
    assert compiler["runtimeCompiler"] == "MTLDevice.newLibraryWithSource"
    assert compiler["mathMode"] == "MTLMathModeSafe"
    assert compiler["fastMathRequested"] is False
    assert "options.mathMode = MTLMathModeSafe" in foundation_text
    assert "stringByAppendingString:SynaptikTask0061MatmulKernelSource" in foundation_text

    floating = certificate["floatingContract"]
    assert floating["sourceFmaSites"] == 1
    assert floating["reassociationSites"] == 0
    assert floating["pretruncationSites"] == 0
    assert kernel_text.count("fma(") == 1
    assert kernel_text.count("kernel void matmul_") == 7
    assert (
        "inline float matmul_fma_0061(float left, float right, float accumulator) "
        "{ return fma(left, right, accumulator); }"
    ) in kernel_text
    assert "float accumulator = 0.0f" in kernel_text
    assert (
        "for (ulong contraction = 0ul; contraction < contractions; ++contraction)"
        in kernel_text
    )
    assert kernel_text.count("accumulator = matmul_fma_0061(") == 1
    assert "leftWord << 16u" in kernel_text
    assert "rightWord << 16u" in kernel_text
    assert "as_type<uint>(matmul_f32_0061(" in kernel_text
    assert "accumulator += leftValue * rightValue" not in kernel_text
    assert "fast::fma" not in kernel_text

    inventory = certificate["siteInventory"]
    assert [site["id"] for site in inventory] == [
        "left-load",
        "right-load",
        "bfloat16-widen",
        "ordered-fma",
        "result-store",
    ]
    assert sum(site["floatingArithmetic"] for site in inventory) == 1
    assert all(site["proof"] for site in inventory)
    contributor = certificate["contributorProof"]
    assert set(contributor) == {
        "loopInitialization",
        "loopCondition",
        "loopAdvance",
        "contributorUse",
        "noDrop",
        "noReassociation",
        "noPretruncation",
    }
    special = certificate["specialValueProof"]
    assert set(special) == {
        "daz",
        "ftz",
        "nan",
        "infinity",
        "signedZero",
        "finalNonemptyZeroSignFreedom",
        "finiteContributorPreservation",
    }
    assert all(special.values())
    print("Task0061 source/compiler-site certificate verified")


if __name__ == "__main__":
    main()
