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
    binary32 = ROOT / certificate["leanProofs"][0]["path"]
    tree = ROOT / certificate["leanProofs"][1]["path"]
    compiled_audit = ROOT / certificate["compiledMslAudit"]["path"]
    for entry, path in (
        (certificate["kernelSource"], kernel),
        (certificate["foundationSource"], foundation),
        (certificate["leanProofs"][0], binary32),
        (certificate["leanProofs"][1], tree),
        (certificate["compiledMslAudit"], compiled_audit),
    ):
        assert digest(path) == entry["sha256"]

    kernel_text = kernel.read_text(encoding="utf-8")
    foundation_text = foundation.read_text(encoding="utf-8")
    proof_text = binary32.read_text(encoding="utf-8") + tree.read_text(encoding="utf-8")

    compiler = certificate["compilerSite"]
    assert compiler["runtimeCompiler"] == "MTLDevice.newLibraryWithSource"
    assert compiler["mathMode"] == "MTLMathModeSafe"
    assert compiler["fastMathRequested"] is False
    assert compiler["languageVersion"] == "MTLLanguageVersion3_2"
    assert compiler["compiledMslAuditRequired"] is True
    assert "options.mathMode = MTLMathModeSafe" in foundation_text
    assert "options.languageVersion = MTLLanguageVersion3_2" in foundation_text
    assert "newLibraryWithSource:kernel_source options:options" in foundation_text
    assert "stringByAppendingString:SynaptikTask0069AggregateKernelSource" in foundation_text

    floating = certificate["floatingContract"]
    assert floating["sourceAdditionSites"] == 1
    assert floating["reassociationSites"] == 0
    assert floating["pretruncationSites"] == 0
    assert kernel_text.count("inline float l1_add_0069(") == 1
    assert kernel_text.count("l1_add_0069(") == 2
    assert kernel_text.count("return accumulator + contributor;") == 1
    assert kernel_text.count("store_word(output, 0ul, 4u") == 1
    assert "firstWord = l1_abs_word_0069(uint(load_word(input, 0ul, 4u)))" in kernel_text
    assert "float accumulator = as_type<float>(firstWord)" in kernel_text
    assert (
        "for (ulong contributor = 1ul; contributor < m.inputExtents[0]; ++contributor)"
        in kernel_text
    )
    assert "word & 0x7fffffffu" in kernel_text
    assert "accumulator = 0.0" not in kernel_text
    for forbidden in ("fma(", "fast::", "simdgroup", "atomic_", "epsilon"):
        assert forbidden not in kernel_text

    assert "BOOL task0069 = node.operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM" in foundation_text
    assert "meta.inputExtents[0] = shape_element_count(input)" in foundation_text
    assert "step.grid = MTLSizeMake(1U, 1U, 1U)" in foundation_text
    assert "return @\"l1_norm_f32_0069\"" in foundation_text

    assert "sorry" not in proof_text
    assert "axiom " not in proof_text
    assert "sourceTree_has_n_minus_one_additions" in proof_text
    assert "singleton_source_is_direct_abs" in proof_text
    assert "source_result_is_binary32_model_result" in proof_text
    assert "daz_complete" in proof_text
    assert "ftz_complete" in proof_text
    assert "typedOne : Word" in proof_text
    assert "typedOne_class" in proof_text
    assert "exactRneNonnegative" in proof_text
    assert "structure Binary32RneContract" in proof_text
    assert "binary32AddSite rne contract" in proof_text
    assert "source_result_l1_class_contract" in proof_text
    assert "sourceFold_nan_of_contains" in proof_text
    assert "sourceFold_positive_infinity_of_contains" in proof_text
    assert "absWord_class_contract" in proof_text

    inventory = certificate["siteInventory"]
    assert [site["id"] for site in inventory] == [
        "raw-load-and-abs",
        "ordered-add",
        "raw-store",
    ]
    assert sum(site["floatingArithmetic"] for site in inventory) == 1
    assert all(site["proof"] for site in inventory)
    assert all(certificate["specialValueProof"].values())
    print("Task0069 source/compiler-site certificate verified")
    assert certificate["proof"]["compiledMslAudit"]


if __name__ == "__main__":
    main()
