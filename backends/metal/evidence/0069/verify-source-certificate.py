#!/usr/bin/env python3
import hashlib
import json
import runpy
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
    fixed_source = runpy.run_path(
        str(Path(__file__).with_name("extract-runtime-source.py"))
    )["runtime_source"]().encode("utf-8")
    fixed_entry = certificate["fixedRuntimeSource"]
    compiled = json.loads(compiled_audit.read_text(encoding="utf-8"))
    assert certificate["status"] == compiled["status"] == "accepted"
    assert compiled["runtimeCompilerSite"]["sourceAssembly"] == (
        "exact ordered concatenation of the ten digest-authenticated production "
        "NSString constants; no Task0053 source is assembled"
    )
    assert len(fixed_source) == fixed_entry["bytes"]
    assert hashlib.sha256(fixed_source).hexdigest() == fixed_entry["sha256"]
    assert compiled["artifacts"]["source"]["bytes"] == fixed_entry["bytes"]
    assert compiled["artifacts"]["source"]["sha256"] == fixed_entry["sha256"]
    authenticated_fixed_source = foundation_text.split(
        "static NSString *synaptik_authenticated_fixed_source(void) {", 1
    )[1].split("static NSString *synaptik_authenticated_low_precision_source(void) {", 1)[0]
    assert f'== {fixed_entry["bytes"]}U' in authenticated_fixed_source
    assert f'"{fixed_entry["sha256"]}"' in authenticated_fixed_source

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
    scatter = certificate["scatterAddContract"]
    variance = certificate["varianceContract"]
    assert floating["sourceAdditionSites"] == 1
    assert scatter["sourceAdditionSites"] == 1
    assert variance["sourceDivisionSites"] == 2
    assert variance["sourceSubtractionSites"] == 1
    assert variance["sourceMultiplicationSites"] == 1
    assert "profile-free occurrence" in variance["geometry"]
    assert "accelerator profile" not in variance["geometry"]
    assert floating["reassociationSites"] == 0
    assert scatter["reassociationSites"] == 0
    assert floating["pretruncationSites"] == 0
    assert scatter["pretruncationSites"] == 0
    assert kernel_text.count("inline float l1_add_0069(") == 1
    assert kernel_text.count("l1_add_0069(") == 2
    assert kernel_text.count("inline float scatter_add_site_0069(") == 1
    assert kernel_text.count("scatter_add_site_0069(") == 2
    assert kernel_text.count("return accumulator + contributor;") == 2
    assert kernel_text.count("store_word(output, 0ul, 4u") == 2
    assert kernel_text.count("store_word(output, target, 4u") == 1
    assert "firstWord = l1_abs_word_0069(uint(load_word(input, 0ul, 4u)))" in kernel_text
    assert "float accumulator = as_type<float>(firstWord)" in kernel_text
    assert (
        "for (ulong contributor = 1ul; contributor < m.inputExtents[0]; ++contributor)"
        in kernel_text
    )
    assert "word & 0x7fffffffu" in kernel_text
    assert (
        "for (ulong ordinal = 0ul; ordinal < m.auxDims[0]; ++ordinal)"
        in kernel_text
    )
    assert "selected == long(target)" in kernel_text
    assert "addressed ? as_type<uint>(accumulator) : baseWord" in kernel_text
    assert kernel_text.count("inline float variance_div_site_0069(") == 1
    assert kernel_text.count("variance_div_site_0069(") == 3
    assert kernel_text.count("inline float variance_sub_site_0069(") == 1
    assert kernel_text.count("variance_sub_site_0069(") == 2
    assert kernel_text.count("inline float variance_mul_site_0069(") == 1
    assert kernel_text.count("variance_mul_site_0069(") == 2
    assert "float mean = variance_div_site_0069(x, one)" in kernel_text
    assert "float difference = variance_sub_site_0069(x, mean)" in kernel_text
    assert "float square = variance_mul_site_0069(difference, difference)" in kernel_text
    assert "float result = variance_div_site_0069(square, one)" in kernel_text
    assert "m.elementCount != 1ul || m.inputExtents[0] != 1ul" in kernel_text
    assert "accumulator = 0.0" not in kernel_text
    for forbidden in ("fma(", "fast::", "simdgroup", "atomic_", "epsilon"):
        assert forbidden not in kernel_text

    assert (
        "BOOL task0069_l1 = node.operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM"
        in foundation_text
    )
    assert (
        "BOOL task0069_scatter = node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD"
        in foundation_text
    )
    assert (
        "BOOL task0069_variance = node.operation == SYNAPTIK_METAL_MPSGRAPH_VARIANCE"
        in foundation_text
    )
    assert "meta.inputExtents[0] = shape_element_count(input)" in foundation_text
    assert "step.grid = MTLSizeMake(1U, 1U, 1U)" in foundation_text
    assert "return @\"l1_norm_f32_0069\"" in foundation_text
    assert "return @\"scatter_add_f32_0069\"" in foundation_text
    assert "return @\"variance_f32_0069\"" in foundation_text
    assert "meta.reserved = UINT32_C(0x3f800000)" in foundation_text
    assert "validation.preflight = YES" in foundation_text
    assert "validate_index_buffer(validation, index_buffer)" in foundation_text
    preflight_scan = foundation_text.index(
        "for (SynaptikMetalIndexValidation *validation in box.indexValidations)"
    )
    program_dispatch = foundation_text.index(
        "while (cursor < box.programSteps.count)", preflight_scan
    )
    command_buffer = foundation_text.index(
        "id<MTLCommandBuffer> command_buffer", program_dispatch
    )
    assert preflight_scan < program_dispatch < command_buffer

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
    assert "exactRneFiniteSigned" in proof_text
    assert "exactRneGeneral" in proof_text
    assert "structure GeneralBinary32RneContract" in proof_text
    assert "generalBinary32AddSite rne contract" in proof_text
    assert "source_result_l1_class_contract" in proof_text
    assert "sourceFold_nan_of_contains" in proof_text
    assert "sourceFold_positive_infinity_of_contains" in proof_text
    assert "absWord_class_contract" in proof_text
    assert "def L1Reachable (word : Word) : Prop :=" in proof_text
    assert "rawClass word = .nan ∨ word.sign = false" in proof_text
    assert "rawClass left = .nan → L1Reachable right" in proof_text
    assert "rawClass right = .nan → L1Reachable left" in proof_text
    assert "daz_preserves_l1_reachable" in proof_text
    assert "L1ClassInvariant current →" in proof_text
    assert "AllLeaves (fun leaf => L1ClassInvariant leaf.word) rest →" in proof_text
    assert "def labelScatterFrom" in proof_text
    assert "labelScatterFrom_source_positions" in proof_text
    assert "matchingOccurrences_duplicate_multiplicity" in proof_text
    assert "matchingOccurrences_preserve_source_payloads" in proof_text
    assert "matchingOccurrences_in_source_order" in proof_text
    assert "matchingLeaves_ordinals" in proof_text
    assert "scatter_source_result_is_general_binary32_model_result" in proof_text
    assert "scatter_unaddressed_raw_identity" in proof_text
    assert "scatterWriterThread_injective" in proof_text
    assert "def binary32PrimitiveSite" in proof_text
    assert "def exactDivRne" in proof_text
    assert "def exactSubRne" in proof_text
    assert "def exactMulRne" in proof_text
    assert "def task0069ExactSubValue" in proof_text
    assert "def task0069RneProductBelowHalf" in proof_text
    assert "exactRneSubValue" not in proof_text
    assert "exactRneProductNumerator" not in proof_text
    assert "natSignedDifference_bound" in proof_text
    assert "smallFinite_difference_bound" in proof_text
    assert "smallFinite_product_below_half" in proof_text
    assert "singletonVarianceLiteralSource_exact" in proof_text
    assert "binary32PrimitiveSite exactDivRne source typedOne mean" in proof_text
    assert "binary32PrimitiveSite exactSubRne source mean difference" in proof_text
    assert "binary32PrimitiveSite exactMulRne difference difference square" in proof_text
    assert "binary32PrimitiveSite exactDivRne square typedOne result" in proof_text
    assert "DivOneContract" not in proof_text
    assert "SubtractionContract" not in proof_text
    assert "MultiplicationContract" not in proof_text
    assert "singletonVariance_special_is_nan" in proof_text
    assert "singletonVariance_finite_is_positiveZero" in proof_text
    assert "singletonVariance_only_thread_zero_dispatched" in proof_text
    assert "singletonVariance_store_once" in proof_text

    inventory = certificate["siteInventory"]
    assert [site["id"] for site in inventory] == [
        "raw-load-and-abs",
        "ordered-add",
        "raw-store",
        "scatter-raw-base",
        "scatter-stable-filtered-add",
        "scatter-single-store",
        "variance-mean-div",
        "variance-difference-sub",
        "variance-square-mul",
        "variance-final-div",
        "variance-single-store",
    ]
    assert sum(site["floatingArithmetic"] for site in inventory) == 6
    assert all(site["proof"] for site in inventory)
    assert all(certificate["specialValueProof"].values())
    assert all(certificate["scatterContributorProof"].values())
    assert all(certificate["varianceSpecialValueProof"].values())
    assert certificate["proof"]["compiledMslAudit"]
    print("Task0069 source/compiler-site certificate verified")


if __name__ == "__main__":
    main()
