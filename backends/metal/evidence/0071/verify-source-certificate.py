#!/usr/bin/env python3
import hashlib
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
CERTIFICATE = EVIDENCE / "source-compiler-site-certificate.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(text: str, fragment: str) -> None:
    assert fragment in text, fragment


def main() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert certificate["abiVersion"] == 5
    assert certificate["exportCount"] == 13
    assert certificate["programSchema"] == 18
    assert certificate["generatorSchema"] == 2
    assert certificate["identityVersion"] == 27
    assert certificate["fixedCorpus"]["bytes"] == 84541
    assert certificate["fixedCorpus"]["sha256"] == (
        "5e7d65bfc46f52532781a3730aeda2202b798022b61fb60ea5beeb95307e77bb"
    )
    assert certificate["lifecycleTrace"] == "deferred-shared-native-resource-lifetime"

    texts = {}
    for source in certificate["sources"]:
        path = ROOT / source["path"]
        assert digest(path) == source["sha256"], source["path"]
        assert source["role"] not in texts
        texts[source["role"]] = path.read_text(encoding="utf-8")

    program = texts["program-image"]
    plan = texts["fusion-plan"]
    planner = texts["fusion-planner"]
    recognizer = texts["anchor-recognizer"]
    native = texts["native-runtime"]
    header = texts["anchor-kernels"]
    observer_test = texts["dispatch-observer-test"]
    raw_test = texts["raw-abi-test"]
    trace_test = texts["trace-test"]
    execution_trace_test = texts["execution-trace-test"]
    preparation_payload = texts["preparation-payload"]
    invocation_payload = texts["invocation-payload"]
    public_smoke = texts["public-smoke"]
    package_builder = texts["package-builder"]
    package_verifier = texts["package-verifier"]
    proof = texts["formal-proof"]

    for fragment in (
        "static final int SCHEMA_VERSION = 18;",
        "static final int MAGIC = 0x38314d53;",
        "encodedProgramImage(",
        "updateDigest(",
    ):
        require(program, fragment)
    for fragment in (
        "static final int GENERATOR_SCHEMA = 2;",
        "static final int FIXED_CORPUS_UTF8_BYTES = 84_541;",
        "static final int MAX_ANCHOR_UNITS = 64;",
        "static final int MAX_EXECUTION_INSTRUCTIONS = 512;",
        "ANCHOR_EPILOGUE(4)",
        "record AnchorInstruction",
    ):
        require(plan, fragment)
    for fragment in (
        "MetalAnchorEpilogueRecognizer.recognize",
        "anchorKindWire",
        "anchor-opcodes scalar-mul=1 add=2 relu=3 clamp=4",
        "execution-caps 64 512",
        "format 2",
        "schema 18",
        "generator 2",
    ):
        require(planner, fragment)
    for fragment in (
        "numericalProfile != NumericalProfile.ACCELERATOR",
        "case SCALAR_MUL",
        "case ADD",
        "case RELU",
        "case CLAMP",
        "broadcastsTo",
        "privateIntermediate",
        "external.externalValue() == anchorInput",
    ):
        require(recognizer, fragment)

    exports = re.findall(r"^SYNAPTIK_EXPORT \w+ (synaptik_[a-z0-9_]+)\(", native, re.MULTILINE)
    assert len(set(exports)) == 13
    for fragment in (
        "synaptik_metal_foundation_abi_version(void) { return 5U; }",
        "SYNAPTIK_ANCHOR_EPILOGUE_STEP",
        "synaptik_validate_anchor_step",
        "SynaptikMetalAnchorEpilogueMeta",
        "sizeof(SynaptikMetalAnchorEpilogueMeta) == 6096U",
        "options.mathMode = MTLMathModeSafe",
        "SynaptikTask0071AnchorEpilogueKernelSource",
        "return [fixed lengthOfBytesUsingEncoding:NSUTF8StringEncoding] == 84541U",
        "step.anchorEpilogue = YES",
        "step.epilogueInput = program_to_slot[step.epilogueInput]",
    ):
        require(native, fragment)
    assert "-ffast-math" not in header
    assert "fast::" not in header
    for symbol in (
        "anchor_mul_site_0071",
        "anchor_add_site_0071",
        "anchor_terminal_0071",
        "matmul_f32_f32_f32_epilogue_0071",
        "matmul_f32_bf16_f32_add_epilogue_0071",
        "conv2d_epilogue_0071",
        "conv2d_add_epilogue_0071",
    ):
        require(header, symbol)
    assert header.count("store_word(output, linear, 4u") == 8

    for fragment in (
        "matmulAnchorEpilogueRunsAsOneObservedPhysicalDispatch",
        "matmulAddOnlyAnchorMakesCustomRouteExecutable",
        "generatedPointwiseAfterAnchorUsesTheNextPlanOrdinal",
        "convRankOneExternalAddBroadcastsAcrossWidthNotChannel",
        "matmulClampCoversNanZerosInfinitiesAndSubnormalEndpoint",
        "assertEquals(1, observer.count())",
    ):
        require(observer_test, fragment)
    for fragment in (
        "nativeAbiFiveAcceptsCanonicalSchemaEighteenAndRejectsMalformedImages",
        "0x37314d53",
        "schema <= 17",
        "schemaEighteenRejectsCorruptFusionRecordsManifestAndDigest",
    ):
        require(raw_test, fragment)
    for fragment in (
        "structuralEventsDistinguishFusedAndComposedPlansAndPrecedeOutcomes",
        "structuralCallbackRuntimeAndErrorFailuresDisableWithoutEscaping",
        "structuralPayloadsExposeNoSensitiveOrResourceIdentityFields",
        "MetalInvocationPlan.class",
    ):
        require(trace_test, fragment)
    for fragment in (
        "MetalStructuralPrepareErrorDisablesBeforeFinalizationAndSkipsByteAggregation",
        "MetalInvocationPlanErrorDisablesWithoutAbortingNativeRun",
    ):
        require(execution_trace_test, fragment)
    for fragment in (
        "aggregateInputBytes",
        "aggregateOutputBytes",
        "aggregateInternalBytes",
        "aggregateSplatBytes",
        "aggregateWorkspaceBytes",
    ):
        require(trace_test, fragment)
        require(invocation_payload, fragment)
    for text in (preparation_payload, invocation_payload):
        for forbidden in ("sourceText", "dataValue", "scalarBits", "pointer", "handle", "path", "secret"):
            assert forbidden not in text
    for fragment in (
        "acceleratorMetalEngineRunsFusedMatmulAndConv2dEpiloguesWithStructuralTrace",
        "\"FUSED\"",
        "\"MATMUL\"",
        "\"CONV2D\"",
        "\"MetalInvocationPlan\"",
    ):
        require(public_smoke, fragment)
    require(package_builder, '\\"nodeSchemaVersion\\":18')
    require(package_verifier, '\\"nodeSchemaVersion\\":18')

    assert "sorry" not in proof
    assert "axiom " not in proof
    for fragment in (
        "scalar_add_terminal_source_order",
        "add_terminal_source_order",
        "one_physical_dispatch",
        "no_intermediate_store",
        "one_final_store",
        "rank_one_add_is_width_broadcast",
        "daz_boundary_is_exhaustive",
        "row_major_linear_decodes_width",
        "ftz_boundary_is_exhaustive",
    ):
        require(proof, fragment)

    audit_path = ROOT / certificate["compiledAirAudit"]["path"]
    assert digest(audit_path) == certificate["compiledAirAudit"]["sha256"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["runtimeMathMode"] == "MTLMathModeSafe"
    assert "-fno-fast-math" in audit["compilerFlags"]
    assert len(audit["kernelBodies"]) == 9
    assert all(not body["unsafeFlags"] for body in audit["kernelBodies"])
    assert sum(body["staticByteStores"] for body in audit["kernelBodies"]) == 8
    assert sum(body["fma"] for body in audit["kernelBodies"]) == 7
    assert sum(body["fmul"] for body in audit["kernelBodies"]) == 6
    assert sum(body["fadd"] for body in audit["kernelBodies"]) == 4
    print("Task0071 anchor-epilogue source/compiler-site certificate verified")


if __name__ == "__main__":
    main()
