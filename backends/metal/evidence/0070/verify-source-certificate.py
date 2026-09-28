#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CERTIFICATE = Path(__file__).with_name("source-compiler-site-certificate.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(text: str, fragment: str) -> None:
    assert fragment in text, fragment


def main() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert certificate["schema"] == "synaptik.metal.pointwise-fusion-certificate.v1"
    assert certificate["programSchema"] == 17
    assert certificate["generatorSchema"] == 1
    assert certificate["magic"] == "0x37314d53"

    texts: dict[str, str] = {}
    for entry in certificate["sources"]:
        path = ROOT / entry["path"]
        assert digest(path) == entry["sha256"], entry["path"]
        texts[entry["role"]] = path.read_text(encoding="utf-8")

    java_image = texts["java-program-image"]
    java_plan = texts["java-fusion-plan"]
    java_planner = texts["java-planner"]
    native = texts["native-parser-runtime"]
    generator = texts["native-generator"]
    native_build = texts["native-build"]
    observer_test = texts["observer-test"]
    observer = texts["observer-library"]
    public_smoke = texts["public-smoke"]
    planner_test = texts["planner-test"]
    raw_abi_test = texts["raw-abi-test"]
    audit_extractor = texts["audit-extractor"]
    audit_verifier = texts["audit-verifier"]
    native_audit_fixture = texts["native-audit-fixture"]
    cap_fixture = texts["cap-fixture"]
    audit_runner = texts["audit-runner"]
    proof_runner = texts["proof-runner"]
    observer_runner = texts["observer-runner"]
    runtime_matrix_test = texts["runtime-matrix-test"]
    proof_text = "\n".join(
        texts[role] for role in (
            "binary32-model", "floor-proof", "ceil-proof", "sign-proof", "relu-proof"
        )
    )

    for fragment in (
        "static final int SCHEMA_VERSION = 17;",
        "static final int HEADER_BYTES = 128;",
        "static final int MAGIC = 0x37314d53;",
        "MPSGraph image cannot carry an extension",
        ".putInt(extension ? EXECUTION_EXTENSION_PRESENT : 0)",
        "for (int value : plan.materializedProgramValueIndices()) out.putInt(value);",
    ):
        require(java_image, fragment)
    assert "if ((bytes & 7)" not in java_image
    for fragment in (
        "static final int MAX_GENERATED_UNITS = 32;",
        "static final int MAX_GENERATED_INSTRUCTIONS = 256;",
        "static final int MAX_FUNCTION_SOURCE_UTF8_BYTES = 16_384;",
        "static final int MAX_GENERATED_SOURCE_UTF8_BYTES = 262_144;",
        "static final int MAX_TOTAL_SOURCE_UTF8_BYTES = 1_048_576;",
        "static final int FIXED_CORPUS_UTF8_BYTES = 77_411;",
        "FLOOR(1, 10), CEIL(2, 9), SIGN(3, 9), RELU(4, 9)",
    ):
        require(java_plan, fragment)
    for fragment in (
        "if (length < 2) break;",
        "opcodes.size() < 2 || opcodes.size() > 8",
        "source-size-table 1",
        "fixed-corpus-bytes 77411",
        "pointmeta-abi size=32 align=8 elementCount=u64@0 gridWidth=u64@8 gridHeight=u64@16 scalar=u32@24 reserved=u32@28",
        "text.append(\"materialized \"",
        ".append(elements).append(' ').append(elements).append(\" 1 0 0\\n\")",
        "words.length == 4",
        "words[0] == 1L",
        "words[1] == 0L",
        "(words[2] == 0L || words[2] == 1L)",
        "words[3] == 0L",
        "input.dimensions()[0] == 1L",
        "eligiblePointwiseGeometry(input)",
        "value.rank() < 1 || value.rank() > MetalMpsGraphProgram.MAX_RANK",
        "elements > 0xffff_ffffL / dimension",
    ):
        require(java_planner, fragment)
    for forbidden in (
        "kernel void",
        "floor_bits",
        "ceil_bits",
        "sign_bits",
        "relu_bits",
        "MessageDigest",
        "fixed-corpus-sha256",
        "generated-source-sha256",
        "generatedSource(",
        "generatedSourceDigest",
    ):
        assert forbidden not in java_planner
    assert "source bytes and hashes remain exclusively" in java_plan

    for fragment in (
        "synaptik_read_le32(program) != UINT32_C(0x37314d53)",
        "synaptik_read_le32(program + 4U) != 17U",
        "synaptik_read_le32(program + 8U) != 128U",
        "BOOL extension = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM;",
        "members_offset + (uint64_t)member_count * 4U",
        "bindings_offset + (uint64_t)binding_count * 24U",
        "materialized_offset + (uint64_t)materialized_count * 4U",
        "instructions_offset + (uint64_t)instruction_count * 64U",
        "fusion->materialized_values[slot]",
        "uint64_t attributes_offset = references_end;",
        "synaptik_authenticated_fixed_source",
        "synaptik_assembled_source_is_authentic",
        "synaptik_validate_pointwise_cap_stop",
        "synaptik_pointwise_node_is_eligible",
        "member_count != node_count",
        "first_member >= node_count",
        "step.function_bytes > SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES",
        "SynaptikPointwiseGeneratedBytes(",
        "MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo",
        "binding.access != MTLBindingAccessReadWrite",
        "buffer.bufferPointerType.access != MTLBindingAccessReadWrite",
        "buffer.bufferPointerType.elementStructType != buffer.bufferStructType",
        "[buffer.bufferStructType memberByName:names[index]] != member",
        "reflection.bindings.count != 3U",
        "buffer.bufferDataSize != sizeof(SynaptikMetalPointMeta)",
        "pointmeta %u %llu %llu 1 0 0",
    ):
        require(native, fragment)
    assert native.count("SynaptikPointwiseGeneratedSource(") == 1
    assert "fixed-corpus-sha256" not in native
    assert "generated-source-sha256" not in native
    assert "generated_source_digest" not in native
    assert "task0053" not in native.lower()
    assert "task0053" not in native_build.lower()

    fixed_symbols = [
        "SynaptikExactKernelSource",
        "SynaptikTask0059DataKernelSource",
        "SynaptikTask0060ReductionKernelSource",
        "SynaptikTask0061MatmulKernelSource",
        "SynaptikTask0063OrderingKernelSource",
        "SynaptikTask0064ConvolutionPoolingKernelSource",
        "SynaptikTask0069AggregateKernelSource",
        "SynaptikTask0065RngDropoutKernelSource",
        "SynaptikTask0066DtypeLayoutKernelSource",
    ]
    positions = [native.index(symbol) for symbol in fixed_symbols]
    assert positions == sorted(positions)
    assert "(references_end + 7U)" not in native
    assert "if ((bytes & 7)" not in java_image
    fixed_digests = {
        component["symbol"]: component["sha256"]
        for component in certificate["fixedCorpus"]["components"]
    }
    assert list(fixed_digests) == fixed_symbols
    for symbol, expected_digest in fixed_digests.items():
        require(native, symbol)
        require(native, expected_digest)
    require(native, "stringByAppendingString:generated_source")

    for fragment in (
        "#define SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES 77411U",
        "#define SYNAPTIK_POINTWISE_MAX_UNITS 32U",
        "#define SYNAPTIK_POINTWISE_MAX_INSTRUCTIONS 256U",
        "#define SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES 16384U",
        "#define SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES 262144U",
        "#define SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES 1048576U",
        "typedef struct {\n    uint8_t *bytes;",
        "SynaptikPointwiseSinkWrite",
        "SynaptikPointwiseSinkFormat",
        "SynaptikPointwiseEmitFunction",
        "SynaptikPointwiseEmitGeneratedSource",
        "SynaptikPointwiseCapReason",
        "SynaptikPointwiseGeneratedBytesWithMode",
        "SynaptikPointwiseGeneratedSourceWithMode",
        "case 1U: return \"floor_bits\";",
        "case 2U: return \"ceil_bits\";",
        "case 3U: return \"sign_bits\";",
        "case 4U: return \"relu_bits\";",
        "!allow_audit_singleton && instruction_count < 2U",
        "kernel void synaptik_pw_g1_s%u",
    ):
        require(generator, fragment)
    assert "[NSMutableString" not in generator
    assert generator.count("kernel void synaptik_pw_g1_s%u") == 1
    for fragment in (
        "schemaSeventeenPacksOddReferencePoolDirectlyBeforeAttributes",
        "pointwiseGeometryRequiresRankOneThroughSixteenAndUnsignedElementCount",
        "thirtyThirdGeneratedUnitStopsAtTheAuthenticatedUnitCap",
    ):
        require(planner_test, fragment)
    for fragment in (
        "schemaSeventeenRejectsCorruptFusionRecordsManifestAndDigest",
        "schemaSeventeenAuthenticatesCapStopPrecedenceAndFirstRejectedNode",
        "schemaSeventeenRejectsFunctionCapAndCompactVirtualValueForgeries",
        "nativeParsesUnpaddedOddReferencePoolBeforeAttributes",
        "nativePointwiseEligibilityUsesExplicitRankAndUnsignedElementBounds",
        "appendOutOfRangeMemberStep",
        "fixed-corpus-bytes 77412",
    ):
        require(raw_abi_test, fragment)
    assert "fixed-corpus-sha256" not in raw_abi_test
    assert "generated-source-sha256" not in raw_abi_test
    for fragment in (
        "Path(sys.argv[1]).read_bytes()",
        '"generatedSha256"',
        '"assembledSha256"',
        "functionBytes",
    ):
        require(audit_extractor, fragment)
    assert "kernel void" not in audit_extractor
    assert "Path(sys.argv[1]).write" not in audit_extractor
    for fragment in (
        '#import "SynaptikPointwiseFusionKernelSource.h"',
        "SynaptikPointwiseFunctionBytesWithMode",
        "SynaptikPointwiseGeneratedBytesWithMode",
        "SynaptikPointwiseGeneratedSourceWithMode",
        "[first isEqualToString:second]",
        "memcmp(first_digest, second_digest",
        "writeToFile:",
    ):
        require(native_audit_fixture, fragment)
    assert "kernel void" not in native_audit_fixture
    assert native_audit_fixture.count("writeToFile:") == 1
    for reason in range(1, 6):
        require(cap_fixture, f"projection, {reason}U")
    require(cap_fixture, "SynaptikPointwiseCapReason")
    require(cap_fixture, "projection, 0U")
    require(audit_runner, 'generated-source-fixture.m -o "$WORK/generated-source-fixture"')
    require(audit_runner, '"$WORK/generated-source-fixture" "$WORK/generated-fixtures.metal"')
    assert audit_runner.index("generated-source-fixture.m -o") < audit_runner.index(
        "./extract-generated-fixtures.py"
    )
    require(proof_runner, "--rerun-tasks")
    require(observer_runner, "--rerun-tasks")
    require(runtime_matrix_test, "generatedChainsExecuteAllFourRawOperationsUnderBothProfiles")
    require(runtime_matrix_test, "MetalMpsGraphProgram.NodeKind.RELU")
    for fragment in (
        "singleton operation ledger drift",
        "2..8 chain ledger drift",
        "runtime reflection evidence drift",
        "floating AIR operation present in raw helper site",
    ):
        require(audit_verifier, fragment)

    observer_start = native.index("#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)", native.index("[encoder setBuffer:output.buffer"))
    callback = native.index("synaptik_metal_test_dispatch_observer(", observer_start)
    observer_end = native.index("#endif", callback)
    dispatch = native.index("[encoder dispatchThreads:step.grid", observer_end)
    assert observer_start < callback < observer_end < dispatch
    assert native[observer_end:dispatch].strip() == "#endif"
    require(observer_test, "assertEquals(2, observer.count());")
    require(observer_test, "oneGeneratedUnitProducesOneObservedNativeDispatch")
    require(observer_test, "mixedPlanExecutesGeneratedMpsGraphAndEveryRequiredFixedKind")
    require(observer_test, "assertEquals(4, observer.count());")
    require(observer_test, "MetalMpsGraphProgram.NodeKind.L1_NORM")
    require(observer_test, "MetalMpsGraphProgram.NodeKind.VARIANCE")
    require(observer_test, "MetalMpsGraphProgram.Node.scatterAdd")
    require(observer_test, "assertArrayEquals(fusion.canonicalManifestDigest(), observer.digest(record));")
    require(observer, "memcpy(record->manifest_digest, manifest_digest, sizeof(record->manifest_digest));")
    require(public_smoke, "cpuFreeMetalEngineRunsGeneratedPointwiseChainThroughOnePartition")
    require(public_smoke, "input.floor().ceil().sign()")

    assert "sorry" not in proof_text
    assert "axiom " not in proof_text
    for fragment in (
        "structure Bijection",
        "abbrev Raw32 := Fin (2 ^ 32)",
        "def rawWord (word : Word) : Raw32",
        "def wordOfRaw (raw : Raw32) : Word",
        "theorem wordOfRaw_rawWord",
        "theorem rawWord_wordOfRaw",
        "theorem raw_word_count : 2 * 256 * 8388608 = 2 ^ 32",
        "floor_universal_raw_certificate",
        "floor_permitted_iff",
        "floor_all_raw32_certificate",
        "ceil_universal_raw_certificate",
        "ceil_permitted_iff",
        "ceil_all_raw32_certificate",
        "sign_universal_raw_certificate",
        "sign_permitted_iff",
        "sign_all_raw32_certificate",
        "relu_universal_raw_certificate",
        "relu_permitted_iff",
        "relu_all_raw32_certificate",
    ):
        require(proof_text, fragment)

    compiled = json.loads((ROOT / certificate["compiledMslAudit"]["path"]).read_text(encoding="utf-8"))
    assert digest(ROOT / certificate["compiledMslAudit"]["path"]) == certificate["compiledMslAudit"]["sha256"]
    assert compiled["developerDir"] == "/Applications/Xcode.app/Contents/Developer"
    assert compiled["compilerFlags"] == [
        "-std=metal3.2", "-fno-fast-math", "-Wall", "-Werror", "-isysroot", "<absolute SDKROOT>"
    ]
    fixed = compiled["fixedCorpus"]
    assert fixed["bytes"] == certificate["fixedCorpus"]["bytes"] == 77411
    assert fixed["sha256"] == certificate["fixedCorpus"]["sha256"]
    assert fixed["components"] == certificate["fixedCorpus"]["components"]
    assert compiled["requiredAirFacts"]["bufferCount"] == 3
    assert compiled["requiredAirFacts"]["pointMetaBytes"] == 32
    assert compiled["requiredAirFacts"]["generatedSymbols"] == [
        f"synaptik_pw_g1_s{step}" for step in range(11)
    ]
    assert [fixture["instructions"] for fixture in compiled["fixtures"][:4]] == [
        ["FLOOR"], ["CEIL"], ["SIGN"], ["RELU"]
    ]
    assert [len(fixture["instructions"]) for fixture in compiled["fixtures"][4:]] == list(range(2, 9))
    reflection = compiled["runtimeReflection"]
    assert reflection["pipelineOptions"] == [
        "MTLPipelineOptionBindingInfo", "MTLPipelineOptionBufferTypeInfo"
    ]
    assert reflection["input"]["bindingAccess"] == "readOnly"
    assert reflection["output"]["bindingAccess"] == "readWrite"
    assert reflection["output"]["airAccess"] == "writeOnly"
    assert reflection["meta"]["members"] == [
        {"name": "elementCount", "type": "ulong", "offset": 0},
        {"name": "gridWidth", "type": "ulong", "offset": 8},
        {"name": "gridHeight", "type": "ulong", "offset": 16},
        {"name": "scalar", "type": "uint", "offset": 24},
        {"name": "reserved", "type": "uint", "offset": 28},
    ]
    print("Task0070 source/compiler-site certificate verified")


if __name__ == "__main__":
    main()
