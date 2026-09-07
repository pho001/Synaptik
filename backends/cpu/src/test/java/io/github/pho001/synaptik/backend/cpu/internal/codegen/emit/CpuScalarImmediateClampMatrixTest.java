package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Collections;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import jdk.incubator.vector.FloatVector;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import org.junit.jupiter.api.Test;

/** Source-derived admission and exact artifact fixture closure for scalar immediates and clamp. */
class CpuScalarImmediateClampMatrixTest {
    private static final String RESOURCE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";

    /** The checked table is a seal, not a fixture generator: source must close it both ways. */
    @Test void sealedFixtureTableIsVersionedExactAndBidirectionallyClosed() throws Exception {
        String outputDirectory = System.getenv("SYNAPTIK_CPU_SCALAR_MATRIX_WRITE");
        if (outputDirectory != null) {
            CpuScalarImmediateClampMatrixOracle.writeResources(Path.of(outputDirectory));
            return;
        }
        var rows = table("scalar-immediate-clamp-matrix-fixtures.tsv",
                List.of("id", "operation", "type", "category", "model-capability",
                        "lowering-admitted", "preparation-eligible", "carrier-input", "carrier-output",
                        "preference", "parallelism", "entry-descriptor", "strategy", "generator-schema",
                        "class-identity-schema", "structural-key", "class-file-sha256"));
        assertEquals(203, rows.size(), "finite semantic/category basis");
        Map<String, CpuScalarImmediateClampMatrixOracle.Artifact> actual = new LinkedHashMap<>();
        for (var fixture : CpuScalarImmediateClampMatrixOracle.fixtures()) actual.put(fixture.id(),
                CpuScalarImmediateClampMatrixOracle.artifact(fixture));
        assertEquals(actual.keySet().stream().toList(), rows.keySet().stream().toList(),
                "no missing, unknown, stale, or reordered sealed fixture");
        for (var entry : actual.entrySet()) {
            var fixture = entry.getValue().fixture(); var artifact = entry.getValue(); var row = rows.get(entry.getKey());
            assertEquals(fixture.operation().name(), row.get("operation"), fixture.id());
            assertEquals(fixture.type().name(), row.get("type"), fixture.id());
            assertEquals(fixture.category().name(), row.get("category"), fixture.id());
            assertEquals("true", row.get("model-capability"), fixture.id());
            assertTrue(CpuScalarImmediateClampMatrixOracle.capability(fixture), fixture.id());
            assertEquals("true", row.get("lowering-admitted"), fixture.id());
            assertEquals("true", row.get("preparation-eligible"), fixture.id());
            assertEquals(fixture.carriers().getFirst().name(), row.get("carrier-input"), fixture.id());
            assertEquals(fixture.carriers().getLast().name(), row.get("carrier-output"), fixture.id());
            assertEquals(fixture.preference().name(), row.get("preference"), fixture.id());
            assertEquals(Integer.toString(fixture.parallelism()), row.get("parallelism"), fixture.id());
            assertEquals(artifact.descriptor(), row.get("entry-descriptor"), fixture.id());
            assertEquals(artifact.strategy(), row.get("strategy"), fixture.id());
            assertEquals("64", artifact.generatorSchema(), fixture.id());
            assertEquals(artifact.generatorSchema(), row.get("generator-schema"), fixture.id());
            assertEquals(artifact.classIdentitySchema(), row.get("class-identity-schema"), fixture.id());
            assertEquals(artifact.structuralKey(), row.get("structural-key"), fixture.id());
            assertEquals(artifact.hash(), row.get("class-file-sha256"), fixture.id());
        }
        var formRows = table("scalar-immediate-clamp-matrix-forms.tsv", formHeader());
        var forms = validateFormRows(formRows);
        var projections = table("scalar-immediate-clamp-matrix-projections.tsv", projectionHeader());
        assertEquals(List.of("array-scalar-orchestration", "array-vector-orchestration"),
                projections.keySet().stream().toList(), "exact ordered source-proved projections");
        for (var projection : projections.values()) {
            assertEquals(CpuScalarImmediateClampMatrixStructuralTest.NORMALIZER_VERSION,
                    projection.get("normalizer-version"));
            assertEquals("IDENTITY", projection.get("disposition"), "orchestration is byte identity, never constants-only");
            assertEquals("CALLER_ORCHESTRATION", projection.get("source-category"));
            assertTrue(projection.get("specialization-facts").endsWith("/SCALAR")
                    || projection.get("specialization-facts").endsWith("/VECTOR"));
            var ids = List.of(projection.get("members").split(",", -1));
            var hashes = List.of(projection.get("member-sha256").split(",", -1));
            assertEquals(ids.size(), hashes.size());
            assertEquals(ids.size(), new java.util.LinkedHashSet<>(ids).size(), "projection duplicate member");
            for (int index = 0; index < ids.size(); index++) {
                var artifact = forms.get(ids.get(index));
                assertNotNull(artifact, "projection exact member " + ids.get(index));
                assertEquals(artifact.hash(), hashes.get(index));
            }
            assertEquals("NONE", projection.get("constant-locations"));
            var members = ids.stream().map(forms::get).toList();
            assertTrue(members.stream().map(CpuScalarImmediateClampMatrixOracle.Artifact::hash).distinct().count() == 1,
                    "identity projection has one exact Class-File hash");
        }
    }

    private static Map<String, Map<String, String>> table(String name, List<String> expectedHeader) throws Exception {
        return parseTable(name, resourceText(name), expectedHeader);
    }
    private static String resourceText(String name) throws Exception {
        try (InputStream stream = CpuScalarImmediateClampMatrixTest.class.getResourceAsStream(RESOURCE + name)) {
            assertNotNull(stream, name); return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
    /** Strict parser shared unchanged by checked resources and hostile in-memory mutations. */
    private static Map<String, Map<String, String>> parseTable(String name, String text, List<String> expectedHeader) {
        assertFalse(text.contains("\\t"), name + " uses a literal backslash-t");
        assertFalse(text.contains("\r"), name + " must use canonical LF newlines");
        assertTrue(text.endsWith("\n"), name + " must end with one canonical LF");
        assertFalse(text.endsWith("\n\n"), name + " has an extra trailing blank row");
        var sourceLines = text.lines().toList(); assertFalse(sourceLines.isEmpty(), name);
        String metadata = sourceLines.getFirst();
        String prefix = "# matrix-resource-sha256=";
        assertTrue(metadata.matches("^" + prefix + "[0-9a-f]{64}; canonical UTF-8 bytes of every header/data row plus LF, excluding this metadata line$"),
                name + " malformed digest convention");
        String expectedDigest = metadata.substring(prefix.length(), prefix.length() + 64);
        var lines = sourceLines.subList(1, sourceLines.size()); assertFalse(lines.isEmpty(), name + " missing header");
        assertEquals(expectedDigest, CpuScalarImmediateClampMatrixOracle.sha256(
                (String.join("\n", lines) + "\n").getBytes(StandardCharsets.UTF_8)), name + " canonical digest");
        var header = List.of(lines.getFirst().split("\\t", -1)); assertEquals(expectedHeader, header, name);
        Map<String, Map<String, String>> result = new LinkedHashMap<>();
        for (int line = 1; line < lines.size(); line++) {
            String value = lines.get(line); assertFalse(value.isBlank(), name + " blank row " + line);
            String[] cells = value.split("\\t", -1); assertEquals(header.size(), cells.length, name + " row " + line);
            for (String cell : cells) assertFalse(cell.isBlank(), name + " blank cell " + line);
            Map<String, String> row = new LinkedHashMap<>();
            for (int column = 0; column < header.size(); column++) row.put(header.get(column), cells[column]);
            assertNull(result.put(cells[0], Map.copyOf(row)), name + " duplicate key " + cells[0]);
            assertFalse(value.contains("UNSEALED") || value.contains("SOURCE_DERIVED")
                    || value.contains("validated-at-runtime") || value.contains("placeholder"), name + " unsealed marker");
        }
        return Collections.unmodifiableMap(result);
    }
    private static List<String> formHeader() {
        return List.of("id", "operation", "type", "category", "model-capability", "lowering-admitted",
                "preparation-eligible", "carrier-input", "carrier-output", "shape", "layout",
                "access-regime", "preference", "parallelism", "selected-strategy", "artifact-strategy", "entry-descriptor",
                "matrix-version", "generator-schema", "class-identity-schema", "structural-key",
                "class-file-sha256", "materialization-candidates", "materialization-selected", "disposition");
    }
    private static Map<String, CpuScalarImmediateClampMatrixOracle.Artifact> validateFormRows(
            Map<String, Map<String, String>> rows) {
        assertEquals(256, rows.size(), "finite compositional artifact ledger");
        Map<String, CpuScalarImmediateClampMatrixOracle.Artifact> actual = new LinkedHashMap<>();
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            var formArtifact = CpuScalarImmediateClampMatrixOracle.formArtifact(form);
            var artifact = formArtifact.artifact(); var row = rows.get(form.id());
            assertNotNull(row, "missing representative " + form.id()); actual.put(form.id(), artifact);
            assertEquals("matrix-v3", row.get("matrix-version"), form.id() + " matrix version");
            assertEquals(form.fixture().operation().name(), row.get("operation"), form.id());
            assertEquals(form.fixture().type().name(), row.get("type"), form.id());
            assertEquals(form.fixture().category().name(), row.get("category"), form.id());
            assertEquals("true", row.get("model-capability"), form.id());
            assertTrue(CpuScalarImmediateClampMatrixOracle.capability(form.fixture()), form.id());
            assertEquals("true", row.get("lowering-admitted"), form.id());
            assertEquals("true", row.get("preparation-eligible"), form.id());
            assertEquals(form.fixture().carriers().getFirst().name(), row.get("carrier-input"), form.id());
            assertEquals(form.fixture().carriers().getLast().name(), row.get("carrier-output"), form.id());
            assertEquals(form.shape(), row.get("shape"), form.id()); assertEquals(form.layout(), row.get("layout"), form.id());
            assertEquals(form.accessRegime().name(), row.get("access-regime"), form.id());
            assertEquals(form.fixture().preference().name(), row.get("preference"), form.id());
            assertEquals(Integer.toString(form.fixture().parallelism()), row.get("parallelism"), form.id());
            assertEquals(strategyName(new CpuPartitionPreparer().analyze(CpuScalarImmediateClampMatrixOracle.contextFor(form.fixture(), form.inputShape(),
                    form.inputLayout(), form.outputShape(), form.outputLayout(), form.materializationPolicy())).plan().units().getFirst().executionStrategy()), row.get("selected-strategy"), form.id());
            assertEquals(artifact.strategy(), row.get("artifact-strategy"), form.id());
            assertEquals(artifact.descriptor(), row.get("entry-descriptor"), form.id());
            assertEquals("64", artifact.generatorSchema(), form.id());
            assertEquals(artifact.generatorSchema(), row.get("generator-schema"), form.id());
            assertEquals(artifact.classIdentitySchema(), row.get("class-identity-schema"), form.id());
            assertEquals(artifact.structuralKey(), row.get("structural-key"), form.id());
            assertEquals(artifact.hash(), row.get("class-file-sha256"), form.id());
            assertEquals(Integer.toString(formArtifact.materializationCandidates()), row.get("materialization-candidates"), form.id());
            assertEquals(Integer.toString(formArtifact.materializationSelected()), row.get("materialization-selected"), form.id());
            assertEquals(formArtifact.materializationCandidates() == 0 ? "DIRECT_ONLY"
                    : "CANDIDATE_AVAILABLE_DIRECT_SELECTED", row.get("disposition"), form.id());
        }
        assertEquals(actual.keySet().stream().toList(), rows.keySet().stream().toList(),
                "no unknown, stale, or reordered representative");
        return Collections.unmodifiableMap(actual);
    }
    private static String strategyName(ExecutionStrategy strategy) {
        if (strategy.equals(ExecutionStrategy.SCALAR)) return "SCALAR";
        if (strategy.equals(ExecutionStrategy.PARALLEL_SCALAR)) return "PARALLEL_SCALAR";
        if (strategy.equals(ExecutionStrategy.VECTOR)) return "VECTOR";
        if (strategy.equals(ExecutionStrategy.PARALLEL_VECTOR)) return "PARALLEL_VECTOR";
        throw new AssertionError(strategy);
    }

    @Test void resourceAndProjectionMutationsFailClosedThroughTheCheckedParserAndValidator() throws Exception {
        String forms = resourceText("scalar-immediate-clamp-matrix-forms.tsv");
        String projections = resourceText("scalar-immediate-clamp-matrix-projections.tsv");
        for (String mutation : List.of(reseal(forms.replaceFirst("\\toperation\\t", "\\twrong-operation\\t")),
                reseal(forms.replaceFirst("matrix-v3", "matrix-v9")), forms.replaceFirst("[0-9a-f]{64}", "0".repeat(64)),
                reseal(forms.replaceFirst("DIRECT_ONLY\\n", "DIRECT_ONLY\\tEXTRA\\n")), reseal(forms.replaceFirst("\\t", "\\\\t")),
                reseal(forms.replaceFirst("ADD", "UNSEALED")), reseal(forms + forms.lines().skip(2).findFirst().orElseThrow() + "\n"),
                forms.replace("\n", "\r\n"), forms.substring(0, forms.length() - 1), swapFirstDataRows(forms)))
            assertThrows(AssertionError.class, () -> validateFormRows(parseTable("forms", mutation, formHeader())));
        var parsed = parseTable("forms", forms, formHeader());
        String firstId = parsed.keySet().iterator().next();
        assertThrows(AssertionError.class, () -> validateFormRows(new LinkedHashMap<>(parsed) {{ remove(firstId); }}));
        assertThrows(AssertionError.class, () -> validateFormRows(new LinkedHashMap<>(parsed) {{ put("UNKNOWN", get(firstId)); }}));
        assertThrows(AssertionError.class, () -> validateFormRows(new LinkedHashMap<>(parsed) {{
            put(firstId, new LinkedHashMap<>(get(firstId)) {{ put("operation", "UNKNOWN"); }}); }}));
        for (String mutation : List.of(reseal(projections.replace("ORCHESTRATION-PARALLEL_SCALAR", "UNKNOWN")),
                reseal(projections.replace("ORCHESTRATION-PARALLEL_SCALAR", "ORCHESTRATION-SCALAR,ORCHESTRATION-SCALAR")),
                reseal(projections.replaceFirst("fd091e[0-9a-f]{58}", "f".repeat(64))), reseal(projections.replace("IDENTITY", "PROVED_CONSTANTS_ONLY")),
                reseal(projections.replace("NONE", "1"))))
            assertThrows(AssertionError.class, () -> validateProjections(parseTable("projections", mutation,
                    projectionHeader()), validateFormRows(parsed)));
    }
    private static String reseal(String text) {
        var lines = text.lines().toList(); String body = String.join("\n", lines.subList(1, lines.size())) + "\n";
        return "# matrix-resource-sha256=" + CpuScalarImmediateClampMatrixOracle.sha256(body.getBytes(StandardCharsets.UTF_8))
                + "; canonical UTF-8 bytes of every header/data row plus LF, excluding this metadata line\n" + body;
    }
    private static String swapFirstDataRows(String text) {
        var lines = new java.util.ArrayList<>(text.lines().toList());
        Collections.swap(lines, 2, 3);
        return reseal(String.join("\n", lines) + "\n");
    }
    private static void validateProjections(Map<String, Map<String, String>> projections,
            Map<String, CpuScalarImmediateClampMatrixOracle.Artifact> forms) {
        for (var projection : projections.values()) {
            assertEquals(CpuScalarImmediateClampMatrixStructuralTest.NORMALIZER_VERSION, projection.get("normalizer-version"));
            assertEquals("IDENTITY", projection.get("disposition")); assertEquals("NONE", projection.get("constant-locations"));
            assertEquals("CALLER_ORCHESTRATION", projection.get("source-category"));
            assertFalse(projection.get("specialization-facts").isBlank());
            var ids = List.of(projection.get("members").split(",", -1)); var hashes = List.of(projection.get("member-sha256").split(",", -1));
            assertEquals(ids.size(), hashes.size()); assertEquals(ids.size(), new java.util.LinkedHashSet<>(ids).size());
            for (int i = 0; i < ids.size(); i++) { assertNotNull(forms.get(ids.get(i))); assertEquals(forms.get(ids.get(i)).hash(), hashes.get(i)); }
        }
    }
    private static List<String> projectionHeader() {
        return List.of("unit", "normalizer-version", "disposition", "source-category",
                "specialization-facts", "members", "member-sha256", "constant-locations");
    }
    @Test void everyFiniteSourceCategoryBuildsAnExactPreparedArtifact() {
        var fixtures = CpuScalarImmediateClampMatrixOracle.fixtures();
        assertEquals(203, fixtures.size());
        assertEquals(Map.of(ScalarElementwiseKind.ADD, 28L, ScalarElementwiseKind.SUB, 28L,
                ScalarElementwiseKind.MUL, 28L, ScalarElementwiseKind.DIV, 18L,
                ScalarElementwiseKind.MIN, 28L, ScalarElementwiseKind.MAX, 28L,
                ScalarElementwiseKind.POW, 21L, ScalarElementwiseKind.CLAMP, 24L),
                fixtures.stream().collect(java.util.stream.Collectors.groupingBy(
                        CpuScalarImmediateClampMatrixOracle.Fixture::operation,
                        java.util.stream.Collectors.counting())));
        assertEquals(Map.of(DataType.BFLOAT16, 51L, DataType.FLOAT32, 51L,
                DataType.FLOAT64, 51L, DataType.INT32, 25L, DataType.INT64, 25L),
                fixtures.stream().collect(java.util.stream.Collectors.groupingBy(
                        CpuScalarImmediateClampMatrixOracle.Fixture::type,
                        java.util.stream.Collectors.counting())));
        for (var fixture : fixtures) {
            var artifact = CpuScalarImmediateClampMatrixOracle.artifact(fixture);
            assertEquals(64, artifact.hash().length(), fixture.id());
            assertFalse(artifact.descriptor().isBlank(), fixture.id());
            assertTrue(List.of("scalar", "vector").contains(artifact.strategy()), fixture.id());
            assertEquals("64", artifact.generatorSchema(), fixture.id());
            assertEquals(fixture.type() == DataType.BFLOAT16 ? "59" : "52",
                    artifact.classIdentitySchema(), fixture.id());
        }
    }

    @Test void loweringPreservesRawSignedZeroAndEverySourcePowerRealization() {
        var realized = java.util.EnumSet.noneOf(
                io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.PowerRealization.class);
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            var positive = fixture("ADD-" + type + "-POSITIVE_ZERO");
            var negative = fixture("ADD-" + type + "-NEGATIVE_ZERO");
            long positiveBits = CpuScalarImmediateClampMatrixOracle.instruction(positive)
                    .scalarImmediate().bits();
            long negativeBits = CpuScalarImmediateClampMatrixOracle.instruction(negative)
                    .scalarImmediate().bits();
            assertEquals(0L, positiveBits, type + " positive zero");
            assertEquals(switch (type) {
                case BFLOAT16 -> 0x8000L;
                case FLOAT32 -> 0x8000_0000L;
                case FLOAT64 -> 0x8000_0000_0000_0000L;
                default -> throw new AssertionError(type);
            }, negativeBits, type + " negative zero");
            assertNotEquals(CpuScalarImmediateClampMatrixOracle.artifact(positive).hash(),
                    CpuScalarImmediateClampMatrixOracle.artifact(negative).hash(), type.toString());
        }
        for (var fixture : CpuScalarImmediateClampMatrixOracle.fixtures()) {
            if (fixture.operation() == ScalarElementwiseKind.POW) {
                realized.add(CpuScalarImmediateClampMatrixOracle.instruction(fixture)
                        .powerRealization());
            }
        }
        assertEquals(java.util.EnumSet.allOf(
                io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.PowerRealization.class),
                realized);
    }

    @Test void everyExactArtifactExecutesAgainstItsIndependentTypedCleanJavaLoop() throws Throwable {
        for (var fixture : CpuScalarImmediateClampMatrixOracle.fixtures()) {
            Object expected = CpuScalarImmediateClampMatrixOracle.cleanJava(fixture);
            Object actual = CpuScalarImmediateClampMatrixOracle.generated(fixture);
            if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, fixture.id());
            else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, fixture.id());
            else if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, fixture.id());
            else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, fixture.id());
            else assertArrayEquals((long[]) expected, (long[]) actual, fixture.id());
        }
    }

    @Test void everyCompositionalFormExecutesAgainstItsResolvedLayoutOracle() throws Throwable {
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            assertTypedEquals(CpuScalarImmediateClampMatrixOracle.cleanJava(form),
                    CpuScalarImmediateClampMatrixOracle.generated(form), form.id());
        }
    }

    @Test void reachableSegmentAndOrderedMixedCarriersHaveTheirOwnDescriptorAndTypedExecution() throws Throwable {
        for (var form : CpuScalarImmediateClampMatrixOracle.forms().stream()
                .filter(value -> value.id().startsWith("CARRIER-")).toList()) {
            var fixture = form.fixture();
            var artifact = CpuScalarImmediateClampMatrixOracle.artifact(fixture);
            assertTrue(artifact.descriptor().contains("MemorySegment"), fixture.id());
            assertTypedEquals(CpuScalarImmediateClampMatrixOracle.cleanJava(fixture),
                    CpuScalarImmediateClampMatrixOracle.generated(fixture), fixture.id());
        }
    }

    @Test void sourceCapabilityRowsCloseExactlyAndAdjacentCasesFailBeforeGeneration() {
        var provider = new CpuCapabilityProvider();
        var numeric = List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64);
        var expectedPairs = new java.util.LinkedHashSet<String>();
        for (var kind : ScalarElementwiseKind.values()) {
            for (var type : DataType.values()) {
                boolean expected = type != DataType.BOOL
                        && (kind != ScalarElementwiseKind.CLAMP
                            && kind != ScalarElementwiseKind.DIV
                            && kind != ScalarElementwiseKind.POW || type.isFloating());
                assertEquals(expected, provider.supports(query(kind, type, type, type)), kind + " " + type);
                if (expected) expectedPairs.add(kind + "/" + type);
            }
        }
        assertEquals(expectedPairs, CpuScalarImmediateClampMatrixOracle.fixtures().stream()
                .map(fixture -> fixture.operation() + "/" + fixture.type())
                .collect(java.util.stream.Collectors.toCollection(java.util.LinkedHashSet::new)),
                "every and only source-admitted operation/type pair has semantic fixtures");
        for (var kind : ScalarElementwiseKind.values()) {
            for (var input : numeric) for (var output : numeric) if (input != output) {
                assertFalse(provider.supports(query(kind, input, output, output)),
                        kind + " mixed input/output " + input + '/' + output);
                assertFalse(provider.supports(query(kind, input, input, output)),
                        kind + " mixed immediate " + input + '/' + output);
            }
        }
        for (var type : List.of(DataType.INT32, DataType.INT64)) {
            assertFalse(provider.supports(query(ScalarElementwiseKind.DIV, type, type, type)));
            assertFalse(provider.supports(query(ScalarElementwiseKind.POW, type, type, type)));
            assertFalse(provider.supports(query(ScalarElementwiseKind.CLAMP, type, type, type)));
        }
    }

    @Test void nonFiniteImmediatesRemainExplicitlyOutsideTheFiniteProjectionBoundary() {
        for (var fixture : CpuScalarImmediateClampMatrixOracle.fixtures()) {
            assertTrue(isFinite(fixture.immediate()), fixture.id());
            if (fixture.upper() != null) assertTrue(isFinite(fixture.upper()), fixture.id());
        }
        var base = base(DataType.FLOAT32);
        var admittedButExcluded = new CpuScalarImmediateClampMatrixOracle.Fixture(
                "OUTSIDE-FINITE-ADD-FLOAT32-NAN", base.operation(), base.type(), base.category(),
                ScalarValue.float32(Float.NaN), null, base.carriers(), base.preference(),
                base.parallelism());
        assertTrue(CpuScalarImmediateClampMatrixOracle.capability(admittedButExcluded),
                "non-finite immediates do not create another source/emitter code-shape category");
        assertFalse(CpuScalarImmediateClampMatrixOracle.fixtures().stream()
                .anyMatch(fixture -> fixture.id().equals(admittedButExcluded.id())));
    }

    @Test void sourceDerivedFormsAreDeterministicUniqueAndCandidateTruthIsNotSelectionTruth() {
        var first = CpuScalarImmediateClampMatrixOracle.forms();
        var second = CpuScalarImmediateClampMatrixOracle.forms();
        assertEquals(first.stream().map(CpuScalarImmediateClampMatrixOracle.Form::id).toList(),
                second.stream().map(CpuScalarImmediateClampMatrixOracle.Form::id).toList());
        assertEquals(first.size(), first.stream().map(CpuScalarImmediateClampMatrixOracle.Form::id)
                .collect(java.util.stream.Collectors.toSet()).size());
        var candidateForms = first.stream().filter(form -> form.id().startsWith("MATERIALIZATION-")).toList();
        assertEquals(4, candidateForms.size());
        for (var form : candidateForms) {
            var actual = CpuScalarImmediateClampMatrixOracle.formArtifact(form);
            assertEquals(1, actual.materializationCandidates(), form.id());
            assertEquals(0, actual.materializationSelected(), form.id());
        }
        assertEquals(4, candidateForms.stream().map(CpuScalarImmediateClampMatrixOracle::formArtifact)
                .mapToInt(CpuScalarImmediateClampMatrixOracle.FormArtifact::materializationCandidates).sum());
        var bfloat = base(DataType.BFLOAT16);
        var matrix = Shape.of(2, 3);
        var enabled = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);
        var plan = new CpuPartitionPreparer().analyze(CpuScalarImmediateClampMatrixOracle.contextFor(
                bfloat, matrix, LayoutDescriptor.of(matrix, new long[] {5, 2}, 0, true), matrix,
                LayoutDescriptor.contiguous(matrix), enabled)).plan();
        assertEquals(0, plan.representationDecisions().stream()
                .filter(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant.class::isInstance)
                .map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant.class::cast)
                .filter(candidate -> !candidate.identity().materializations().isEmpty()).count());
        assertTrue(plan.materializations().isEmpty());
    }

    @Test void vectorRangeWitnessUsesArbitraryBoundsAndScalarTail() throws Throwable {
        var fixture = base(DataType.FLOAT32);
        int start = FloatVector.SPECIES_PREFERRED.length();
        float[] full = (float[]) CpuScalarImmediateClampMatrixOracle.cleanJava(fixture);
        float[] actual = (float[]) CpuScalarImmediateClampMatrixOracle.generated(fixture, start, 31L);
        float[] expected = new float[32];
        System.arraycopy(full, start, expected, start, 31 - start);
        assertArrayEquals(expected, actual);
    }

    @Test void scalarImmediateAccessWitnessesSelectEverySourceNormalizedRegime() {
        var base = base();
        record Witness(String id, Shape inputShape, LayoutDescriptor inputLayout, Shape outputShape,
                CpuAccessPlan.Regime regime, ExecutionStrategy strategy) { }
        Shape matrix = Shape.of(2, 3);
        var witnesses = List.of(
                new Witness("dense", matrix, LayoutDescriptor.contiguous(matrix), matrix,
                        CpuAccessPlan.Regime.DENSE_LINEAR, ExecutionStrategy.PARALLEL_VECTOR),
                new Witness("scalar", matrix, LayoutDescriptor.of(matrix,
                        new long[] {0, 0}, 0, true), matrix,
                        CpuAccessPlan.Regime.SCALAR_ALL_ZERO, ExecutionStrategy.PARALLEL_VECTOR),
                new Witness("last-axis-bias", matrix, LayoutDescriptor.of(matrix, new long[] {0, 1}, 0, true), matrix,
                        CpuAccessPlan.Regime.LAST_AXIS_BIAS, ExecutionStrategy.PARALLEL_SCALAR),
                new Witness("block-outer", matrix, LayoutDescriptor.of(matrix, new long[] {5, 1}, 0, true), matrix,
                        CpuAccessPlan.Regime.BLOCK_OUTER, ExecutionStrategy.PARALLEL_SCALAR),
                new Witness("general-odometer", matrix, LayoutDescriptor.of(matrix, new long[] {5, 2}, 0, true), matrix,
                        CpuAccessPlan.Regime.GENERAL_ODOMETER, ExecutionStrategy.PARALLEL_SCALAR));
        for (var witness : witnesses) {
            var context = CpuScalarImmediateClampMatrixOracle.contextFor(base, witness.inputShape(),
                    witness.inputLayout(), witness.outputShape(), LayoutDescriptor.contiguous(witness.outputShape()));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var unit = plan.units().getFirst();
            assertEquals(witness.regime(), unit.accessBindings().getFirst().plan().regime(), witness.id());
            assertEquals(witness.strategy(), unit.executionStrategy(), witness.id());
            assertTrue(plan.materializations().isEmpty(), witness.id() + " direct candidate is not selected");
        }
    }

    @Test void rankZeroAndZeroWorkRemainMateriallyDistinctDirectScalarWitnesses() {
        var base = base();
        for (var shape : List.of(Shape.scalar(), Shape.of(0, 3))) {
            var context = CpuScalarImmediateClampMatrixOracle.contextFor(base, shape,
                    LayoutDescriptor.contiguous(shape), shape, LayoutDescriptor.contiguous(shape));
            var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
            assertEquals(shape.rank() == 0 ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                    : CpuAccessPlan.Regime.DENSE_LINEAR,
                    unit.accessBindings().getFirst().plan().regime());
            assertEquals(CpuAccessPlan.Regime.DENSE_LINEAR,
                    unit.accessBindings().getLast().plan().regime());
            assertEquals(shape.rank() == 0 ? 1L : 0L, unit.accessBindings().getFirst().elementCount());
            assertEquals(ExecutionStrategy.SCALAR, unit.executionStrategy());
            assertTrue(unit.runtimeFacts().materialization().isEmpty());
        }
    }

    @Test void extentChangesDoNotAlterTheProvedRegimeLoopCarrierOrStrategyFacts() {
        var base = base();
        for (int mode = 0; mode < 5; mode++) {
            // Both witnesses clear the preferred-species full-lane threshold, so this isolates
            // extent variation from the source's independently tested vector-admission boundary.
            var small = regimeContext(base, Shape.of(8, 8), mode);
            var large = regimeContext(base, Shape.of(9, 9), mode);
            var left = new CpuPartitionPreparer().analyze(small).plan().units().getFirst();
            var right = new CpuPartitionPreparer().analyze(large).plan().units().getFirst();
            assertEquals(left.accessBindings().getFirst().plan(), right.accessBindings().getFirst().plan());
            assertEquals(left.executionStrategy(), right.executionStrategy());
            assertEquals(left.carrierPattern(), right.carrierPattern());
            assertEquals(left.portablePlan().kernelIr().structuralKey(), right.portablePlan().kernelIr().structuralKey());
        }
    }

    @Test void eachAdmittedRegimeExecutesItsOwnTypedAddressingOracleAcrossCarrierForms() throws Throwable {
        var base = base(); Shape matrix = Shape.of(2, 3);
        record Case(String id, LayoutDescriptor layout, float[] source,
                java.util.function.IntUnaryOperator address, List<CpuKernelSpecialization.CarrierAccess> carriers) { }
        var array = CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        var cases = List.of(
                new Case("dense", LayoutDescriptor.contiguous(matrix), new float[] {0, 1, 2, 3, 4, 5},
                        ordinal -> ordinal, List.of(array, array)),
                new Case("scalar", LayoutDescriptor.of(matrix, new long[] {0, 0}, 0, true), new float[] {7},
                        ordinal -> 0, List.of(segment, segment)),
                new Case("last-axis-bias", LayoutDescriptor.of(matrix, new long[] {0, 1}, 0, true),
                        new float[] {10, 20, 30}, ordinal -> ordinal % 3, List.of(array, segment)),
                new Case("block-outer", LayoutDescriptor.of(matrix, new long[] {5, 1}, 0, true),
                        new float[] {0, 1, 2, 0, 0, 10, 11, 12}, ordinal -> (ordinal / 3) * 5 + ordinal % 3,
                        List.of(segment, array)),
                new Case("general-odometer", LayoutDescriptor.of(matrix, new long[] {5, 2}, 0, true),
                        new float[] {0, 1, 2, 3, 4, 10, 6, 11, 8, 12}, ordinal -> (ordinal / 3) * 5 + (ordinal % 3) * 2,
                        List.of(array, segment)));
        for (var testCase : cases) {
            var fixture = new CpuScalarImmediateClampMatrixOracle.Fixture(base.id() + '-' + testCase.id(),
                    base.operation(), base.type(), base.category(), base.immediate(), base.upper(),
                    testCase.carriers(), base.preference(), base.parallelism());
            var context = CpuScalarImmediateClampMatrixOracle.contextFor(fixture, matrix, testCase.layout(),
                    matrix, LayoutDescriptor.contiguous(matrix));
            var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
            byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            float[] output = new float[6];
            Object input = testCase.carriers().getFirst() == segment ? MemorySegment.ofArray(testCase.source()) : testCase.source();
            Object result = testCase.carriers().getLast() == segment ? MemorySegment.ofArray(output) : output;
            MethodHandle handle = new CpuClassFileKernelGenerator().defineClassBytes(
                    route.specialization(), bytes).entryPoint();
            handle.invokeWithArguments(input, result, CpuScalarImmediateClampMatrixOracle.geometryFor(context), 0L, 6L);
            float[] expected = new float[6];
            for (int ordinal = 0; ordinal < expected.length; ordinal++)
                expected[ordinal] = testCase.source()[testCase.address().applyAsInt(ordinal)] + .5f;
            assertArrayEquals(expected, output, testCase.id());
        }
    }

    @Test void computeArtifactIsSharedAcrossCallerOrchestrationButNotAcrossComputeKinds() {
        var base = base();
        var scalar = fixture(base, ComputePreference.SCALAR, 1);
        var parallelScalar = fixture(base, ComputePreference.SCALAR, 4);
        var vector = fixture(base, ComputePreference.VECTOR_IF_ELIGIBLE, 1);
        var parallelVector = fixture(base, ComputePreference.VECTOR_IF_ELIGIBLE, 4);
        assertSelected(scalar, ExecutionStrategy.SCALAR);
        assertSelected(parallelScalar, ExecutionStrategy.PARALLEL_SCALAR);
        assertSelected(vector, ExecutionStrategy.VECTOR);
        assertSelected(parallelVector, ExecutionStrategy.PARALLEL_VECTOR);
        var scalarArtifact = CpuScalarImmediateClampMatrixOracle.artifact(scalar);
        var parallelScalarArtifact = CpuScalarImmediateClampMatrixOracle.artifact(parallelScalar);
        var vectorArtifact = CpuScalarImmediateClampMatrixOracle.artifact(vector);
        var parallelVectorArtifact = CpuScalarImmediateClampMatrixOracle.artifact(parallelVector);
        assertArrayEquals(scalarArtifact.bytes(), parallelScalarArtifact.bytes());
        assertEquals(scalarArtifact.hash(), parallelScalarArtifact.hash());
        assertArrayEquals(vectorArtifact.bytes(), parallelVectorArtifact.bytes());
        assertEquals(vectorArtifact.hash(), parallelVectorArtifact.hash());
        assertNotEquals(scalarArtifact.hash(), vectorArtifact.hash());
    }

    private static CpuScalarImmediateClampMatrixOracle.Fixture base() {
        return base(DataType.FLOAT32);
    }
    private static CpuScalarImmediateClampMatrixOracle.Fixture base(DataType type) {
        return CpuScalarImmediateClampMatrixOracle.fixtures().stream()
                .filter(f -> f.id().equals("ADD-" + type + "-OTHER")).findFirst().orElseThrow();
    }
    private static CpuScalarImmediateClampMatrixOracle.Fixture fixture(String id) {
        return CpuScalarImmediateClampMatrixOracle.fixtures().stream()
                .filter(value -> value.id().equals(id)).findFirst().orElseThrow();
    }

    private static OperationCapabilityQuery query(ScalarElementwiseKind kind, DataType inputType,
            DataType outputType, DataType immediateType) {
        Shape shape = Shape.of(8);
        var input = new TensorDescriptor(inputType, shape,
                java.util.Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var output = new TensorDescriptor(outputType, shape,
                java.util.Optional.of(LayoutDescriptor.contiguous(shape)), false);
        DataType attributeType = immediateType == DataType.BOOL ? DataType.FLOAT32 : immediateType;
        Operation operation = kind == ScalarElementwiseKind.CLAMP
                ? new Operation(kind, new ClampRangeAttrs(scalar(attributeType, -1), scalar(attributeType, 1)))
                : new Operation(kind, new ScalarValueAttrs(scalar(immediateType, 1)));
        return new OperationCapabilityQuery(operation, List.of(input), List.of(output));
    }

    private static ScalarValue scalar(DataType type, int value) {
        return switch (type) {
            case BFLOAT16 -> ScalarValue.bfloat16(value);
            case FLOAT32 -> ScalarValue.float32(value);
            case FLOAT64 -> ScalarValue.float64(value);
            case INT32 -> ScalarValue.int32(value);
            case INT64 -> ScalarValue.int64(value);
            case BOOL -> ScalarValue.bool(value != 0);
        };
    }

    private static boolean isFinite(ScalarValue value) {
        return switch (value.dataType()) {
            case BFLOAT16 -> Float.isFinite(
                    CpuScalarImmediateClampEquivalenceOracle.floatValue(value.bfloat16Bits()));
            case FLOAT32 -> Float.isFinite(value.float32Value());
            case FLOAT64 -> Double.isFinite(value.float64Value());
            case INT32, INT64 -> true;
            case BOOL -> throw new AssertionError(value);
        };
    }

    private static void assertTypedEquals(Object expected, Object actual, String message) {
        if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else assertArrayEquals((long[]) expected, (long[]) actual, message);
    }

    private static CpuScalarImmediateClampMatrixOracle.Fixture fixture(
            CpuScalarImmediateClampMatrixOracle.Fixture base, ComputePreference preference, int parallelism) {
        return new CpuScalarImmediateClampMatrixOracle.Fixture(base.id() + '-' + preference + '-' + parallelism,
                base.operation(), base.type(), base.category(), base.immediate(), base.upper(), base.carriers(),
                preference, parallelism);
    }

    private static void assertSelected(CpuScalarImmediateClampMatrixOracle.Fixture fixture,
            ExecutionStrategy expected) {
        assertEquals(expected, new CpuPartitionPreparer().analyze(
                CpuScalarImmediateClampMatrixOracle.contextFor(fixture)).plan().units().getFirst()
                .executionStrategy(), fixture.id());
    }

    private static io.github.pho001.synaptik.prepare.analysis.PrepareContext<
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> regimeContext(
            CpuScalarImmediateClampMatrixOracle.Fixture base, Shape shape, int mode) {
        long columns = shape.toLongArray()[1];
        LayoutDescriptor layout = switch (mode) {
            case 0 -> LayoutDescriptor.contiguous(shape);
            case 1 -> LayoutDescriptor.of(shape, new long[] {0, 0}, 0, true);
            case 2 -> LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true);
            case 3 -> LayoutDescriptor.of(shape, new long[] {columns + 2, 1}, 0, true);
            case 4 -> LayoutDescriptor.of(shape, new long[] {columns + 2, 2}, 0, true);
            default -> throw new AssertionError(mode);
        };
        return CpuScalarImmediateClampMatrixOracle.contextFor(base, shape, layout, shape,
                LayoutDescriptor.contiguous(shape));
    }


}
