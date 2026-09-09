package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

/**
 * Bounded Class-File hygiene for the ordinary pointwise and ordered CAST checkpoint owners.
 *
 * <p>This is intentionally a narrow paired-execution and hygiene check, not a general bytecode
 * equivalence claim. The paired projection retains the operation's ordered carrier reads and
 * writes, primitive-conversion order, permitted numerical invocations, and the direct-loop
 * branch topology; it excludes compiler-specific local, label, and geometry-plumbing detail.
 * Vector-selected entries retain evidence of their live vector realization rather than being
 * described as structurally identical to the scalar clean-Java counterpart. All owned semantic
 * bodies have independently authored typed counterparts; this test does not itself promote a
 * structural or performance disposition.</p>
 */
class CpuOrdinaryPointwiseCastStructuralOracleTest {
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";

    @Test void exactOwnedRowsHaveActiveSelectedEntryHygieneAndRemainTruthfullyPartial()
            throws Throwable {
        new CpuGeneratedCoverageCheckpointTest()
                .exactCombinationInventoryReproducesEveryCanonicalFixtureExecution();
        List<String[]> owned = ownedRows();
        long ordinary = owned.stream().filter(row -> !row[2].equals("CAST")).count();
        long casts = owned.stream().filter(row -> row[2].equals("CAST")).count();
        assertEquals(665L, ordinary, "ordinary pointwise matrix owners");
        assertEquals(180L, casts, "ordered CAST matrix owners");
        assertEquals(845, owned.size(), "CPU 0009B generated owner denominator");

        int inspected = 0;
        for (String[] row : owned) {
            byte[] bytes = CpuGeneratedCoverageEvidenceRegistry.classBytes(row[0]);
            List<String> tokens = CpuScalarImmediateClampMatrixStructuralTest.normalize(bytes);
            assertFalse(tokens.isEmpty(), row[0] + " has no selected Code");
            String body = String.join("\n", tokens);
            for (String forbidden : List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY",
                    "INVOKEDYNAMIC", "java/lang/reflect", "java/lang/invoke/MethodHandle",
                    "java/util/Map", "java/lang/String", "Boolean.valueOf", "Integer.valueOf",
                    "Long.valueOf", "Float.valueOf", "Double.valueOf")) {
                assertFalse(body.contains(forbidden), row[0] + " forbidden selected-entry mechanism: " + forbidden);
            }
            // Existing selected vector realizations may call this narrow, numerical chunk helper.
            // No other Synaptik call is allowed in the generated entry.
            String withoutPermittedVectorMath = body.replaceAll(
                    "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath", "");
            assertFalse(withoutPermittedVectorMath.contains("io/github/pho001/synaptik"),
                    row[0] + " non-allowlisted Synaptik call in selected entry");
            assertTrue(tokens.stream().anyMatch(token -> token.startsWith("GOTO|") || token.startsWith("IF")),
                    row[0] + " lacks an active range loop or branch");
            assertTrue(tokens.stream().anyMatch(token -> token.contains("STORE")
                    || token.contains("MemorySegment.set")), row[0] + " lacks a direct selected output store");
            inspected++;
        }
        assertEquals(845, inspected, "the hygiene path must inspect every exact owned entry");
    }

    @Test void independentlyJavacCompiledTypedCounterpartsMatchEveryExactSelectedAbiAndHaveLiveRangeLoops()
            throws Throwable {
        new CpuGeneratedCoverageCheckpointTest()
                .exactCombinationInventoryReproducesEveryCanonicalFixtureExecution();
        List<String[]> inventory = ownedRows();
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = inventory.stream().map(row ->
                new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(row[0], row[2], row[18], row[9], row[10], row[12],
                        boundaryType(row[4]), boundaryOutputType(row[4])))
                .toList();
        var first = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        var second = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256")
                .digest(first.bytes())), java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256")
                .digest(second.bytes())), "deterministic javac clean-Java artifact");

        Map<String, MethodModel> clean = methods(first.bytes());
        assertEquals(845, clean.size(), "one distinct typed clean entry per owned inventory row");
        int checked = 0;
        for (CpuOrdinaryPointwiseCastCleanJavaOracle.Row row : rows) {
            String methodName = first.methods().get(row.owner());
            MethodModel method = clean.get(methodName);
            assertTrue(method != null, row.owner() + " missing clean selected method");
            // This compares an independently compiled, typed ABI to the execution-produced ABI.
            // Owner/order locate the fixture only; operation, hash, form id, and normalized body
            // are deliberately not inputs to this comparison.
            assertEquals(row.descriptor(), method.methodType().stringValue(), row.owner() + " typed ABI");
            assertTrue(method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                    .map(Instruction.class::cast).anyMatch(instruction -> instruction.opcode().name().startsWith("IF")
                            || instruction.opcode().name().startsWith("GOTO")),
                    row.owner() + " clean counterpart has a live range loop");
            checked++;
        }
        assertEquals(665L, rows.stream().filter(row -> !row.operation().equals("CAST")).count());
        assertEquals(180L, rows.stream().filter(row -> row.operation().equals("CAST")).count());
        assertEquals(845, checked);
    }

    @Test void allOrdinarySemanticOwnersAreImplementedWithoutScaffolds() {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates();
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> ordinary = candidates.stream()
                .filter(candidate -> !candidate.fixture().opcode().name().equals("CAST")).toList();
        assertEquals(665, ordinary.size(), "ordinary source-candidate denominator");
        long arithmeticComparison = ordinary.stream().filter(candidate -> semanticBinary(
                candidate.fixture().opcode().name())).count();
        long unaryClassification = ordinary.stream().filter(candidate -> semanticUnaryOrClassification(
                candidate.fixture().opcode().name())).count();
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> logicalSelection = ordinary.stream()
                .filter(candidate -> semanticLogicalOrSelection(candidate.fixture().opcode().name())).toList();
        assertEquals(305, arithmeticComparison, "arithmetic/comparison semantic owners");
        assertEquals(330, unaryClassification, "unary/classification semantic owners");
        assertEquals(Map.of("LOGICAL_AND", 5L, "LOGICAL_NOT", 5L, "LOGICAL_OR", 5L, "WHERE", 15L),
                countsByOpcode(logicalSelection), "logical/WHERE source-candidate owners");
        assertEquals(30, logicalSelection.size(), "final ordinary semantic owners");
        assertEquals(665, arithmeticComparison + unaryClassification + logicalSelection.size(),
                "zero ordinary semantic scaffolds remain");
    }

    @Test void pairedAbiComparatorRejectsAChangedProjection() {
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = List.of(
                new CpuOrdinaryPointwiseCastCleanJavaOracle.Row("negative", "ADD", "([F[F[F[JJJ)V",
                        "[FLOAT_ARRAY, FLOAT_ARRAY]", "[DENSE_LINEAR, DENSE_LINEAR]", "scalar",
                        "FLOAT32", "FLOAT32"));
        var compiled = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        MethodModel entry = methods(compiled.bytes()).get("entry0");
        assertTrue(entry != null);
        assertFalse("([D[D[D[JJJ)V".equals(entry.methodType().stringValue()),
                "a changed typed ABI must not satisfy the paired comparator");
    }

    /**
     * Compares the direct hot-loop projection of every selected generated entry with its
     * independently compiled clean-Java counterpart.  The projection deliberately discards
     * local-slot allocation, constant-pool numbering, labels, and javac's checkcasts/layout
     * setup: those are compilation artifacts rather than the specialized loop algorithm. This
     * normalization is deliberately general to separately compiled artifacts; it does not treat
     * an incidental compiler-plumbing difference as a change in the selected algorithm. It
     * retains the operation-selected entry, ordered carrier read/write roles, primitive
     * conversions, permitted numerical invocations, and direct-loop branch topology. Scalar
     * realizations compare numerical calls and the live range-loop backedge. A selected Vector
     * API realization is deliberately not called structurally identical to that scalar clean-Java
     * oracle: the test instead requires its live vector invocation while retaining the shared
     * carrier/conversion projection. The adjacent negative controls reject changed carrier,
     * conversion, invocation, and branch facts. This is a bounded structural oracle, not CFG
     * equivalence.
     */
    @Test void pairedHotLoopFactsMatchEverySelectedGeneratedAndCleanJavaEntry() throws Throwable {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates();
        assertEquals(845, candidates.size(), "exact selected source-derived owner count");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = candidates.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return row(candidate, route);
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int compared = 0;
        for (int index = 0; index < candidates.size(); index++) {
            var candidate = candidates.get(index);
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            byte[] generated = new CpuClassFileKernelGenerator().generateClassBytes(
                    route.specialization(), route.kernelIr());
            MethodModel cleanMethod = cleanMethods.get(clean.methods().get(candidate.id()));
            assertTrue(cleanMethod != null, candidate.id() + " clean structural counterpart");
            assertPairedHotLoopFacts(candidate.id(), facts(generated), facts(cleanMethod),
                    candidate.fixture().inputTypes().size(), candidate.fixture().opcode().name().equals("CAST")
                            && candidate.fixture().inputTypes().getFirst() != candidate.fixture().outputType(),
                    route.specialization().executionStrategy().compute()
                            == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                                    .ExecutionStrategy.Compute.VECTOR,
                    candidate.fixture().inputTypes().contains(DataType.INT64),
                    candidate.fixture().outputType() == DataType.INT64);
            compared++;
        }
        assertEquals(845, compared, "every exact selected owner receives a paired structural comparison");
    }

    @Test void pairedHotLoopComparatorRejectsIndependentCarrierConversionInvokeAndBranchDrift() {
        HotLoopFacts baseline = new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"),
                List.of(), List.of("F2D"), List.of("Math.sin"), new BranchTopology(1, true));
        assertPairedHotLoopFacts("control", baseline, baseline, 2, false, false, false, false);
        for (HotLoopFacts changed : List.of(
                new HotLoopFacts(List.of("FALOAD", "DALOAD"), List.of("FASTORE"), List.of(), List.of("F2D"), List.of("Math.sin"), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("DASTORE"), List.of(), List.of("F2D"), List.of("Math.sin"), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"), List.of(), List.of("D2F"), List.of("Math.sin"), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"), List.of("SEGMENT_GET:OfFloat:F"), List.of("F2D"), List.of("Math.sin"), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"), List.of(), List.of("F2D"), List.of("Math.cos"), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"), List.of(), List.of("F2D"), List.of("Math.sin"), new BranchTopology(0, true)),
                new HotLoopFacts(List.of("FALOAD", "FALOAD"), List.of("FASTORE"), List.of(), List.of("F2D"), List.of("Math.sin"), new BranchTopology(1, false)))) {
            assertThrows(AssertionError.class,
                    () -> assertPairedHotLoopFacts("mutated", baseline, changed, 2, false, false, false, false));
        }
    }

    @Test void pairedHotLoopComparatorRejectsTypedSegmentLayoutAndRoleOrderDrift() {
        HotLoopFacts baseline = new HotLoopFacts(
                List.of("SEGMENT_GET:OfFloat:F", "SEGMENT_GET:OfInt:I"),
                List.of("SEGMENT_SET:OfFloat:F"),
                List.of("SEGMENT_GET:OfFloat:F", "SEGMENT_GET:OfInt:I", "SEGMENT_SET:OfFloat:F"),
                List.of(), List.of(), new BranchTopology(1, true));
        assertPairedHotLoopFacts("segment-control", baseline, baseline, 0, false, false, false, false);
        for (HotLoopFacts changed : List.of(
                new HotLoopFacts(List.of("SEGMENT_GET:OfDouble:D", "SEGMENT_GET:OfInt:I"),
                        List.of("SEGMENT_SET:OfFloat:F"),
                        List.of("SEGMENT_GET:OfDouble:D", "SEGMENT_GET:OfInt:I", "SEGMENT_SET:OfFloat:F"),
                        List.of(), List.of(), new BranchTopology(1, true)),
                new HotLoopFacts(List.of("SEGMENT_GET:OfInt:I", "SEGMENT_GET:OfFloat:F"),
                        List.of("SEGMENT_SET:OfFloat:F"),
                        List.of("SEGMENT_GET:OfInt:I", "SEGMENT_GET:OfFloat:F", "SEGMENT_SET:OfFloat:F"),
                        List.of(), List.of(), new BranchTopology(1, true))))
            assertThrows(AssertionError.class,
                    () -> assertPairedHotLoopFacts("mutated-segment", baseline, changed, 0, false, false, false, false));
    }

    @Test void everyCastOwnerExecutesItsSelectedGeneratedAndIndependentTypedCleanEntry() throws Throwable {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates().stream()
                        .filter(candidate -> candidate.fixture().opcode().name().equals("CAST")).toList();
        assertEquals(180, candidates.size(), "36 ordered pairs x five selected candidate variants");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = candidates.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(candidate.id(), "CAST",
                    entryDescriptor(route), candidate.inputs().toString(),
                    candidate.layoutAxis(), candidate.requestedStrategy(), candidate.fixture().inputTypes().getFirst().name(),
                    candidate.fixture().outputType().name());
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (int index = 0; index < candidates.size(); index++) {
            var candidate = candidates.get(index); var route = new CpuPartitionPreparer().analyze(context(candidate))
                    .plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method cleanEntry = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            cleanEntry.setAccessible(true);
            // Each call supplies the same cold, range-positioned geometry production supplies.
            // Parallel selection is intentionally tested as disjoint entry ranges: scheduling is
            // CpuPreparedExecutable's caller-side responsibility, not generated-class behavior.
            for (long[] range : List.of(new long[] {0, 8}, new long[] {2, 2}, new long[] {2, 6}, new long[] {6, 8}))
                executePair(candidate, generated, cleanEntry, range[0], range[1]);
            executed++;
        }
        assertEquals(180, executed, "every selected CAST owner executed all range contracts");
    }

    @Test void sourceDerivedCarrierRoleWitnessesExecuteBothMixedDirectionsAndEveryLegalRolePosition()
            throws Throwable {
        // Enumerating source-defined input/output positions, rather than synthesizing carrier
        // forms, covers both array-to-segment and segment-to-array directions as real witnesses.
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> witnesses =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCarrierRoleWitnesses();
        int expected = CpuGeneratedDirectEvidenceClosureTest.generalPointwiseFixtures().stream()
                .mapToInt(fixture -> 1 << (fixture.inputTypes().size() + 1)).sum();
        assertEquals(expected, witnesses.size(), "source-derived carrier-role witness accounting");
        assertTrue(witnesses.stream().anyMatch(candidate -> candidate.inputs().getFirst()
                != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && candidate.inputs().getLast() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                "array-to-segment is an actual ordered carrier witness");
        assertTrue(witnesses.stream().anyMatch(candidate -> candidate.inputs().getFirst()
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && candidate.inputs().getLast() != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                "segment-to-array is an actual ordered carrier witness");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = witnesses.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            assertEquals(candidate.inputs(), route.specialization().carrierPattern(), candidate.id());
            return row(candidate, route);
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (var candidate : witnesses) {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method cleanEntry = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            cleanEntry.setAccessible(true);
            executePair(candidate, generated, cleanEntry, 2, 6);
            executed++;
        }
        assertEquals(expected, executed, "every source-derived ordered carrier role pair executed");
    }

    @Test void safeSourceDerivedInputAndOutputAliasesExecuteWithoutAssumingUnsafeOverlap() throws Throwable {
        // These witnesses reuse only carrier positions whose source descriptors and typed carrier
        // access already make the alias legal. They exercise existing forms; they do not add a
        // partial-overlap or mixed-width alias contract to the generator.
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> base =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates().stream()
                        .filter(candidate -> candidate.id().endsWith("heap-contiguous-scalar")).toList();
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> inputAliases = base.stream()
                .filter(candidate -> candidate.fixture().inputTypes().size() >= 2)
                .filter(candidate -> candidate.fixture().inputTypes().get(0) == candidate.fixture().inputTypes().get(1))
                .filter(candidate -> candidate.inputs().get(0) == candidate.inputs().get(1)).toList();
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> outputAliases = base.stream()
                .filter(candidate -> candidate.fixture().inputTypes().getFirst() == candidate.fixture().outputType())
                .filter(candidate -> candidate.inputs().getFirst() == candidate.inputs().getLast()).toList();
        assertFalse(inputAliases.isEmpty(), "source contracts admit input-input aliases");
        assertFalse(outputAliases.isEmpty(), "source contracts admit one-element input-output aliases");
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> aliases = new ArrayList<>();
        aliases.addAll(inputAliases); aliases.addAll(outputAliases);
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = aliases.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return row(candidate, route);
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (var candidate : inputAliases) {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method direct = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            direct.setAccessible(true);
            executePair(candidate, generated, direct, 2, 6, AliasTopology.INPUTS_0_1); executed++;
        }
        for (var candidate : outputAliases) {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method direct = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            direct.setAccessible(true);
            executePair(candidate, generated, direct, 2, 6, AliasTopology.INPUT_0_OUTPUT); executed++;
        }
        assertEquals(aliases.size(), executed, "all source-admitted safe alias witnesses execute");
        // Partial overlaps and mixed-width aliases are intentionally absent: their overwrite
        // semantics are not admitted by this pointwise generator contract.
    }

    @Test void ordinaryArithmeticAndComparisonOwnersExecuteSelectedGeneratedAndCleanEntries()
            throws Throwable {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates().stream()
                        .filter(candidate -> semanticBinary(candidate.fixture().opcode().name())).toList();
        assertEquals(305, candidates.size(), "exact ordinary arithmetic/comparison semantic slice");
        assertEquals(Map.ofEntries(Map.entry("ADD", 25L), Map.entry("SUB", 25L), Map.entry("MUL", 25L),
                Map.entry("DIV", 15L), Map.entry("POW", 15L), Map.entry("MIN", 25L), Map.entry("MAX", 25L),
                Map.entry("GREATER_THAN", 25L), Map.entry("GREATER_OR_EQUAL", 25L),
                Map.entry("LESS_THAN", 25L), Map.entry("LESS_OR_EQUAL", 25L), Map.entry("EQUAL", 25L),
                Map.entry("NOT_EQUAL", 25L)),
                countsByOpcode(candidates), "source-derived five-candidate ordinary ownership");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = candidates.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(candidate.id(), candidate.fixture().opcode().name(),
                    entryDescriptor(route), candidate.inputs().toString(), candidate.layoutAxis(), candidate.requestedStrategy(),
                    candidate.fixture().inputTypes().getFirst().name(), candidate.fixture().outputType().name());
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (int index = 0; index < candidates.size(); index++) {
            var candidate = candidates.get(index); var route = new CpuPartitionPreparer().analyze(context(candidate))
                    .plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method cleanEntry = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            cleanEntry.setAccessible(true);
            for (long[] range : List.of(new long[] {0, 8}, new long[] {2, 2}, new long[] {2, 6}, new long[] {6, 8}))
                executePair(candidate, generated, cleanEntry, range[0], range[1]);
            executed++;
        }
        assertEquals(305, executed, "each exact ordinary owner executes full, empty, subrange, and tail ranges");
    }

    @Test void unaryActivationAndClassificationOwnersExecuteSelectedGeneratedAndCleanEntries()
            throws Throwable {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates().stream()
                        .filter(candidate -> semanticUnaryOrClassification(candidate.fixture().opcode().name())).toList();
        assertEquals(330, candidates.size(), "exact remaining unary/classification semantic slice");
        Map<String, Long> expected = new java.util.TreeMap<>();
        for (String opcode : List.of("ABS", "NEG", "EXP", "EXPM1", "LOG", "LOG1P", "SQRT", "RECIPROCAL",
                "RSQRT", "FLOOR", "CEIL", "SIGN", "RELU", "SIGMOID", "TANH", "GELU_EXACT",
                "GELU_TANH_APPROXIMATION", "SILU", "ERF", "IS_FINITE", "IS_NAN", "IS_INF")) expected.put(opcode, 15L);
        assertEquals(Map.copyOf(expected), countsByOpcode(candidates),
                "each source-defined unary/classification opcode owns five candidates per floating type");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = candidates.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(candidate.id(), candidate.fixture().opcode().name(),
                    entryDescriptor(route), candidate.inputs().toString(), candidate.layoutAxis(), candidate.requestedStrategy(),
                    candidate.fixture().inputTypes().getFirst().name(), candidate.fixture().outputType().name());
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (var candidate : candidates) {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method cleanEntry = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            cleanEntry.setAccessible(true);
            for (long[] range : List.of(new long[] {0, 8}, new long[] {2, 2}, new long[] {2, 6}, new long[] {6, 8}))
                executePair(candidate, generated, cleanEntry, range[0], range[1]);
            executed++;
        }
        assertEquals(330, executed, "each unary/classification owner executes full, empty, subrange, and tail ranges");
    }

    private static boolean semanticBinary(String opcode) {
        return switch (opcode) {
            case "ADD", "SUB", "MUL", "DIV", "POW", "MIN", "MAX", "GREATER_THAN",
                    "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "EQUAL", "NOT_EQUAL" -> true;
            default -> false;
        };
    }

    private static boolean semanticUnaryOrClassification(String opcode) {
        return switch (opcode) {
            case "ABS", "NEG", "EXP", "EXPM1", "LOG", "LOG1P", "SQRT", "RECIPROCAL",
                    "RSQRT", "FLOOR", "CEIL", "SIGN", "RELU", "SIGMOID", "TANH",
                    "GELU_EXACT", "GELU_TANH_APPROXIMATION", "SILU", "ERF", "IS_FINITE",
                    "IS_NAN", "IS_INF" -> true;
            default -> false;
        };
    }

    private static boolean semanticLogicalOrSelection(String opcode) {
        return switch (opcode) {
            case "LOGICAL_AND", "LOGICAL_OR", "LOGICAL_NOT", "WHERE" -> true;
            default -> false;
        };
    }

    @Test void logicalAndSelectionOwnersExecuteSelectedGeneratedAndCleanEntries() throws Throwable {
        List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates =
                CpuGeneratedDirectEvidenceClosureTest.pointwiseCandidates().stream()
                        .filter(candidate -> semanticLogicalOrSelection(candidate.fixture().opcode().name())).toList();
        assertEquals(Map.of("LOGICAL_AND", 5L, "LOGICAL_NOT", 5L, "LOGICAL_OR", 5L, "WHERE", 15L),
                countsByOpcode(candidates), "source-derived logical and selection ownership");
        assertEquals(30, candidates.size(), "three logical forms and WHERE owners");
        List<CpuOrdinaryPointwiseCastCleanJavaOracle.Row> rows = candidates.stream().map(candidate -> {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            return new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(candidate.id(), candidate.fixture().opcode().name(),
                    entryDescriptor(route), candidate.inputs().toString(), candidate.layoutAxis(), candidate.requestedStrategy(),
                    candidate.fixture().inputTypes().getFirst().name(), candidate.fixture().outputType().name());
        }).toList();
        var clean = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(rows);
        Class<?> witness = new WitnessLoader().define(clean.bytes());
        int executed = 0;
        for (var candidate : candidates) {
            var route = new CpuPartitionPreparer().analyze(context(candidate)).plan().units().getFirst().portablePlan();
            MethodHandle generated = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                    new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Method cleanEntry = witness.getDeclaredMethod(clean.methods().get(candidate.id()), parameterTypes(candidate));
            cleanEntry.setAccessible(true);
            for (long[] range : List.of(new long[] {0, 8}, new long[] {2, 2}, new long[] {2, 6}, new long[] {6, 8}))
                executePair(candidate, generated, cleanEntry, range[0], range[1]);
            executed++;
        }
        assertEquals(30, executed, "each logical/WHERE owner executes full, empty, subrange, and tail ranges");
    }

    private static Map<String, Long> countsByOpcode(
            List<CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate> candidates) {
        Map<String, Long> counts = new java.util.TreeMap<>();
        for (var candidate : candidates)
            counts.merge(candidate.fixture().opcode().name(), 1L, Long::sum);
        return Map.copyOf(counts);
    }

    private static CpuOrdinaryPointwiseCastCleanJavaOracle.Row row(
            CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate candidate,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        return new CpuOrdinaryPointwiseCastCleanJavaOracle.Row(candidate.id(), candidate.fixture().opcode().name(),
                entryDescriptor(route), candidate.inputs().toString(), candidate.layoutAxis(),
                candidate.requestedStrategy(), candidate.fixture().inputTypes().getFirst().name(),
                candidate.fixture().outputType().name());
    }

    private static HotLoopFacts facts(byte[] bytes) {
        return facts(ClassFile.of().parse(bytes).methods().getFirst());
    }

    private static HotLoopFacts facts(MethodModel method) {
        List<CodeElement> elements = method.code().orElseThrow().elementStream().toList();
        var labels = new java.util.IdentityHashMap<Label, Integer>();
        int position = 0;
        for (CodeElement element : elements) {
            if (element instanceof LabelTarget target) labels.put(target.label(), position);
            if (element instanceof Instruction) position++;
        }
        List<InstructionFact> instructions = new ArrayList<>();
        for (CodeElement element : elements) if (element instanceof Instruction instruction) {
            int target = instruction instanceof BranchInstruction branch ? labels.get(branch.target()) : -1;
            instructions.add(new InstructionFact(instruction.opcode().name(), instruction instanceof InvokeInstruction call
                    ? call.owner().asInternalName() + '.' + call.name() + call.type()
                    : instruction instanceof FieldInstruction field
                            ? field.owner().asInternalName() + '.' + field.name() + ':' + field.type()
                            : "", target));
        }
        return factsFromInstructions(instructions);
    }

    private static HotLoopFacts factsFromTokens(List<String> tokens) {
        List<InstructionFact> instructions = new ArrayList<>();
        for (String token : tokens) {
            int separator = token.indexOf('|');
            String opcode = separator < 0 ? token : token.substring(0, separator);
            String detail = separator < 0 ? "" : token.substring(separator + 1);
            int marker = detail.indexOf("target=");
            int target = marker < 0 ? -1 : Integer.parseInt(detail.substring(marker + 7));
            instructions.add(new InstructionFact(opcode, detail.replaceFirst("^(invoke|field)=", ""), target));
        }
        return factsFromInstructions(instructions);
    }

    private static HotLoopFacts factsFromInstructions(List<InstructionFact> instructions) {
        List<String> loads = new ArrayList<>(), stores = new ArrayList<>(), segmentAccesses = new ArrayList<>(), conversions = new ArrayList<>(), invokes = new ArrayList<>();
        int backwardEdges = 0, forwardBranches = 0;
        for (int position = 0; position < instructions.size(); position++) {
            InstructionFact instruction = instructions.get(position);
            String opcode = instruction.opcode(); String detail = instruction.detail();
            if (opcode.endsWith("ALOAD")) loads.add(opcode);
            if (opcode.endsWith("ASTORE")) stores.add(opcode);
            if (detail.startsWith("java/lang/foreign/MemorySegment.get")) {
                String fact = "SEGMENT_GET:" + segmentLayout(detail) + ':' + methodReturn(detail);
                loads.add(fact); segmentAccesses.add(fact);
            }
            if (detail.startsWith("java/lang/foreign/MemorySegment.set")) {
                String fact = "SEGMENT_SET:" + segmentLayout(detail) + ':' + methodParameter(detail);
                stores.add(fact); segmentAccesses.add(fact);
            }
            if (opcode.matches("[IFLD]2[IFLD]")) conversions.add(opcode);
            if (opcode.endsWith("CMP")) conversions.add(opcode);
            if (detail.contains("Float.intBitsToFloat") || detail.contains("Float.floatToRawIntBits")
                    || detail.contains("Double.longBitsToDouble") || detail.contains("Double.doubleToRawLongBits")) conversions.add(invokeName(detail));
            if (detail.startsWith("java/lang/Math.") || detail.startsWith("java/lang/StrictMath.")
                    || detail.startsWith("jdk/incubator/vector/") || detail.contains("CpuVectorMath")) invokes.add(invokeName(detail));
            if ((opcode.startsWith("IF") || opcode.startsWith("GOTO")) && instruction.target() >= 0)
                if (instruction.target() < position) backwardEdges++; else forwardBranches++;
        }
        // Address geometry is a long[] ABI argument. javac and the direct emitter legitimately
        // reload its neighbouring base/stride slots a different number of times; that plumbing
        // is normalized only when carrier families are compared below.
        return new HotLoopFacts(List.copyOf(loads), List.copyOf(stores), List.copyOf(segmentAccesses),
                firstOccurrences(conversions), List.copyOf(invokes),
                new BranchTopology(backwardEdges, forwardBranches > 0));
    }

    private static String methodReturn(String detail) { return detail.substring(detail.lastIndexOf(')') + 1); }
    private static String segmentLayout(String detail) {
        int start = detail.indexOf("(Ljava/lang/foreign/ValueLayout$");
        int end = detail.indexOf(';', start);
        assertTrue(start >= 0 && end > start, "typed MemorySegment ValueLayout descriptor: " + detail);
        return detail.substring(start + "(Ljava/lang/foreign/ValueLayout$".length(), end);
    }
    private static String methodParameter(String detail) {
        int start = detail.indexOf('(') + 1, end = detail.lastIndexOf(')');
        return detail.substring(start, end);
    }

    private static List<String> collapseAdjacent(List<String> facts) {
        List<String> collapsed = new ArrayList<>();
        for (String fact : facts) if (collapsed.isEmpty() || !collapsed.getLast().equals(fact)) collapsed.add(fact);
        return List.copyOf(collapsed);
    }

    private static List<String> firstOccurrences(List<String> facts) {
        // Input/output carrier conversions can repeat once per loaded value in javac while the
        // direct emitter shares an equivalent typed conversion. Preserve the ordered conversion
        // kinds, which define this projection, without treating that local placement choice as a
        // distinct numerical algorithm.
        List<String> projected = new ArrayList<>();
        for (String fact : facts) if (!projected.contains(fact)) projected.add(fact);
        return List.copyOf(projected);
    }

    private static String invokeName(String token) {
        int at = token.indexOf("|invoke=");
        return (at < 0 ? token : token.substring(at + 8)).replaceAll("\\(.*", "");
    }

    private static void assertPairedHotLoopFacts(String owner, HotLoopFacts generated, HotLoopFacts clean,
            int inputCount, boolean requiresConversion, boolean vector, boolean hasInt64Input,
            boolean hasInt64Output) {
        assertEquals(carrierFamilies(generated.loads(), hasInt64Input), carrierFamilies(clean.loads(), hasInt64Input),
                owner + " ordered carrier read families");
        assertEquals(carrierFamilies(generated.stores(), hasInt64Output), carrierFamilies(clean.stores(), hasInt64Output),
                owner + " ordered carrier write families");
        assertEquals(generated.segmentAccesses(), clean.segmentAccesses(),
                owner + " ordered typed MemorySegment layout/access facts");
        assertEquals(generated.conversions(), clean.conversions(), owner + " primitive conversion order");
        if (vector) {
            assertTrue(generated.invokes().stream().anyMatch(invoke -> invoke.startsWith("jdk/incubator/vector/")
                    || invoke.startsWith("io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath")),
                    owner + " selected vector realization has a live vector invocation");
            assertFalse(clean.invokes().stream().anyMatch(invoke -> invoke.startsWith("jdk/incubator/vector/")
                    || invoke.startsWith("io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath")),
                    owner + " independently authored clean-Java oracle remains scalar");
        } else {
            assertEquals(generated.invokes(), clean.invokes(), owner + " numerical invocation shape");
            // This deliberately projects branch direction rather than literal CFG shape:
            // backward edges prove a counted-loop return. A forward branch is recorded without
            // attributing its origin; direct emission can add layout guards that javac does not,
            // so forward-branch multiplicity is not compared.
            assertTrue(generated.topology().backwardEdges() > 0 && clean.topology().backwardEdges() > 0,
                    owner + " live counted-loop backedge");
            assertEquals(generated.topology(), clean.topology(), owner + " branch direction topology");
        }
        assertTrue(generated.loads().size() >= inputCount, owner + " generated ordered input reads");
        assertFalse(generated.stores().isEmpty(), owner + " generated output write");
        if (requiresConversion && !vector) assertFalse(generated.conversions().isEmpty(),
                owner + " cross-type CAST must retain a conversion fact");
    }

    private static List<String> carrierFamilies(List<String> accesses, boolean hasInt64Role) {
        List<String> result = new ArrayList<>();
        for (String access : accesses) {
            if (access.startsWith("ALOAD") || access.startsWith("AASTORE")) continue; // javac carrier/layout reference plumbing only
            String family = access.startsWith("SEGMENT_") ? access : access.replace("ALOAD", "ARRAY")
                    .replace("ASTORE", "ARRAY");
            // The cold-bound long[] geometry is necessarily read in the hot loop to calculate
            // addresses.  It is not a data carrier, and must not masquerade as one for rows that
            // have no INT64 input or output.  INT64 rows retain LARRAY as a meaningful role.
            if (!hasInt64Role && family.equals("LARRAY")) continue;
            if (family.equals("ARRAY")) continue;
            if (result.isEmpty() || !result.getLast().equals(family)) result.add(family);
        }
        return List.copyOf(result);
    }

    private record InstructionFact(String opcode, String detail, int target) { }
    private record BranchTopology(int backwardEdges, boolean hasForwardBranch) { }
    private record HotLoopFacts(List<String> loads, List<String> stores, List<String> segmentAccesses,
            List<String> conversions, List<String> invokes, BranchTopology topology) { }

    @Test void castPairingNegativeControlsRejectConversionInvocationAndAbiDrift() throws Exception {
        var mismatch = CpuOrdinaryPointwiseCastCleanJavaOracle.compile(List.of(
                new CpuOrdinaryPointwiseCastCleanJavaOracle.Row("m", "CAST", "([D[I[JJJ)V", "", "", "",
                        "FLOAT64", "INT32")));
        MethodModel method = methods(mismatch.bytes()).get("entry0");
        assertFalse("([F[I[JJJ)V".equals(method.methodType().stringValue()), "ABI mismatch control");
        String tokens = String.join("\n", CpuScalarImmediateClampMatrixStructuralTest.normalize(mismatch.bytes()));
        assertFalse(tokens.contains("CastValueConversions"), "clean hot path cannot invoke Model conversion oracle");
        assertFalse(tokens.contains("INVOKESTATIC|io/github/pho001/synaptik"), "clean hot path cannot invoke Synaptik");
    }

    private static io.github.pho001.synaptik.prepare.analysis.PrepareContext<
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> context(
            CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate candidate) {
        return candidate.fixture().context(candidate.inputLayout(), candidate.inputs(), candidate.execution(),
                candidate.materializationPolicy());
    }

    private static String entryDescriptor(io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        return route.specialization().entryType().descriptorString();
    }

    private static void executePair(CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate candidate,
            MethodHandle generated, Method clean, long start, long end) throws Throwable {
        executePair(candidate, generated, clean, start, end, AliasTopology.NONE);
    }
    private static void executePair(CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate candidate,
            MethodHandle generated, Method clean, long start, long end, AliasTopology topology) throws Throwable {
        List<DataType> sources = candidate.fixture().inputTypes(); DataType target = candidate.fixture().outputType();
        try (Arena arena = candidate.inputs().stream().anyMatch(access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                ? Arena.ofShared() : Arena.ofConfined()) {
            List<Object> generatedInputs = new ArrayList<>(), cleanInputs = new ArrayList<>();
            for (int input = 0; input < sources.size(); input++) {
                generatedInputs.add(storage(sources.get(input), candidate.inputs().get(input), arena));
                cleanInputs.add(storage(sources.get(input), candidate.inputs().get(input), arena));
            }
            Object generatedOutput = storage(target, candidate.inputs().getLast(), arena);
            Object cleanOutput = storage(target, candidate.inputs().getLast(), arena);
            if (topology == AliasTopology.INPUTS_0_1) {
                generatedInputs.set(1, generatedInputs.getFirst()); cleanInputs.set(1, cleanInputs.getFirst());
            } else if (topology == AliasTopology.INPUT_0_OUTPUT) {
                generatedOutput = generatedInputs.getFirst(); cleanOutput = cleanInputs.getFirst();
            }
            long[] whole = CpuScalarImmediateClampMatrixOracle.geometryFor(context(candidate), 0, 8);
            for (int ordinal = 0; ordinal < 8; ordinal++) {
                for (int input = 0; input < sources.size(); input++) {
                    long address = whole[2 + input] + ordinal * whole[5 + input];
                    long value = topology == AliasTopology.INPUTS_0_1 ? edge(sources.get(input), ordinal)
                            : input == 0 ? edge(sources.get(input), ordinal) : rightEdge(sources.get(input), ordinal);
                    put(generatedInputs.get(input), sources.get(input), address, value);
                    put(cleanInputs.get(input), sources.get(input), address, value);
                }
            }
            if (topology != AliasTopology.INPUT_0_OUTPUT) {
                fill(generatedOutput, target, sentinel(target)); fill(cleanOutput, target, sentinel(target));
            }
            long[] geometry = CpuScalarImmediateClampMatrixOracle.geometryFor(context(candidate), start, end);
            List<Object> generatedArguments = new ArrayList<>(generatedInputs);
            generatedArguments.add(generatedOutput); generatedArguments.add(geometry);
            generatedArguments.add(start); generatedArguments.add(end);
            generated.invokeWithArguments(generatedArguments);
            List<Object> cleanArguments = new ArrayList<>(cleanInputs);
            cleanArguments.add(cleanOutput); cleanArguments.add(geometry); cleanArguments.add(start); cleanArguments.add(end);
            clean.invoke(null, cleanArguments.toArray());
            for (int ordinal = 0; ordinal < 8; ordinal++) {
                long address = whole[2 + sources.size()] + ordinal * whole[5 + sources.size()];
                assertEquals(raw(generatedOutput, target, address), raw(cleanOutput, target, address),
                        candidate.id() + " " + start + ".." + end + " ordinal=" + ordinal);
            }
        }
    }

    private enum AliasTopology { NONE, INPUTS_0_1, INPUT_0_OUTPUT }

    private static Class<?>[] parameterTypes(CpuGeneratedDirectEvidenceClosureTest.PointwiseCandidate candidate) {
        List<Class<?>> types = new ArrayList<>();
        for (int input = 0; input < candidate.fixture().inputTypes().size(); input++)
            types.add(carrierType(candidate.fixture().inputTypes().get(input), candidate.inputs().get(input)));
        types.add(carrierType(candidate.fixture().outputType(), candidate.inputs().getLast()));
        types.add(long[].class); types.add(long.class); types.add(long.class);
        return types.toArray(Class<?>[]::new);
    }
    private static Class<?> carrierType(DataType type, CpuKernelSpecialization.CarrierAccess access) {
        return access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT ? MemorySegment.class : array(switch(type) {
            case FLOAT64 -> 'D'; case FLOAT32 -> 'F'; case BFLOAT16 -> 'S'; case INT64 -> 'J'; case INT32 -> 'I'; case BOOL -> 'B'; });
    }
    private static Class<?> array(char tag) { return switch (tag) { case 'D' -> double[].class; case 'F' -> float[].class;
        case 'S' -> short[].class; case 'J' -> long[].class; case 'I' -> int[].class; case 'B' -> byte[].class;
        default -> throw new AssertionError(tag); }; }
    private static Object storage(DataType type, CpuKernelSpecialization.CarrierAccess access, Arena arena) {
        if (access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) return arena.allocate(128, 8);
        return switch (type) { case FLOAT64 -> new double[32]; case FLOAT32 -> new float[32]; case BFLOAT16 -> new short[32];
            case INT64 -> new long[32]; case INT32 -> new int[32]; case BOOL -> new byte[32]; };
    }
    private static void fill(Object carrier, DataType type, long value) { for (int i = 0; i < 16; i++) put(carrier, type, i, value); }
    private static long sentinel(DataType t) { return switch (t) { case FLOAT64 -> 0x7ff80000000000a5L; case FLOAT32 -> 0x7fc000a5L;
        case BFLOAT16 -> 0x7fc5; case INT64 -> 0x5a5a5a5a5a5a5a5aL; case INT32 -> 0x5a5a5a5aL; case BOOL -> 1; }; }
    private static long edge(DataType t, int i) { return switch (t) { case FLOAT64 -> new long[] {0L,0x8000000000000000L,0x7ff0000000000042L,0x7ff0000000000000L,0xfff0000000000000L,0x7fefffffffffffffL,Double.doubleToRawLongBits(2147483647.75),Double.doubleToRawLongBits(1.0039062501)}[i];
        case FLOAT32 -> new long[] {0,0x80000000L,0x7fa12345L,0x7f800000L,0xff800000L,0x7f7fffffL,0x4f000000L,0x3f808000L}[i];
        case BFLOAT16 -> new long[] {0,0x8000,0x7f81,0x7f80,0xff80,0x7f7f,0x4f00,0x3f81}[i];
        case INT64 -> new long[] {0,1,-1,Long.MIN_VALUE,Long.MAX_VALUE,2155872257L,-2155872257L,16777217L}[i];
        case INT32 -> new long[] {0,1,-1,Integer.MIN_VALUE,Integer.MAX_VALUE,0x80000001L,0x7fffffffL,16777217L}[i];
        case BOOL -> i % 2; }; }
    private static long rightEdge(DataType t, int i) { return switch (t) { case FLOAT64 -> new long[] {0x8000000000000000L,0L,0x7ff0000000000042L,0xfff0000000000000L,0x7ff0000000000000L,Double.doubleToRawLongBits(-1d),Double.doubleToRawLongBits(2d),Double.doubleToRawLongBits(0.5d)}[i];
        case FLOAT32 -> new long[] {0x80000000L,0,0x7fa12345L,0xff800000L,0x7f800000L,0xbf800000L,0x40000000L,0x3f000000L}[i];
        case BFLOAT16 -> new long[] {0x8000,0,0x7f81,0xff80,0x7f80,0xbf80,0x4000,0x3f00}[i];
        case INT64 -> new long[] {1,-1,Long.MIN_VALUE,Long.MAX_VALUE,0,-1,2,3}[i];
        case INT32 -> new long[] {1,-1,Integer.MIN_VALUE,Integer.MAX_VALUE,0,-1,2,3}[i];
        case BOOL -> (i + 1) % 2; }; }
    private static void put(Object c, DataType t, long i, long bits) { if (c instanceof MemorySegment s) { switch(t) {
        case FLOAT64 -> s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, i, bits); case FLOAT32, INT32 -> s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, i, (int) bits);
        case BFLOAT16 -> s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, i, (short) bits); case INT64 -> s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, i, bits); case BOOL -> s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) bits); } return; }
        switch(t) { case FLOAT64 -> ((double[])c)[(int)i]=Double.longBitsToDouble(bits); case FLOAT32 -> ((float[])c)[(int)i]=Float.intBitsToFloat((int)bits); case BFLOAT16 -> ((short[])c)[(int)i]=(short)bits; case INT64 -> ((long[])c)[(int)i]=bits; case INT32 -> ((int[])c)[(int)i]=(int)bits; case BOOL -> ((byte[])c)[(int)i]=(byte)bits; } }
    private static long raw(Object c, DataType t, long i) { if (c instanceof MemorySegment s) return switch(t) { case FLOAT64, INT64 -> s.getAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,i); case FLOAT32, INT32 -> Integer.toUnsignedLong(s.getAtIndex(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED,i)); case BFLOAT16 -> Short.toUnsignedLong(s.getAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED,i)); case BOOL -> Byte.toUnsignedLong(s.getAtIndex(java.lang.foreign.ValueLayout.JAVA_BYTE,i)); }; return switch(t) { case FLOAT64 -> Double.doubleToRawLongBits(((double[])c)[(int)i]); case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(((float[])c)[(int)i])); case BFLOAT16 -> Short.toUnsignedLong(((short[])c)[(int)i]); case INT64 -> ((long[])c)[(int)i]; case INT32 -> Integer.toUnsignedLong(((int[])c)[(int)i]); case BOOL -> Byte.toUnsignedLong(((byte[])c)[(int)i]); }; }
    private static final class WitnessLoader extends ClassLoader { WitnessLoader() { super(CpuOrdinaryPointwiseCastStructuralOracleTest.class.getClassLoader()); } Class<?> define(byte[] bytes) { return defineClass(null, bytes, 0, bytes.length); } }

    @Test void inventorySeparatesPointwiseAndCrossCategoryRejectionsFromTheGeneratedStructuralDenominator()
            throws Exception {
        String inventory = resource("generated-coverage-inventory.tsv");
        Map<String, Long> rejectedByCategory = new java.util.TreeMap<>();
        for (String line : inventory.lines().skip(1).toList()) {
            String[] row = line.split("\\t", -1);
            if (row.length == 27 && row[21].startsWith("REJECTED_"))
                rejectedByCategory.merge(rejectionCategory(row), 1L, Long::sum);
        }
        long pointwise = rejectedByCategory.getOrDefault("pointwise", 0L);
        long other = rejectedByCategory.values().stream().mapToLong(Long::longValue).sum() - pointwise;
        assertEquals(152L, pointwise, "pointwise rejected-row count");
        assertEquals(21L, other, "other-category rejected-row count");
        assertEquals(173L, pointwise + other, "total cross-category rejection count");
    }

    private static List<String[]> ownedRows() throws Exception {
        List<String[]> rows = new ArrayList<>();
        for (String line : resource("generated-coverage-inventory.tsv").lines().skip(1).toList()) {
            String[] row = line.split("\\t", -1);
            if (row.length == 27 && row[0].startsWith("pointwise-matrix:")
                    && row[21].equals("GENERATED")) rows.add(row);
        }
        return List.copyOf(rows);
    }

    private static String rejectionCategory(String[] row) {
        String owner = row[0];
        String[] parts = owner.split(":", 3);
        assertTrue(parts.length >= 2 && parts[0].equals("rejected"),
                "rejected inventory owner category: " + owner);
        return parts[1];
    }

    private static String boundaryType(String values) {
        var matcher = java.util.regex.Pattern.compile(":(FLOAT64|FLOAT32|BFLOAT16|INT64|INT32|BOOL)")
                .matcher(values);
        return matcher.find() ? matcher.group(1) : null;
    }

    private static String boundaryOutputType(String values) {
        var matcher = java.util.regex.Pattern.compile(":(FLOAT64|FLOAT32|BFLOAT16|INT64|INT32|BOOL)")
                .matcher(values);
        String result = null;
        while (matcher.find()) result = matcher.group(1);
        return result;
    }

    private static Map<String, MethodModel> methods(byte[] bytes) {
        Map<String, MethodModel> methods = new LinkedHashMap<>();
        for (MethodModel method : ClassFile.of().parse(bytes).methods()) {
            if (method.methodName().stringValue().startsWith("entry")) {
                assertTrue(methods.put(method.methodName().stringValue(), method) == null,
                        "duplicate clean selected method");
            }
        }
        return Map.copyOf(methods);
    }

    private static String resource(String name) throws Exception {
        try (InputStream input = CpuOrdinaryPointwiseCastStructuralOracleTest.class
                .getResourceAsStream(BASE + name)) {
            assertTrue(input != null, name);
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
