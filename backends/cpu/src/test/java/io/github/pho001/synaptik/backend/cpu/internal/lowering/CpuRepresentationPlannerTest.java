package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Focused bounded-enumeration, typed-fact, and two-copy integration tests for CPU 0008E.
 *
 * <p>Assertions against the historical external evidence corpus are opt-in through
 * {@value #RETAINED_EVIDENCE_PROPERTY}; ordinary module validation does not depend on a
 * machine-local {@code /private/tmp} archive.</p>
 */
class CpuRepresentationPlannerTest {
    private static final String RETAINED_EVIDENCE_PROPERTY =
            "synaptik.cpu.0008e.retainedEvidence";
    private static final Path RETAINED_PREDECESSOR = Path.of(
            "/private/tmp/synaptik-cpu-0008e-retained-evidence-20260827");
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008e-schema53-authoritative-evidence-v3-final2-20260827");
    @Test void exactCeilingsAdmitEightSourcesAndSixtyFourByThirtySevenFacts() {
        assertAll(
                () -> assertTrue(CpuRepresentationPlanner.completeWithinBudgets(
                        new CpuRepresentationPlanner.RepresentationProbe(64, 8, 384,
                                false, false))),
                () -> assertFalse(CpuRepresentationPlanner.completeWithinBudgets(
                        new CpuRepresentationPlanner.RepresentationProbe(64, 9, 384,
                                false, false))),
                () -> assertFalse(CpuRepresentationPlanner.completeWithinBudgets(
                        new CpuRepresentationPlanner.RepresentationProbe(64, 8, 384,
                                true, false))),
                () -> assertEquals(2_368, 64 * (1 + 8 + 28)),
                () -> assertEquals(2_753, 384 + 2_368 + 1));
    }

    @Test void coConsumedPairIsRejectedBeforeRankingAndBothSinglesRemain() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor general = descriptor(shape,
                LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        TensorDescriptor dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 3, 96, 1, 1);
        var analysis = CpuPartitionPreparerTest.analyze(general, general, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy));
        var selected = (CpuRepresentationDecision.Selection) analysis.plan()
                .representationDecisions().getLast();
        var decisions = analysis.plan().representationDecisions();
        var pair = decisions.stream().filter(CpuRepresentationDecision.Rejection.class::isInstance)
                .map(CpuRepresentationDecision.Rejection.class::cast)
                .filter(rejection -> rejection.reason()
                        == CpuRepresentationDecision.RejectionReason.CO_CONSUMED_PAIR)
                .findFirst().orElseThrow();
        var singles = decisions.stream().filter(CpuRepresentationDecision.Variant.class::isInstance)
                .map(CpuRepresentationDecision.Variant.class::cast)
                .filter(variant -> variant.identity().topology().equals(pair.identity().topology()))
                .filter(variant -> variant.identity().materializations().size() == 1).toList();
        int pairPosition = decisions.indexOf(pair);
        assertAll(
                () -> assertTrue(analysis.plan().materializations().isEmpty()),
                () -> assertEquals(
                        CpuRepresentationDecision.SelectionReason.DIRECT_MATERIALIZATION_UNPROVED,
                        selected.reason()),
                () -> assertTrue(selected.selected().materializations().isEmpty()),
                () -> assertEquals(List.of(0, 1), pair.identity().materializations().stream()
                        .map(CpuRepresentationDecision.MaterializationIdentity
                                ::sourceBoundaryPosition).toList()),
                () -> assertEquals(List.of(8, 9), pair.identity().materializations().stream()
                        .map(CpuRepresentationDecision.MaterializationIdentity
                                ::workspaceRequirementId).toList()),
                () -> assertEquals(List.of(List.of(0), List.of(1)), singles.stream()
                        .map(variant -> variant.identity().materializations().stream()
                                .map(CpuRepresentationDecision.MaterializationIdentity
                                        ::sourceBoundaryPosition).toList()).toList()),
                () -> assertEquals(List.of(CpuRepresentationDecision.Variant.class,
                                CpuRepresentationDecision.Variant.class,
                                CpuRepresentationDecision.Variant.class,
                                CpuRepresentationDecision.Rejection.class),
                        decisions.subList(pairPosition - 3, pairPosition + 1).stream()
                                .map(Object::getClass).toList()),
                () -> assertEquals(decisions.indexOf(decisions.stream()
                        .filter(CpuRepresentationDecision.Variant.class::isInstance)
                        .map(CpuRepresentationDecision.Variant.class::cast)
                        .filter(variant -> variant.identity().equals(selected.selected()))
                        .findFirst().orElseThrow()), selected.stableRank()),
                () -> assertEquals(List.of(CpuRepresentationDecision.Variant.class,
                                CpuRepresentationDecision.Rejection.class,
                                CpuRepresentationDecision.Selection.class),
                        List.of(CpuRepresentationDecision.class.getPermittedSubclasses())));

    }

    void selectedClassesAreDirectLoopsAndFiveIsolatedForksMeetOracleGate()
            throws Exception {
        Files.createDirectories(EVIDENCE.resolve("classes"));
        Shape shape = Shape.of(64, 64);
        TensorDescriptor general = descriptor(shape,
                LayoutDescriptor.of(shape, new long[] {1, 64}, 0, true));
        TensorDescriptor dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 3, 2L * 64 * 64 * 8, 1, 1);
        var selected = CpuPartitionPreparerTest.analyze(general, general, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy)).plan();
        var direct = CpuPartitionPreparerTest.analyze(general, general, dense, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan();
        var store = new CpuGeneratedKernelArtifactStore();
        byte[] directBytes = store.loadOrGenerate(
                direct.units().getFirst().portablePlan().specialization(),
                direct.units().getFirst().portablePlan().kernelIr()).classBytes();
        byte[] retainedDirectBytes = store.loadOrGenerate(
                selected.units().getFirst().portablePlan().specialization(),
                selected.units().getFirst().portablePlan().kernelIr()).classBytes();
        assertArrayEquals(directBytes, retainedDirectBytes,
                "direct representation class bytes must remain exact");
        Path controls = EVIDENCE.resolve("controls");
        Files.createDirectories(controls);
        GeneratedCopy affected = affectedControl();
        assertEquals("9fd48bf5c6da8abb5549427e7b0597f3ed777127028f0b3715b1f0525faa3b21",
                affected.specialization().structuralKey());
        assertNotEquals("cfd4e5ba18fc1fdbdcb4b30472a82ec38687620bd91c44e9d0c17c53abf6ede6",
                sha256(affected.bytes()));
        Path affectedControl = controls.resolve("affected-affine-array-segment.class");
        Files.write(affectedControl, affected.bytes());
        retainJavap(affectedControl,
                controls.resolve("affected-affine-array-segment.javap.txt"));
        Files.writeString(controls.resolve("affected-affine-array-segment.structural-key"),
                affected.specialization().structuralKey() + "\n");
        Files.writeString(controls.resolve("affected-affine-array-segment.specialization"),
                affected.specialization() + "\n");
        var generated = new ArrayList<byte[]>();
        for (CpuMaterializationPlan copy : selected.materializations()) generated.add(
                store.loadOrGenerate(copy.copySpecialization(), copy.copyIr().encodedKernelIr())
                        .classBytes());
        var denseControl = CpuPartitionPreparerTest.analyze(dense, dense, dense, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan();
        var represented = denseControl.units().getFirst().portablePlan();
        byte[] representedBytes = store.loadOrGenerate(represented.specialization(),
                represented.kernelIr()).classBytes();
        assertEquals("5bb43f82369fcb43a3eb50b9be42d856016ec91bab3aeb0f83b4b0af928a6efd",
                sha256(representedBytes));
        assertEquals("600f01e111b3c40b06e406d8e5047818e2ec875166fb27d706b004a121a8cdb3",
                represented.specialization().structuralKey());
        Path directControl = controls.resolve("direct-pointwise.class");
        Files.write(directControl, representedBytes);
        retainJavap(directControl, controls.resolve("direct-pointwise.javap.txt"));
        Files.writeString(controls.resolve("direct-pointwise.structural-key"),
                represented.specialization().structuralKey() + "\n");
        Files.writeString(controls.resolve("direct-pointwise.specialization"),
                represented.specialization() + "\n");
        generated.add(representedBytes);
        for (int index = 0; index < generated.size(); index++) {
            byte[] bytes = generated.get(index);
            java.lang.classfile.ClassFile.of().parse(bytes);
            String constants = new String(bytes, StandardCharsets.ISO_8859_1);
            assertAll(
                    () -> assertFalse(constants.contains("CpuRepresentation")),
                    () -> assertFalse(constants.contains("CpuMaterializationPlan")),
                    () -> assertFalse(constants.contains("CopyCall")),
                    () -> assertFalse(constants.contains("java/lang/reflect")),
                    () -> assertFalse(constants.contains("java/lang/Integer")));
            Path classFile = EVIDENCE.resolve("classes/selected-" + index + ".class");
            Files.write(classFile, bytes);
            Process javap = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin",
                    "javap").toString(), "-c", "-v", "-p", classFile.toString())
                    .redirectErrorStream(true).start();
            String inspection = new String(javap.getInputStream().readAllBytes(),
                    StandardCharsets.UTF_8);
            assertEquals(0, javap.waitFor(), inspection);
            Files.writeString(EVIDENCE.resolve("classes/selected-" + index + ".javap.txt"),
                    inspection);
        }
        var ratios = new ArrayList<Double>();
        var equalWorkRatios = new ArrayList<Double>();
        var gateLines = new ArrayList<String>();
        var equalWorkLines = new ArrayList<String>();
        Files.createDirectories(EVIDENCE.resolve("forks"));
        for (int fork = 0; fork < 5; fork++) {
            Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin",
                    "java").toString(), "-Xms512m", "-Xmx512m", "--add-modules",
                    "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), CpuRepresentationPlannerTest.class
                            .getName(), Integer.toString(fork)).redirectErrorStream(true).start();
            String output = new String(process.getInputStream().readAllBytes(),
                    StandardCharsets.UTF_8);
            Files.writeString(EVIDENCE.resolve("forks/fork-" + fork + ".csv"), output);
            assertEquals(0, process.waitFor(), output);
            output.lines().filter(line -> line.startsWith("GATE,"))
                    .forEach(line -> {
                        gateLines.add(line);
                        ratios.add(Double.parseDouble(line.substring(line.lastIndexOf(',') + 1)));
                    });
            output.lines().filter(line -> line.startsWith("COPY_DIRECT,"))
                    .forEach(line -> {
                        equalWorkLines.add(line);
                        equalWorkRatios.add(
                                Double.parseDouble(line.substring(line.lastIndexOf(',') + 1)));
                    });
        }
        assertEquals(40, ratios.size());
        assertEquals(40, equalWorkRatios.size());
        assertTrue(ratios.stream().allMatch(ratio -> ratio <= 1.15), ratios.toString());
        assertTrue(equalWorkRatios.stream().allMatch(ratio -> ratio <= 1.15),
                equalWorkRatios.toString());
        var aggregate = new StringBuilder();
        gateLines.forEach(line -> aggregate.append(line).append('\n'));
        equalWorkLines.forEach(line -> aggregate.append(line).append('\n'));
        appendSummary(aggregate, "GATE", gateLines);
        appendSummary(aggregate, "COPY_DIRECT", equalWorkLines);
        Files.writeString(EVIDENCE.resolve("aggregate.csv"), aggregate);
        String manifest = manifest(EVIDENCE);
        Files.writeString(EVIDENCE.resolve("manifest.sha256"), manifest);
        assertEquals(manifest, manifest(EVIDENCE));
    }

    void finalPerformanceMatrixRowsComeFromSelectedProductionPlans() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                DataType.INT64, DataType.BOOL)) assertDoesNotThrow(
                        () -> preflightProductionCase(type, "one-copy"), type.toString());
        preflightProductionCase(DataType.FLOAT64, "same-binary-single-copy");
        preflightProductionCase(DataType.FLOAT64, "disjoint-consumer-two-copy");
        preflightProductionCase(DataType.FLOAT64, "copy-once-reuse");
    }

    void selectedCopyIsFilledOnceAndReusedAcrossSplitConsumers() {
        ProductionCase production = productionCase(DataType.FLOAT64, "copy-once-reuse");
        PreparedExecutable executable = production.finalized();
        assertAll(
                () -> assertEquals(1, executable.memoryPlan().workspaces().size()),
                () -> assertEquals(2, production.unitCount()),
                () -> assertEquals(2, production.materializations().getFirst().consumers().size()));
        var buffers = executable.memoryPlan().buffers().stream().map(entry ->
                CpuNativeBuffer.allocate(DataType.FLOAT64, entry.byteSize(), entry.byteAlignment()))
                .toList();
        int aPosition = production.boundaryValues().indexOf(new ValueId(0));
        int bPosition = production.boundaryValues().indexOf(new ValueId(1));
        int publishedPosition = production.boundaryValues().indexOf(new ValueId(2));
        int outputPosition = production.boundaryValues().indexOf(new ValueId(4));
        assertTrue(aPosition >= 0 && bPosition >= 0 && publishedPosition >= 0
                && outputPosition >= 0);
        int count = 512 * 512;
        for (int logical = 0; logical < count; logical++) {
            int row = logical / 512;
            int column = logical % 512;
            long generalAddress = row + column * 512L;
            buffers.get(aPosition).segment().set(ValueLayout.JAVA_DOUBLE,
                    generalAddress * Double.BYTES, logical % 17 + 1.0);
            buffers.get(bPosition).segment().set(ValueLayout.JAVA_DOUBLE,
                    (long) logical * Double.BYTES, logical % 7 + 2.0);
        }
        var bindings = buffers.stream().map(buffer -> List.of(new BufferRepresentationBinding(
                buffer, RunResourceOwnership.RUN_OWNED))).toList();
        var workspaces = executable.memoryPlan().workspaces().stream().map(entry ->
                CpuContiguousWorkspace.allocate(entry.byteSize(), entry.byteAlignment()))
                .map(io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation.class::cast)
                .toList();
        var state = new RunState(executable.memoryPlan(), bindings, workspaces);
        try {
            executable.bind(state).execute();
            for (int logical = 0; logical < count; logical++) {
                double a = logical % 17 + 1.0;
                double b = logical % 7 + 2.0;
                assertEquals(a + b, buffers.get(publishedPosition).segment().get(
                        ValueLayout.JAVA_DOUBLE, (long) logical * Double.BYTES), 0.0);
                assertEquals((a + b + b) * a, buffers.get(outputPosition).segment().get(
                        ValueLayout.JAVA_DOUBLE, (long) logical * Double.BYTES), 0.0);
            }
        } finally {
            state.close();
        }
    }

    @Test void retainedEvidenceAndGeneratedClassControlsRemainExactWithoutRerunningForks()
            throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean(RETAINED_EVIDENCE_PROPERTY),
                "historical CPU 0008E evidence verification is opt-in");
        Path controls = EVIDENCE.resolve("controls");
        Path final2 = EVIDENCE.resolve("forks/fork-0.csv");
        Path coConsumed = Path.of("/private/tmp/"
                + "synaptik-cpu-0008e-schema53-evidence-v3-complete-20260827/forks/fork-0.csv");
        GeneratedCopy affected = affectedControl();
        Shape shape = Shape.of(64, 64);
        TensorDescriptor dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var direct = CpuPartitionPreparerTest.analyze(dense, dense, dense, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan().units().getFirst().portablePlan();
        byte[] directBytes = new CpuGeneratedKernelArtifactStore().loadOrGenerate(
                direct.specialization(), direct.kernelIr()).classBytes();
        for (byte[] bytes : List.of(affected.bytes(), directBytes)) {
            java.lang.classfile.ClassFile.of().parse(bytes);
            String constants = new String(bytes, StandardCharsets.ISO_8859_1);
            for (String forbidden : List.of("CpuRepresentation", "CpuMaterializationPlan",
                    "CopyCall", "java/lang/reflect", "java/lang/Integer", "valueOf"))
                assertFalse(constants.contains(forbidden), forbidden);
        }
        String final2Text = Files.readString(final2);
        assertAll(
                () -> assertEquals(
                        "c55d07272272fb484b4a6580312aba34aa67483e404f42ecddaf8a717d7f4612",
                        sha256(Files.readAllBytes(final2))),
                () -> assertEquals(
                        "d581b84022d259fd92805e8ad26aab0d6e6437ace96aaed89eef08d8eb15f26d",
                        sha256(Files.readAllBytes(coConsumed))),
                () -> assertEquals(
                        "5bb43f82369fcb43a3eb50b9be42d856016ec91bab3aeb0f83b4b0af928a6efd",
                        sha256(directBytes)),
                () -> assertEquals(
                        "fad075e7abd6586cdd53da6e2787ef77d8a92f972cf6b6fb66a05cf501cb4369",
                        sha256(affected.bytes())),
                () -> assertArrayEquals(Files.readAllBytes(
                        controls.resolve("direct-pointwise.class")), directBytes),
                () -> assertArrayEquals(Files.readAllBytes(
                        controls.resolve("affected-affine-array-segment.class")), affected.bytes()),
                () -> assertTrue(final2Text.contains("1.000326680")),
                () -> assertTrue(final2Text.contains("2.605250934")));
    }

    @Test void formerlyFavorableProductionFormsRemainCompleteCandidateOnlyFacts() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                DataType.INT64, DataType.BOOL)) assertCandidateOnly(type, "one-copy", 1);
        assertCandidateOnly(DataType.FLOAT64, "same-binary-single-copy", 1);
        assertCandidateOnly(DataType.FLOAT64, "disjoint-consumer-two-copy", 2);
        assertCandidateOnly(DataType.FLOAT64, "copy-once-reuse", 1);
    }

    @Test void disabledAndOverflowFallbackReasonsRemainExact() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor general = descriptor(shape,
                LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        TensorDescriptor dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var disabledPlan = CpuPartitionPreparerTest.analyze(general, general, dense, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan();
        var disabled = (CpuRepresentationDecision.Selection) disabledPlan
                .representationDecisions().getLast();
        var overflowPolicy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, 0, 2,
                Long.MAX_VALUE, 0, 0);
        var overflowPlan = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                        overflowPolicy)).plan();
        var overflow = (CpuRepresentationDecision.Selection) overflowPlan
                .representationDecisions().getLast();
        assertAll(
                () -> assertEquals(CpuRepresentationDecision.SelectionReason
                        .DIRECT_POLICY_DISABLED, disabled.reason()),
                () -> assertTrue(disabled.selected().materializations().isEmpty()),
                () -> assertTrue(disabledPlan.materializations().isEmpty()),
                () -> assertEquals(CpuRepresentationDecision.SelectionReason
                        .DIRECT_UNCERTAINTY, overflow.reason()),
                () -> assertEquals(overflow.canonicalDirect(), overflow.selected()),
                () -> assertTrue(overflowPlan.materializations().isEmpty()));
    }

    @Test void generatedCandidateCopyRunsOnceAndItsDenseResultFeedsTwoConsumers()
            throws Throwable {
        GeneratedCopy copy = affectedControl();
        double[] source = {10, 20, 30, 40, 50, 60};
        long[] geometry = {5, 4, 4, 5, 3, 6, 2, 700, 1, 800, 0, 900};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dense = arena.allocate(12 * Double.BYTES, Double.BYTES);
            copy.handle().invokeExact(source, dense, geometry, 0L, 6L);
            double[] left = new double[6], right = new double[6];
            for (int index = 0; index < 6; index++) {
                double value = dense.get(ValueLayout.JAVA_DOUBLE,
                        (long) (index + 4) * Double.BYTES);
                left[index] = value + 1;
                right[index] = value * 2;
            }
            assertArrayEquals(new double[] {61, 51, 41, 31, 21, 11}, left);
            assertArrayEquals(new double[] {120, 100, 80, 60, 40, 20}, right);
        }
    }

    /** Runs one honest warmed performance fork for the retained evidence test. */
    public static void main(String[] arguments) throws Throwable {
        int fork = Integer.parseInt(arguments[0]);
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                DataType.INT64, DataType.BOOL)) benchmark(type, fork, "one-copy");
        benchmark(DataType.FLOAT64, fork, "same-binary-single-copy");
        benchmark(DataType.FLOAT64, fork, "disjoint-consumer-two-copy");
        benchmark(DataType.FLOAT64, fork, "copy-once-reuse");
    }

    private static void benchmark(DataType type, int fork, String form) throws Throwable {
        int count = 262_144;
        ProductionCase production = productionCase(type, form);
        List<GeneratedCopy> generatedCopies = generatedCopies(production);
        List<MethodHandle> oracles = production.materializations().stream().map(materialization -> {
            try { return oracleHandle(type, materialization.sourceCarrier()); }
            catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
        }).toList();
        for (int index = 0; index < generatedCopies.size(); index++) retainInvokedCopy(
                generatedCopies.get(index), oracles.get(index), production, type, form, fork, index);
        Object firstArray = source(type, count, fork + 1);
        Object secondArray = source(type, count, fork + 17);
        Object third = source(type, count, fork + 31);
        Object first = carrierSource(type, firstArray,
                production.materializations().getFirst().sourceCarrier());
        Object second = production.materializations().size() == 2
                ? carrierSource(type, secondArray,
                        production.materializations().get(1).sourceCarrier()) : secondArray;
        List<long[]> geometries = production.materializations().stream()
                .map(CpuMaterializationPlan::affineAddressPairs).toList();
        assertTrue(geometries.stream().allMatch(geometry -> geometry.length == count * 2));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left = arena.allocate((long) count * type.byteWidth(), type.byteWidth());
            MemorySegment right = arena.allocate((long) count * type.byteWidth(), type.byteWidth());
            MemorySegment intermediate = arena.allocate((long) count * type.byteWidth(),
                    type.byteWidth());
            MemorySegment generatedOutput = arena.allocate((long) count * type.byteWidth(),
                    type.byteWidth());
            MemorySegment oracleOutput = arena.allocate((long) count * type.byteWidth(),
                    type.byteWidth());
            MemorySegment directOutput = arena.allocate((long) count * type.byteWidth(),
                    type.byteWidth());
            for (int warm = 0; warm < 80; warm++) {
                generatedCase(form, type, generatedCopies.stream().map(GeneratedCopy::handle)
                        .toList(), production.materializations(), first, second, third, left, right,
                        intermediate, generatedOutput, geometries, count);
                oracleCase(form, type, oracles, production.materializations(), first, second, third,
                        left, right, intermediate, oracleOutput, geometries, count);
                directAccess(form, type, first, second, third, intermediate, directOutput,
                        geometries, count);
            }
            List<MethodHandle> handles = generatedCopies.stream().map(GeneratedCopy::handle).toList();
            long generatedResult = generatedCase(form, type, handles, production.materializations(),
                    first, second, third, left, right, intermediate, generatedOutput, geometries,
                    count);
            long oracleResult = oracleCase(form, type, oracles, production.materializations(),
                    first, second, third, left, right, intermediate, oracleOutput, geometries, count);
            if (generatedResult != oracleResult
                    || generatedOutput.mismatch(oracleOutput) != -1) {
                throw new AssertionError("generated/oracle result mismatch");
            }
            long[] generated = new long[5], direct = new long[5], copyPlan = new long[5],
                    directAccess = new long[5];
            for (int sample = 0; sample < generated.length; sample++) {
                boolean generatedFirst = ((fork + sample) & 1) == 0;
                if (generatedFirst) {
                    generated[sample] = timed(() -> generatedCase(form, type, handles,
                            production.materializations(), first, second, third, left, right,
                            intermediate, generatedOutput, geometries, count));
                    direct[sample] = timed(() -> oracleCase(form, type, oracles,
                            production.materializations(), first, second, third, left, right,
                            intermediate, oracleOutput, geometries, count));
                } else {
                    direct[sample] = timed(() -> oracleCase(form, type, oracles,
                            production.materializations(), first, second, third, left, right,
                            intermediate, oracleOutput, geometries, count));
                    generated[sample] = timed(() -> generatedCase(form, type, handles,
                            production.materializations(), first, second, third, left, right,
                            intermediate, generatedOutput, geometries, count));
                }
                copyPlan[sample] = timed(() -> generatedCase(form, type, handles,
                        production.materializations(), first, second, third, left, right,
                        intermediate, generatedOutput, geometries, count));
                directAccess[sample] = timed(() -> directAccess(form, type, first, second, third,
                        intermediate, directOutput, geometries, count));
                long selectedResult = generatedCase(form, type, handles,
                        production.materializations(), first, second, third, left, right,
                        intermediate, generatedOutput, geometries, count);
                long directResult = directAccess(form, type, first, second, third, intermediate,
                        directOutput, geometries, count);
                if (selectedResult != directResult
                        || generatedOutput.mismatch(directOutput) != -1) {
                    throw new AssertionError("selected/direct result mismatch");
                }
                System.out.printf(Locale.ROOT, "SAMPLE,%s,%s,%d,%d,%d,%d,%.9f,%d,%d,%.9f%n",
                        form, type, fork, sample, generated[sample], direct[sample],
                        (double) generated[sample] / direct[sample], copyPlan[sample],
                        directAccess[sample], (double) copyPlan[sample] / directAccess[sample]);
            }
            double ratio = (double) median(generated) / median(direct);
            double planRatio = (double) median(copyPlan) / median(directAccess);
            System.out.printf(Locale.ROOT, "GATE,%s,%s,%d,%.9f%n", form, type, fork, ratio);
            System.out.printf(Locale.ROOT, "COPY_DIRECT,%s,%s,%d,%.9f%n", form, type, fork,
                    planRatio);
            if (ratio > 1.15) throw new AssertionError(type + " " + form + " ratio " + ratio);
            if (planRatio > 1.15) throw new AssertionError(
                    type + " " + form + " selected/direct ratio " + planRatio);
        }
    }

    @FunctionalInterface private interface ThrowingCall { long invoke() throws Throwable; }
    private static long timed(ThrowingCall call) throws Throwable {
        long start = System.nanoTime(), checksum = 0;
        for (int iteration = 0; iteration < 24; iteration++) checksum ^= call.invoke();
        long elapsed = System.nanoTime() - start;
        if (checksum == Long.MIN_VALUE) throw new AssertionError("impossible checksum");
        return elapsed;
    }

    private static long generatedCase(String form, DataType type, List<MethodHandle> copies,
            List<CpuMaterializationPlan> plans, Object first, Object second, Object third,
            MemorySegment left, MemorySegment right, MemorySegment intermediate,
            MemorySegment output, List<long[]> geometries, int count) throws Throwable {
        invokeCopies(type, copies, plans, first, second, left, right, geometries, count);
        return consume(form, type, left, plans.size() == 2 ? right : second, third,
                intermediate, output, count);
    }

    private static long oracleCase(String form, DataType type, List<MethodHandle> oracles,
            List<CpuMaterializationPlan> plans, Object first, Object second, Object third,
            MemorySegment left, MemorySegment right, MemorySegment intermediate,
            MemorySegment output, List<long[]> geometries, int count) throws Throwable {
        invokeCopies(type, oracles, plans, first, second, left, right, geometries, count);
        return consume(form, type, left, plans.size() == 2 ? right : second, third,
                intermediate, output, count);
    }

    private static long directAccess(String form, DataType type, Object first, Object second,
            Object third, MemorySegment intermediate, MemorySegment output,
            List<long[]> geometries, int count) {
        return consumeDirect(form, type, first, second, third, intermediate, output, geometries,
                count);
    }

    private static List<GeneratedCopy> generatedCopies(ProductionCase production) {
        return production.materializations().stream().map(copy -> {
            var artifact = new CpuGeneratedKernelArtifactStore().loadOrGenerate(
                    copy.copySpecialization(), copy.copyIr().encodedKernelIr());
            assertEquals(copy.copySpecialization(), artifact.specialization());
            return new GeneratedCopy(artifact.entryPoint(), artifact.hiddenClass(),
                    artifact.classBytes(), copy.copySpecialization(),
                    copy.copyIr().encodedKernelIr());
        }).toList();
    }

    private static void preflightProductionCase(DataType type, String form) {
        ProductionCase production = productionCase(type, form);
        List<GeneratedCopy> copies = generatedCopies(production);
        assertEquals(production.materializations().size(), copies.size());
        for (int index = 0; index < copies.size(); index++) {
            CpuMaterializationPlan plan = production.materializations().get(index);
            GeneratedCopy copy = copies.get(index);
            java.lang.classfile.ClassFile.of().parse(copy.bytes());
            assertAll(type + " " + form + " copy " + index,
                    () -> assertEquals(plan.copySpecialization(), copy.specialization()),
                    () -> assertEquals(plan.copyIr().encodedKernelIr(), copy.ir()),
                    () -> assertEquals(plan.copySpecialization().entryType(), copy.handle().type()),
                    () -> assertEquals(copy.handle().type(), MethodType.methodType(
                            copy.owner().getDeclaredMethods()[0].getReturnType(),
                            copy.owner().getDeclaredMethods()[0].getParameterTypes())),
                    () -> assertEquals(copy.handle().type(), oracleHandle(type,
                            plan.sourceCarrier()).type()));
        }
    }

    private static GeneratedCopy affectedControl() {
        var source = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 1,
                List.of(CpuAccessPlan.AxisRole.STRIDED), 0);
        var result = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuAffineCopyIr(DataType.FLOAT64, source, result,
                List.of(new CpuAffineCopyIr.MappingStep(
                        CpuAffineCopyIr.MappingKind.CONTIGUOUS, 1, 1, List.of())),
                CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                        .ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT64, DataType.FLOAT64),
                List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                0, -1, List.of(), false);
        var artifact = new CpuGeneratedKernelArtifactStore().loadOrGenerate(
                specialization, ir.encodedKernelIr());
        return new GeneratedCopy(artifact.entryPoint(), artifact.hiddenClass(),
                artifact.classBytes(), specialization, ir.encodedKernelIr());
    }

    private static void assertCandidateOnly(DataType type, String form, int expectedCopies) {
        Shape shape = Shape.of(512, 512);
        TensorDescriptor general = typedDescriptor(type, shape,
                LayoutDescriptor.of(shape, new long[] {1, 512}, 0, true));
        TensorDescriptor dense = typedDescriptor(type, shape, LayoutDescriptor.contiguous(shape));
        boolean twoCopies = form.equals("disjoint-consumer-two-copy");
        boolean reuse = form.equals("copy-once-reuse");
        var discovery = new CpuPartitionPreparer().analyze(benchmarkContext(type, general,
                twoCopies ? general : dense, dense, dense, form,
                CpuPartitionAnalysisInputs.DEFAULT)).plan();
        boolean segmentCarriers = reuse || twoCopies;
        List<CpuKernelSpecialization.CarrierAccess> carriers = segmentCarriers ? List.of()
                : discovery.boundaryValues().stream()
                    .map(value -> value.value() <= 2 ? heapCarrier(type)
                            : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                    .toList();
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 1, 20, 1, 3,
                2L * shape.knownElementCount().orElseThrow() * Long.BYTES, 1, 1);
        var plan = new CpuPartitionPreparer().analyze(benchmarkContext(type, general,
                twoCopies ? general : dense, dense, dense, form,
                new CpuPartitionAnalysisInputs(false, carriers,
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy))).plan();
        var selection = (CpuRepresentationDecision.Selection) plan.representationDecisions()
                .getLast();
        var variants = plan.representationDecisions().stream()
                .filter(CpuRepresentationDecision.Variant.class::isInstance)
                .map(CpuRepresentationDecision.Variant.class::cast).toList();
        var materialized = variants.stream().filter(variant ->
                variant.identity().materializations().size() == expectedCopies).toList();
        assertAll(type + " " + form,
                () -> assertTrue(plan.materializations().isEmpty()),
                () -> assertTrue(selection.selected().materializations().isEmpty()),
                () -> assertEquals(
                        CpuRepresentationDecision.SelectionReason.DIRECT_MATERIALIZATION_UNPROVED,
                        selection.reason()),
                () -> assertFalse(materialized.isEmpty()),
                () -> assertTrue(materialized.stream().allMatch(variant ->
                        variant.selectedDirectCost().orElseThrow()
                                > variant.selectedCopiedCost().orElseThrow()
                                && variant.netBenefit().orElseThrow() > 0)),
                () -> assertTrue(materialized.stream().allMatch(variant ->
                        variant.identity().materializations().stream()
                                .map(CpuRepresentationDecision.MaterializationIdentity
                                        ::workspaceRequirementId).toList()
                                .equals(java.util.stream.IntStream.range(8,
                                        8 + expectedCopies).boxed().toList()))));
        if (twoCopies) assertTrue(materialized.stream().anyMatch(variant ->
                variant.identity().materializations().stream().map(
                        CpuRepresentationDecision.MaterializationIdentity
                                ::sourceBoundaryPosition).toList().equals(List.of(0, 1))));
        if (reuse) assertTrue(materialized.stream().anyMatch(variant ->
                variant.identity().materializations().getFirst().consumers().size() >= 2));
        if (type == DataType.FLOAT64 && form.equals("one-copy")) {
            CpuRepresentationDecision.Variant candidate = materialized.getFirst();
            assertAll("final2 production facts",
                    () -> assertEquals(15_728_640L,
                            candidate.selectedDirectCost().orElseThrow()),
                    () -> assertEquals(1_572_864L,
                            candidate.selectedCopiedCost().orElseThrow()),
                    () -> assertEquals(14_155_776L, candidate.netBenefit().orElseThrow()),
                    () -> assertEquals(9_000, candidate.benefitBasisPoints().orElseThrow()));
        }
    }

    private static ProductionCase productionCase(DataType type, String form) {
        Shape shape = Shape.of(512, 512);
        TensorDescriptor general = typedDescriptor(type, shape,
                LayoutDescriptor.of(shape, new long[] {1, 512}, 0, true));
        TensorDescriptor dense = typedDescriptor(type, shape, LayoutDescriptor.contiguous(shape));
        boolean twoCopies = form.equals("disjoint-consumer-two-copy");
        boolean reuse = form.equals("copy-once-reuse");
        PrepareContext<CpuPartitionAnalysisInputs> discovery = benchmarkContext(type, general,
                twoCopies ? general : dense, dense, dense, form,
                CpuPartitionAnalysisInputs.DEFAULT);
        var discovered = new CpuPartitionPreparer().analyze(discovery).plan();
        boolean defaultSegmentCarriers = reuse || twoCopies;
        List<CpuKernelSpecialization.CarrierAccess> carriers = defaultSegmentCarriers ? List.of()
                : discovered.boundaryValues().stream()
                    .map(value -> value.value() <= 2 ? heapCarrier(type)
                            : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                    .toList();
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 1, 20, 1, 3,
                2L * shape.knownElementCount().orElseThrow() * Long.BYTES, 1, 1);
        var inputs = new CpuPartitionAnalysisInputs(false, carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy);
        var analysis = new CpuPartitionPreparer().analyze(benchmarkContext(type, general,
                twoCopies ? general : dense, dense, dense, form, inputs));
        var plan = analysis.plan();
        var selection = (CpuRepresentationDecision.Selection) plan.representationDecisions()
                .getLast();
        int expectedCopies = twoCopies ? 2 : 1;
        assertAll(type + " " + form + " production selection",
                () -> assertEquals(CpuRepresentationDecision.SelectionReason.COPIED_PROFITABLE,
                        selection.reason()),
                () -> assertEquals(expectedCopies, plan.materializations().size()),
                () -> assertTrue(plan.materializations().stream().allMatch(copy ->
                        copy.sourceCarrier() == (defaultSegmentCarriers
                                ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                                : heapCarrier(type)))),
                () -> assertTrue(plan.materializations().stream().allMatch(copy ->
                        copy.copySpecialization().carrierPattern().equals(List.of(
                                defaultSegmentCarriers
                                        ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                                        : heapCarrier(type),
                                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)))),
                () -> assertEquals(plan.materializations().stream()
                                .map(CpuMaterializationPlan::identity).toList(),
                        selection.selected().materializations()),
                () -> assertTrue(plan.materializations().stream().allMatch(copy ->
                        copy.copySpecialization().equals(plan.materializations().getFirst()
                                .copySpecialization())
                                && copy.copyIr().encodedKernelIr().equals(plan.materializations()
                                        .getFirst().copyIr().encodedKernelIr()))),
                () -> assertTrue(!reuse || plan.materializations().getFirst().useCount() >= 2),
                () -> assertTrue(!reuse
                        || plan.materializations().getFirst().consumers().size() >= 2),
                () -> assertTrue(!reuse || plan.units().size() >= 2));
        if (twoCopies) assertAll("disjoint pair proof",
                () -> assertEquals(List.of(0, 1), plan.materializations().stream()
                        .map(CpuMaterializationPlan::sourceBoundaryIndex).toList()),
                () -> assertEquals(List.of(8, 9), plan.materializations().stream()
                        .map(CpuMaterializationPlan::workspaceRequirementId).toList()),
                () -> assertTrue(plan.representationDecisions().stream()
                        .noneMatch(CpuRepresentationDecision.Rejection.class::isInstance)));
        PreparedExecutable finalized;
        try {
            finalized = CpuPartitionFinalizerTest.finalizePreparedExecutable(analysis,
                    Optional.empty());
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(type + " " + form + " finalization", exception);
        }
        assertEquals("CpuPreparedPartitionExecutable", finalized.getClass().getSimpleName());
        return new ProductionCase(plan.materializations(), selection, plan.boundaryValues(),
                carriers, plan.representationUnits(), plan.units().size(), finalized);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> benchmarkContext(DataType type,
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor cDescriptor, TensorDescriptor outputDescriptor, String form,
            CpuPartitionAnalysisInputs inputs) {
        if (form.equals("copy-once-reuse")) return reuseBenchmarkContext(type, aDescriptor, bDescriptor,
                outputDescriptor, inputs);
        if (form.equals("disjoint-consumer-two-copy")) return disjointBenchmarkContext(
                aDescriptor, bDescriptor, outputDescriptor, inputs);
        ValueId a = new ValueId(0), b = new ValueId(1), c = new ValueId(2);
        ValueId first = new ValueId(3), second = new ValueId(4), output = new ValueId(5);
        List<CompiledNode> nodes;
        if (type == DataType.BOOL) {
            nodes = List.of(
                    new CompiledNode(new NodeId(0),
                            new Operation(BooleanLogicalKind.AND, NoOperationAttrs.INSTANCE),
                            List.of(a, b), List.of(first)),
                    new CompiledNode(new NodeId(1),
                            new Operation(BooleanLogicalKind.NOT, NoOperationAttrs.INSTANCE),
                            List.of(first), List.of(second)),
                    new CompiledNode(new NodeId(2),
                            new Operation(BooleanLogicalKind.OR, NoOperationAttrs.INSTANCE),
                            List.of(second, c), List.of(output)));
        } else {
            nodes = List.of(
                    new CompiledNode(new NodeId(0),
                            new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                            List.of(a, b), List.of(first)),
                    new CompiledNode(new NodeId(1),
                            new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                            List.of(first, b), List.of(second)),
                    new CompiledNode(new NodeId(2),
                            new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                            List.of(second, c), List.of(output)));
        }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var virtual = typedDescriptor(type, outputDescriptor.shape(),
                LayoutDescriptor.contiguous(outputDescriptor.shape()));
        List<TensorDescriptor> descriptors = List.of(aDescriptor, bDescriptor, cDescriptor,
                virtual, virtual, outputDescriptor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = descriptors.get(index);
            values.add(new GraphValue(id, descriptor));
            boolean produced = index >= 3;
            boolean consumed = index != 5;
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), index == 5));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> disjointBenchmarkContext(
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor outputDescriptor, CpuPartitionAnalysisInputs inputs) {
        ValueId a = new ValueId(0), b = new ValueId(1), na = new ValueId(2);
        ValueId ab = new ValueId(3), output = new ValueId(4);
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                        List.of(a), List.of(na)),
                new CompiledNode(new NodeId(1),
                        new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                        List.of(b), List.of(ab)),
                new CompiledNode(new NodeId(2),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(na, ab), List.of(output)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var virtual = typedDescriptor(DataType.FLOAT64, outputDescriptor.shape(),
                LayoutDescriptor.contiguous(outputDescriptor.shape()));
        List<TensorDescriptor> descriptors = List.of(aDescriptor, bDescriptor, virtual, virtual,
                outputDescriptor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = descriptors.get(index);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    index >= 2 ? Optional.of(partition) : Optional.empty(),
                    index == 4 ? List.of() : List.of(partition), index == 4));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> reuseBenchmarkContext(DataType type,
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor outputDescriptor, CpuPartitionAnalysisInputs inputs) {
        ValueId a = new ValueId(0), b = new ValueId(1);
        ValueId publishedIntermediate = new ValueId(2);
        ValueId second = new ValueId(3);
        ValueId output = new ValueId(4);
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(a, b), List.of(publishedIntermediate)),
                new CompiledNode(new NodeId(1),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(publishedIntermediate, b), List.of(second)),
                new CompiledNode(new NodeId(2),
                        new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                        List.of(second, a), List.of(output)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var virtual = typedDescriptor(type, outputDescriptor.shape(),
                LayoutDescriptor.contiguous(outputDescriptor.shape()));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < 5; index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = index == 0 ? aDescriptor
                    : index == 1 ? bDescriptor : index == 4 ? outputDescriptor : virtual;
            values.add(new GraphValue(id, descriptor));
            boolean produced = index >= 2;
            boolean consumed = index != 4;
            boolean publication = index == 2 || index == 4;
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), publication));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    private static TensorDescriptor typedDescriptor(DataType type, Shape shape,
            LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    private static CpuKernelSpecialization.CarrierAccess heapCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
            case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
        };
    }

    private static Object source(DataType type, int count, int seed) {
        return switch (type) {
            case FLOAT64 -> { double[] a = new double[count]; for (int i=0;i<count;i++)a[i]=i+seed; yield a; }
            case FLOAT32 -> { float[] a = new float[count]; for (int i=0;i<count;i++)a[i]=i+seed; yield a; }
            case INT32 -> { int[] a = new int[count]; for (int i=0;i<count;i++)a[i]=i+seed; yield a; }
            case INT64 -> { long[] a = new long[count]; for (int i=0;i<count;i++)a[i]=i+seed; yield a; }
            case BOOL -> { byte[] a = new byte[count]; for (int i=0;i<count;i++)a[i]=(byte)((i+seed)&1); yield a; }
            case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
        };
    }

    private static Object carrierSource(DataType type, Object array,
            CpuKernelSpecialization.CarrierAccess carrier) {
        return carrier == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? segment(type, array) : array;
    }

    private static void invokeCopies(DataType type, List<MethodHandle> handles,
            List<CpuMaterializationPlan> plans, Object first, Object second,
            MemorySegment left, MemorySegment right, List<long[]> geometries, int count)
            throws Throwable {
        assertEquals(handles.size(), plans.size());
        for (int index = 0; index < plans.size(); index++) {
            CpuMaterializationPlan plan = plans.get(index);
            Object source = switch (plan.sourceBoundaryIndex()) {
                case 0 -> first;
                case 1 -> second;
                default -> throw new IllegalArgumentException("unmeasured copied boundary");
            };
            invoke(type, handles.get(index), source, index == 0 ? left : right,
                    geometries.get(index), count);
        }
    }

    private static void invoke(DataType type, MethodHandle handle, Object source,
            MemorySegment target, long[] geometry, int count) throws Throwable {
        if (source instanceof MemorySegment segment) {
            handle.invokeExact(segment, target, geometry, 0L, (long) count);
            return;
        }
        switch (type) {
            case FLOAT64 -> handle.invokeExact((double[]) source, target, geometry, 0L, (long) count);
            case FLOAT32 -> handle.invokeExact((float[]) source, target, geometry, 0L, (long) count);
            case INT32 -> handle.invokeExact((int[]) source, target, geometry, 0L, (long) count);
            case INT64 -> handle.invokeExact((long[]) source, target, geometry, 0L, (long) count);
            case BOOL -> handle.invokeExact((byte[]) source, target, geometry, 0L, (long) count);
            case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
        }
    }

    private static MethodHandle oracleHandle(DataType type,
            CpuKernelSpecialization.CarrierAccess sourceCarrier) throws ReflectiveOperationException {
        String method;
        Class<?> sourceType;
        if (sourceCarrier == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && type == DataType.FLOAT64) {
            method = "oracleDoubleSegment";
            sourceType = MemorySegment.class;
        } else {
            method = switch (type) {
                case FLOAT64 -> "oracleDoubleArray";
                case FLOAT32 -> "oracleFloatArray";
                case INT32 -> "oracleIntArray";
                case INT64 -> "oracleLongArray";
                case BOOL -> "oracleByteArray";
                case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
            };
            sourceType = switch (sourceCarrier) {
                case DOUBLE_ARRAY -> double[].class;
                case FLOAT_ARRAY -> float[].class;
                case INT_ARRAY -> int[].class;
                case LONG_ARRAY -> long[].class;
                case BYTE_ARRAY -> byte[].class;
                default -> throw new IllegalArgumentException(
                        "no clean-Java oracle for " + type + " from " + sourceCarrier);
            };
        }
        return MethodHandles.lookup().findStatic(CpuRepresentationPlannerTest.class, method,
                MethodType.methodType(void.class, sourceType, MemorySegment.class,
                        long[].class, long.class, long.class));
    }

    private static void oracleDoubleArray(double[] source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            double value = source[(int) geometry[(int) (index * 2)]];
            target.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                    targetAddress * Double.BYTES, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static void oracleFloatArray(float[] source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            float value = source[(int) geometry[(int) (index * 2)]];
            target.set(ValueLayout.JAVA_FLOAT_UNALIGNED,
                    targetAddress * Float.BYTES, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static void oracleIntArray(int[] source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            int value = source[(int) geometry[(int) (index * 2)]];
            target.set(ValueLayout.JAVA_INT_UNALIGNED,
                    targetAddress * Integer.BYTES, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static void oracleLongArray(long[] source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            long value = source[(int) geometry[(int) (index * 2)]];
            target.set(ValueLayout.JAVA_LONG_UNALIGNED,
                    targetAddress * Long.BYTES, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static void oracleByteArray(byte[] source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            byte value = source[(int) geometry[(int) (index * 2)]];
            target.set(ValueLayout.JAVA_BYTE, targetAddress, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static void oracleDoubleSegment(MemorySegment source, MemorySegment target,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        long targetAddress = geometry[(int) (start * 2 + 1)];
        long index = start;
        do {
            long sourceAddress = geometry[(int) (index * 2)];
            double value = source.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                    sourceAddress * Double.BYTES);
            target.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                    targetAddress * Double.BYTES, value);
            index++;
            targetAddress++;
        } while (index < end);
    }

    private static long consume(String form, DataType type, Object first, Object second,
            Object third, MemorySegment intermediate, MemorySegment output, int count) {
        if (form.equals("copy-once-reuse")) return consumeReuse(type, first, second,
                intermediate, output, count);
        long checksum = 0;
        for (int pass = 0; pass < 12; pass++) for (int index = 0; index < count; index++) {
            long offset = (long) index * type.byteWidth();
            checksum += switch (type) {
                case FLOAT64 -> { double a = readDouble(first, index);
                    double b = readDouble(second, index);
                    double value = form.equals("disjoint-consumer-two-copy")
                            ? -a + Math.abs(b)
                            : ((a + b) + b) * readDouble(third, index);
                    output.set(ValueLayout.JAVA_DOUBLE, offset, value);
                    yield Double.doubleToRawLongBits(value); }
                case FLOAT32 -> { float a = readFloat(first, index), b = readFloat(second, index);
                    float value = ((a + b) + b) * readFloat(third, index);
                    output.set(ValueLayout.JAVA_FLOAT, offset, value);
                    yield Float.floatToRawIntBits(value); }
                case INT32 -> { int a = readInt(first, index), b = readInt(second, index);
                    int value = ((a + b) + b) * readInt(third, index);
                    output.set(ValueLayout.JAVA_INT, offset, value); yield value; }
                case INT64 -> { long a = readLong(first, index), b = readLong(second, index);
                    long value = ((a + b) + b) * readLong(third, index);
                    output.set(ValueLayout.JAVA_LONG, offset, value); yield value; }
                case BOOL -> { byte a = readByte(first, index), b = readByte(second, index);
                    byte value = (byte) (((a & b) == 0 ? 1 : 0) | readByte(third, index));
                    output.set(ValueLayout.JAVA_BYTE, offset, value); yield value; }
                case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
            };
        }
        return checksum;
    }

    private static long consumeDirect(String form, DataType type, Object first, Object second,
            Object third, MemorySegment intermediate, MemorySegment output,
            List<long[]> geometries, int count) {
        if (form.equals("copy-once-reuse")) return consumeDirectReuse(type, first, second,
                intermediate, output, geometries.getFirst(), count);
        long[] firstGeometry = geometries.getFirst();
        long[] secondGeometry = form.equals("disjoint-consumer-two-copy") ? geometries.get(1) : null;
        long checksum = 0;
        for (int pass = 0; pass < 12; pass++) for (int index = 0; index < count; index++) {
            int firstIndex = mappedIndex(firstGeometry, index);
            int secondIndex = secondGeometry == null ? index : mappedIndex(secondGeometry, index);
            long offset = (long) index * type.byteWidth();
            checksum += switch (type) {
                case FLOAT64 -> { double a = readDouble(first, firstIndex);
                    double b = readDouble(second, secondIndex);
                    double value = form.equals("disjoint-consumer-two-copy")
                            ? -a + Math.abs(b)
                            : ((a + b) + b) * readDouble(third, index);
                    output.set(ValueLayout.JAVA_DOUBLE, offset, value);
                    yield Double.doubleToRawLongBits(value); }
                case FLOAT32 -> { float a = readFloat(first, firstIndex);
                    float b = readFloat(second, secondIndex);
                    float value = ((a + b) + b) * readFloat(third, index);
                    output.set(ValueLayout.JAVA_FLOAT, offset, value);
                    yield Float.floatToRawIntBits(value); }
                case INT32 -> { int a = readInt(first, firstIndex);
                    int b = readInt(second, secondIndex);
                    int value = ((a + b) + b) * readInt(third, index);
                    output.set(ValueLayout.JAVA_INT, offset, value); yield value; }
                case INT64 -> { long a = readLong(first, firstIndex);
                    long b = readLong(second, secondIndex);
                    long value = ((a + b) + b) * readLong(third, index);
                    output.set(ValueLayout.JAVA_LONG, offset, value); yield value; }
                case BOOL -> { byte a = readByte(first, firstIndex);
                    byte b = readByte(second, secondIndex);
                    byte value = (byte) (((a & b) == 0 ? 1 : 0) | readByte(third, index));
                    output.set(ValueLayout.JAVA_BYTE, offset, value); yield value; }
                case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
            };
        }
        return checksum;
    }

    private static long consumeReuse(DataType type, Object first, Object second,
            MemorySegment intermediate, MemorySegment output, int count) {
        long checksum = 0;
        for (int pass = 0; pass < 12; pass++) {
            for (int index = 0; index < count; index++) {
                long offset = (long) index * Double.BYTES;
                intermediate.set(ValueLayout.JAVA_DOUBLE, offset,
                        readDouble(first, index) + readDouble(second, index));
            }
            for (int index = 0; index < count; index++) {
                long offset = (long) index * Double.BYTES;
                double value = (intermediate.get(ValueLayout.JAVA_DOUBLE, offset)
                        + readDouble(second, index)) * readDouble(first, index);
                output.set(ValueLayout.JAVA_DOUBLE, offset, value);
                checksum += Double.doubleToRawLongBits(value);
            }
        }
        return checksum;
    }

    private static long consumeDirectReuse(DataType type, Object first, Object second,
            MemorySegment intermediate, MemorySegment output, long[] geometry, int count) {
        if (type != DataType.FLOAT64) throw new IllegalArgumentException("reuse is FLOAT64");
        long checksum = 0;
        for (int pass = 0; pass < 12; pass++) {
            for (int index = 0; index < count; index++) {
                long offset = (long) index * Double.BYTES;
                intermediate.set(ValueLayout.JAVA_DOUBLE, offset,
                        readDouble(first, mappedIndex(geometry, index)) + readDouble(second, index));
            }
            for (int index = 0; index < count; index++) {
                long offset = (long) index * Double.BYTES;
                double value = (intermediate.get(ValueLayout.JAVA_DOUBLE, offset)
                        + readDouble(second, index))
                        * readDouble(first, mappedIndex(geometry, index));
                output.set(ValueLayout.JAVA_DOUBLE, offset, value);
                checksum += Double.doubleToRawLongBits(value);
            }
        }
        return checksum;
    }

    private static int mappedIndex(long[] geometry, int logicalIndex) {
        return Math.toIntExact(geometry[logicalIndex * 2]);
    }

    private static MemorySegment segment(DataType type, Object array) {
        return switch (type) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) array);
            case FLOAT32 -> MemorySegment.ofArray((float[]) array);
            case INT32 -> MemorySegment.ofArray((int[]) array);
            case INT64 -> MemorySegment.ofArray((long[]) array);
            case BOOL -> MemorySegment.ofArray((byte[]) array);
            case BFLOAT16 -> throw new IllegalArgumentException("BFLOAT16 is not admitted");
        };
    }

    private static double readDouble(Object source, int index) {
        return source instanceof MemorySegment segment
                ? segment.get(ValueLayout.JAVA_DOUBLE, (long) index * 8)
                : ((double[]) source)[index];
    }

    private static float readFloat(Object source, int index) {
        return source instanceof MemorySegment segment
                ? segment.get(ValueLayout.JAVA_FLOAT, (long) index * 4)
                : ((float[]) source)[index];
    }

    private static int readInt(Object source, int index) {
        return source instanceof MemorySegment segment
                ? segment.get(ValueLayout.JAVA_INT, (long) index * 4)
                : ((int[]) source)[index];
    }

    private static long readLong(Object source, int index) {
        return source instanceof MemorySegment segment
                ? segment.get(ValueLayout.JAVA_LONG, (long) index * 8)
                : ((long[]) source)[index];
    }

    private static byte readByte(Object source, int index) {
        return source instanceof MemorySegment segment
                ? segment.get(ValueLayout.JAVA_BYTE, index)
                : ((byte[]) source)[index];
    }

    private static void retainInvokedCopy(GeneratedCopy copy, MethodHandle oracle,
            ProductionCase production, DataType type, String form, int fork, int copyIndex)
            throws Exception {
        assertEquals(copy.specialization().entryType(), copy.handle().type());
        assertEquals(copy.handle().type(), oracle.type());
        assertEquals(copy.handle().type(), MethodType.methodType(
                copy.owner().getDeclaredMethods()[0].getReturnType(),
                copy.owner().getDeclaredMethods()[0].getParameterTypes()));
        String constants = new String(copy.bytes(), StandardCharsets.ISO_8859_1);
        for (String forbidden : List.of("nativeOrder", "withOrder", "CpuRepresentation",
                "CpuMaterializationPlan", "CopyCall", "java/lang/reflect", "java/lang/Integer",
                "java/lang/Long", "java/lang/Double")) {
            assertFalse(constants.contains(forbidden), forbidden);
        }
        Path root = EVIDENCE.resolve("invoked/fork-" + fork + '/' + form + '-' + type
                + "/copy-" + copyIndex);
        Files.createDirectories(root);
        Files.write(root.resolve("generated.class"), copy.bytes());
        retainJavap(root.resolve("generated.class"), root.resolve("generated.javap.txt"));
        String oracleResource = '/' + CpuRepresentationPlannerTest.class.getName()
                .replace('.', '/') + ".class";
        try (var stream = CpuRepresentationPlannerTest.class.getResourceAsStream(oracleResource)) {
            assertNotNull(stream, oracleResource);
            Files.write(root.resolve("oracle.class"), stream.readAllBytes());
        }
        retainJavap(root.resolve("oracle.class"), root.resolve("oracle.javap.txt"));
        var oracleInfo = MethodHandles.lookup().revealDirect(oracle);
        Files.writeString(root.resolve("method-type.txt"), copy.handle().type() + "\n");
        Files.writeString(root.resolve("owner.txt"), copy.owner().getName() + "\n");
        Files.writeString(root.resolve("oracle-owner.txt"),
                oracleInfo.getDeclaringClass().getName() + '#' + oracleInfo.getName() + "\n");
        Files.writeString(root.resolve("specialization.txt"), copy.specialization() + "\n");
        Files.writeString(root.resolve("ir.txt"), copy.ir() + "\n");
        Files.writeString(root.resolve("production-selection.txt"), production.selection() + "\n");
        Files.writeString(root.resolve("production-materializations.txt"),
                production.materializations() + "\n");
        Files.writeString(root.resolve("production-boundaries.txt"),
                production.boundaryValues() + "\n");
        Files.writeString(root.resolve("production-carriers.txt"),
                production.carriers() + "\n");
        Files.writeString(root.resolve("production-finalizer.txt"),
                "units=" + production.unitCount() + "\nclass="
                        + production.finalized().getClass().getName() + "\n");
        var partition = (CpuPreparedPartitionExecutable) production.finalized();
        assertEquals(production.representationUnits().size(), partition.children().size());
        for (int index = 0; index < partition.children().size(); index++) {
            int childIndex = index;
            var child = partition.children().get(childIndex);
            var represented = production.representationUnits().get(childIndex);
            assertEquals(represented.portablePlan().specialization(),
                    child.artifact().specialization());
            byte[] consumerBytes = child.artifact().classBytes();
            java.lang.classfile.ClassFile.of().parse(consumerBytes);
            String consumerConstants = new String(consumerBytes, StandardCharsets.ISO_8859_1);
            for (String forbidden : List.of("CpuRepresentation", "CpuMaterializationPlan",
                    "CopyCall", "java/lang/reflect")) {
                assertFalse(consumerConstants.contains(forbidden), forbidden);
            }
            Path consumerClass = root.resolve("consumer-" + childIndex + ".class");
            Files.write(consumerClass, consumerBytes);
            retainJavap(consumerClass, root.resolve("consumer-" + childIndex + ".javap.txt"));
            Files.writeString(root.resolve("consumer-" + childIndex + "-specialization.txt"),
                    child.artifact().specialization() + "\n");
            Files.writeString(root.resolve("consumer-" + childIndex + "-ir.txt"),
                    represented.portablePlan().kernelIr() + "\n");
        }
        CpuMaterializationPlan first = production.materializations().get(copyIndex);
        assertAll(
                () -> assertEquals(first.copySpecialization(), copy.specialization()),
                () -> assertEquals(first.copyIr().encodedKernelIr(), copy.ir()),
                () -> assertEquals(first.copySpecialization().entryType(), copy.handle().type()));
    }

    private static void retainJavap(Path classFile, Path output) throws Exception {
        Process javap = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin",
                "javap").toString(), "-c", "-v", "-p", classFile.toString())
                .redirectErrorStream(true).start();
        String inspection = new String(javap.getInputStream().readAllBytes(),
                StandardCharsets.UTF_8);
        assertEquals(0, javap.waitFor(), inspection);
        Files.writeString(output, inspection);
    }

    private record GeneratedCopy(MethodHandle handle, Class<?> owner, byte[] bytes,
            CpuKernelSpecialization specialization,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr ir) { }

    private record ProductionCase(List<CpuMaterializationPlan> materializations,
            CpuRepresentationDecision.Selection selection, List<ValueId> boundaryValues,
            List<CpuKernelSpecialization.CarrierAccess> carriers,
            List<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                    .RepresentationUnitPlan> representationUnits, int unitCount,
            PreparedExecutable finalized) {
        private ProductionCase {
            materializations = List.copyOf(materializations);
            boundaryValues = List.copyOf(boundaryValues);
            carriers = List.copyOf(carriers);
            representationUnits = List.copyOf(representationUnits);
        }
    }

    private static long median(long[] values) {
        long[] copy=values.clone(); Arrays.sort(copy); return copy[copy.length/2];
    }

    private static void appendSummary(StringBuilder target, String kind, List<String> lines) {
        for (String form : List.of("all", "one-copy", "same-binary-single-copy",
                "disjoint-consumer-two-copy", "copy-once-reuse")) {
            double[] values = lines.stream().filter(line -> form.equals("all")
                            || line.split(",", -1)[1].equals(form))
                    .mapToDouble(line -> Double.parseDouble(
                            line.substring(line.lastIndexOf(',') + 1))).sorted().toArray();
            target.append("AGGREGATE,").append(kind).append(',').append(form).append(',')
                    .append(String.format(Locale.ROOT, "%.9f", values[values.length / 2]))
                    .append(',').append(String.format(Locale.ROOT, "%.9f",
                            values[values.length - 1])).append('\n');
        }
    }

    private static String manifest(Path root) throws Exception {
        var files = Files.walk(root).filter(Files::isRegularFile)
                .filter(path -> !path.getFileName().toString().equals("manifest.sha256"))
                .sorted().toList();
        var result = new StringBuilder();
        for (Path file : files) result.append(java.util.HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file))))
                .append("  ").append(root.relativize(file)).append('\n');
        return result.toString();
    }

    private static String sha256(byte[] bytes) throws Exception {
        return java.util.HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
