package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.route.CpuLocalTuningBatch;
import io.github.pho001.synaptik.backend.cpu.internal.route.CpuLocalTuningDecision;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Behavioral finite-domain and exact frozen-realization regressions without native providers. */
final class CpuFiniteLocalRealizationTest {
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy LOCAL =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 0, 0, 0, 1,
                    Long.MAX_VALUE, 0, 0);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy COMPLETE =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 3, 1, 1,
                    Long.MAX_VALUE, 0, 0);

    @Test void portableTailOnlyDomainNeedsNoNativeQualification() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (int k : new int[] {0, 1, 2}) {
            var context = context(type, 1, k, 1, inputs(1, 1, 1, LOCAL, Optional.empty()));
            var preparer = new CpuPartitionPreparer();
            var batch = preparer.localTuningBatch(context).orElseThrow();
            assertEquals(4, batch.candidates().size(), "tail-only fixed forms are implemented legal alternatives");
            assertTrue(batch.workload().qualification().isEmpty());
            for (var selected : batch.candidates()) {
                var decision = new CpuLocalTuningDecision(batch.schemaVersion(), batch.workload(), selected.identity(), batch);
                var frozen = withInputs(context, inputs(1, 1, 1, COMPLETE, Optional.of(decision)));
                var complete = preparer.completePlanCandidates(frozen);
                assertTrue(complete.complete());
                assertEquals(1, complete.identities().size());
                var plan = preparer.analyzeSelected(frozen,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, complete.identities().getFirst(), List.of(1))).plan();
                assertEquals(selected.identity(), plan.selectedLocalTuningCandidate().orElseThrow());
                executeAndCompare(type, 1, k, 1, plan.units().getFirst());
            }
        }
    }
    @Test void legalRangesRespectConfigurationCapacityAndChunkBoundary() {
        var preparer = new CpuPartitionPreparer();
        var two = preparer.localTuningBatch(context(DataType.FLOAT64, 7, 2, 1,
                inputs(4, 2, 4, LOCAL, Optional.empty()))).orElseThrow();
        assertEquals(List.of(2, 1, 1, 2, 1, 1), two.candidates().stream()
                .map(candidate -> ((CpuLocalTuningBatch.PortableIdentity) candidate.identity()).rangeCount()).toList());
        var one = preparer.localTuningBatch(context(DataType.FLOAT64, 7, 2, 1,
                inputs(4, 1, 4, LOCAL, Optional.empty()))).orElseThrow();
        assertEquals(4, one.candidates().size());
        assertEquals(1, ((CpuLocalTuningBatch.PortableIdentity) one.candidates().getFirst().identity()).rangeCount());
    }

    @Test void aggregateLimitRejectsTheWholeDomainBeforeExpansion() {
        int maximum = CpuLocalTuningBatch.MAXIMUM_CANDIDATES;
        var preparer = new CpuPartitionPreparer();
        int parallelism = maximum / 4;
        assertThrows(IllegalArgumentException.class, () -> preparer.localTuningBatch(
                context(DataType.FLOAT32, 2 * (parallelism + 1), 2, 1,
                        inputs(parallelism + 1, parallelism + 1, 1, LOCAL, Optional.empty()))));
        var boundary = preparer.localTuningBatch(context(DataType.FLOAT32, 2 * parallelism, 2, 1,
                inputs(parallelism, parallelism, 1, LOCAL, Optional.empty()))).orElseThrow();
        assertEquals(maximum, boundary.candidates().size());
        assertEquals(maximum, boundary.candidates().stream().map(CpuLocalTuningBatch.Candidate::identity).distinct().count());
    }

    @Test void everyFrozenPortableChoiceActuallyExecutesAfterCompletePolicyChange() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int lanes = type == DataType.FLOAT32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                    : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
            int m = 3, k = 3, n = 2 * lanes + 1;
            var context = context(type, m, k, n, inputs(3, 3, 1, LOCAL, Optional.empty()));
            var preparer = new CpuPartitionPreparer();
            var batch = preparer.localTuningBatch(context).orElseThrow();
            assertEquals(12, batch.candidates().size(), "four legal forms times three legal ranges");
            assertEquals(4, batch.candidates().stream().map(candidate -> candidate.realization().orElseThrow()
                    .units().getFirst().portablePlan().specialization().matmulIr().orElseThrow().realization())
                    .distinct().count());
            for (var candidate : batch.candidates()) {
                var decision = new CpuLocalTuningDecision(batch.schemaVersion(), batch.workload(), candidate.identity(), batch);
                var phaseTwo = withInputs(context, inputs(3, 3, 1, COMPLETE, Optional.of(decision)));
                var complete = preparer.completePlanCandidates(phaseTwo);
                assertEquals(1, complete.identities().size());
                var realized = preparer.analyzeSelected(phaseTwo,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, complete.identities().getFirst(), List.of(1))).plan();
                var unit = realized.units().getFirst();
                var identity = (CpuLocalTuningBatch.PortableIdentity) candidate.identity();
                assertEquals(candidate.identity(), realized.selectedLocalTuningCandidate().orElseThrow());
                assertEquals(identity.rangeCount(), unit.selectedRangeCount());
                assertEquals(identity.vectorSpeciesBits(), unit.vectorSpeciesBitSize());
                assertEquals(candidate.realization().orElseThrow().units().getFirst().portablePlan().specialization()
                        .matmulIr().orElseThrow().realization(), unit.portablePlan().specialization().matmulIr().orElseThrow().realization());
                executeAndCompare(type, m, k, n, unit);
            }
        }
    }

    @Test void everyExistingPromotedNumericTupleExecutesItsFrozenScalarAndTileChoices() throws Throwable {
        var floating = List.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3,
                DataType.FLOAT8_E5M2, DataType.FLOAT32, DataType.FLOAT64);
        var integral = List.of(DataType.INT32, DataType.INT64);
        for (var group : List.of(floating, integral)) for (var left : group) for (var right : group) {
            var result = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteNumeric(left, right);
            var context = mixedContext(left, right, result);
            var preparer = new CpuPartitionPreparer();
            var batch = preparer.localTuningBatch(context).orElseThrow();
            boolean vector = left == right && (left == DataType.FLOAT32 || left == DataType.FLOAT64
                    || left == DataType.INT32 || left == DataType.INT64);
            assertEquals(vector ? 4 : 2, batch.candidates().size(), left + " x " + right);
            for (var candidate : batch.candidates()) {
                var selected = new CpuLocalTuningDecision(batch.schemaVersion(), batch.workload(), candidate.identity(), batch);
                var phaseTwo = withInputs(context, inputs(1, 1, 1, COMPLETE, Optional.of(selected)));
                var complete = preparer.completePlanCandidates(phaseTwo);
                assertEquals(1, complete.identities().size());
                var unit = preparer.analyzeSelected(phaseTwo, new CpuPartitionPreparer.SelectedCompletePlan(
                        5, complete.identities().getFirst(), List.of(1))).plan().units().getFirst();
                var route = unit.portablePlan();
                var generator = new CpuClassFileKernelGenerator();
                var generated = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
                Object a = numericValues(left, 2, 3), b = numericValues(right, 4, 5), output = numericValues(result, 0);
                generated.entryPoint().invokeWithArguments(segment(a), segment(b), segment(output),
                        unit.matmulGeometry().orElseThrow().pack(0, 0, 0), 0L, unit.elementCount());
                double expected = result == DataType.FLOAT8_E4M3 || result == DataType.FLOAT8_E5M2 ? 24 : 23;
                assertEquals(expected, numericFirst(result, output), 0.0, left + " x " + right);
            }
        }
    }
    @Test void staleSourceOrCapacityNeverFallsThroughToProductionHeuristics() {
        var preparer = new CpuPartitionPreparer();
        var source = context(DataType.FLOAT32, 2, 3, 2, inputs(2, 2, 1, LOCAL, Optional.empty()));
        var batch = preparer.localTuningBatch(source).orElseThrow();
        var chosen = batch.candidates().getLast();
        var decision = new CpuLocalTuningDecision(batch.schemaVersion(), batch.workload(), chosen.identity(), batch);
        assertThrows(IllegalArgumentException.class, () -> preparer.analyze(withInputs(source,
                inputs(2, 1, 1, LOCAL, Optional.of(decision)))));
        assertThrows(IllegalArgumentException.class, () -> preparer.analyze(context(DataType.FLOAT32,
                2, 4, 2, inputs(2, 2, 1, LOCAL, Optional.of(decision)))));
        assertThrows(IllegalArgumentException.class, () -> preparer.analyze(withInputs(source,
                inputs(2, 2, 1, COMPLETE, Optional.of(decision)))),
                "only authenticated Phase-2 selection may transport the materialization policy");
    }

    @Test void phaseTwoScoringTransportCannotRestoreDisabledCopiesOrAnOlderWorkspaceCeiling() {
        var nativeConfig = CpuOpenBlasRouteSelectorTest.defaultConfig();
        var source = context(DataType.FLOAT32, 2, 3, 2, qualifiedInputs(
                inputs(1, 1, 1, LOCAL, Optional.empty()), nativeConfig));
        var preparer = new CpuPartitionPreparer();
        var batch = preparer.localTuningBatch(source).orElseThrow();
        var candidate = batch.candidates().stream().filter(value -> value.route()
                == CpuLocalTuningBatch.RouteKind.OPENBLAS && value.openBlasPlan()
                    .map(plan -> !plan.workspaceRequirements().isEmpty()).orElse(false))
                .findFirst().orElseThrow();
        var decision = new CpuLocalTuningDecision(batch.schemaVersion(), batch.workload(),
                candidate.identity(), batch);
        var valid = withInputs(source, qualifiedInputs(
                inputs(1, 1, 1, COMPLETE, Optional.of(decision)), nativeConfig));
        var identity = preparer.completePlanCandidates(valid).identities().getFirst();
        var selected = new CpuPartitionPreparer.SelectedCompletePlan(5, identity, List.of(1));
        var validPlan = preparer.analyzeSelected(valid, selected).plan();
        long copiedBytes = validPlan.openBlasPlan().orElseThrow().workspaceRequirements().stream()
                .mapToLong(requirement -> requirement.byteSize()).sum();
        assertTrue(copiedBytes > 0, "the regression needs a genuinely selected copy workspace");
        var tooSmall = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 3, 1, 1, copiedBytes - 1, 0, 0);
        for (var policy : List.of(CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, tooSmall)) {
            var changed = withInputs(source, qualifiedInputs(
                    inputs(1, 1, 1, policy, Optional.of(decision)), nativeConfig));
            assertThrows(IllegalArgumentException.class, () -> preparer.analyzeSelected(changed, selected));
        }
    }

    private static void executeAndCompare(DataType type, int m, int k, int n,
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit) throws Throwable {
        var route = unit.portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var generated = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] a = new double[m * k], b = new double[k * n], expected = new double[m * n];
        for (int i = 0; i < a.length; i++) a[i] = i % 7 - 3;
        for (int i = 0; i < b.length; i++) b[i] = i % 5 - 2;
        for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
            for (int inner = 0; inner < k; inner++) expected[row * n + column] += a[row * k + inner] * b[inner * n + column];
        Object left, right, output;
        if (type == DataType.FLOAT32) {
            float[] af = new float[a.length], bf = new float[b.length];
            for (int i = 0; i < a.length; i++) af[i] = (float) a[i];
            for (int i = 0; i < b.length; i++) bf[i] = (float) b[i];
            left = af; right = bf; output = new float[m * n];
        } else { left = a; right = b; output = new double[m * n]; }
        MemorySegment as = segment(left), bs = segment(right), out = segment(output);
        long[] geometry = unit.matmulGeometry().orElseThrow().pack(0, 0, 0);
        int ranges = unit.selectedRangeCount();
        for (int range = ranges - 1; range >= 0; range--) {
            long start = unit.elementCount() * range / ranges;
            long end = unit.elementCount() * (range + 1L) / ranges;
            generated.entryPoint().invokeWithArguments(as, bs, out, geometry, start, end);
        }
        for (int i = 0; i < expected.length; i++) {
            double actual = output instanceof float[] values ? values[i] : ((double[]) output)[i];
            assertEquals(Double.doubleToRawLongBits(expected[i]), Double.doubleToRawLongBits(actual),
                    type + " " + route.specialization().matmulIr().orElseThrow().realization() + " cell=" + i);
        }
    }

    private static MemorySegment segment(Object array) {
        return switch (array) {
            case float[] values -> MemorySegment.ofArray(values);
            case double[] values -> MemorySegment.ofArray(values);
            case int[] values -> MemorySegment.ofArray(values);
            case long[] values -> MemorySegment.ofArray(values);
            case short[] values -> MemorySegment.ofArray(values);
            case byte[] values -> MemorySegment.ofArray(values);
            default -> throw new AssertionError("unexpected numeric carrier");
        };
    }

    private static Object numericValues(DataType type, int... values) {
        return switch (type) {
            case FLOAT32 -> { float[] result = new float[values.length];
                for (int i = 0; i < values.length; i++) result[i] = values[i]; yield result; }
            case FLOAT64 -> { double[] result = new double[values.length];
                for (int i = 0; i < values.length; i++) result[i] = values[i]; yield result; }
            case INT32 -> values.clone();
            case INT64 -> { long[] result = new long[values.length];
                for (int i = 0; i < values.length; i++) result[i] = values[i]; yield result; }
            case BFLOAT16 -> { short[] result = new short[values.length];
                for (int i = 0; i < values.length; i++) result[i] =
                        io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(values[i]); yield result; }
            case FLOAT16 -> { short[] result = new short[values.length];
                for (int i = 0; i < values.length; i++) result[i] = Float.floatToFloat16(values[i]); yield result; }
            case FLOAT8_E4M3, FLOAT8_E5M2 -> { byte[] result = new byte[values.length];
                for (int i = 0; i < values.length; i++) result[i] = type == DataType.FLOAT8_E4M3
                        ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromFloat(values[i])
                        : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromFloat(values[i]); yield result; }
            case BOOL -> throw new AssertionError("BOOL is not a numeric MATMUL carrier");
        };
    }

    private static double numericFirst(DataType type, Object output) {
        return switch (type) {
            case FLOAT32 -> ((float[]) output)[0]; case FLOAT64 -> ((double[]) output)[0];
            case INT32 -> ((int[]) output)[0]; case INT64 -> ((long[]) output)[0];
            case BFLOAT16 -> io.github.pho001.synaptik.model.datatype.BFloat16Bits.toFloat(((short[]) output)[0]);
            case FLOAT16 -> Float.float16ToFloat(((short[]) output)[0]);
            case FLOAT8_E4M3 -> io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.toFloat(((byte[]) output)[0]);
            case FLOAT8_E5M2 -> io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.toFloat(((byte[]) output)[0]);
            case BOOL -> throw new AssertionError("BOOL is not a numeric MATMUL carrier");
        };
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> mixedContext(DataType left, DataType right, DataType result) {
        var base = context(DataType.FLOAT32, 1, 2, 1, inputs(1, 1, 1, LOCAL, Optional.empty()));
        var types = List.of(left, right, result);
        var values = new java.util.ArrayList<io.github.pho001.synaptik.model.graph.GraphValue>();
        var memory = new java.util.ArrayList<io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement>();
        for (int index = 0; index < 3; index++) {
            var old = base.values().get(index);
            var descriptor = new TensorDescriptor(types.get(index), old.descriptor().shape(), old.descriptor().layout(), false);
            values.add(new io.github.pho001.synaptik.model.graph.GraphValue(old.id(), descriptor));
            var requirement = base.memoryRequirements().get(index);
            memory.add(new io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement(old.id(), descriptor,
                    requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput()));
        }
        return new PrepareContext<>(base.partitionDag(), values, memory, base.constants(), base.backendInputs(), base.scanActivityPlans());
    }

    private static CpuPartitionAnalysisInputs inputs(int configured, int capacity, long minimum,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy, Optional<CpuLocalTuningDecision> decision) {
        return new CpuPartitionAnalysisInputs(false, List.of(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        configured, capacity, minimum), policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, List.of(),
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.DISABLED,
                CpuLocalTuningBatch.HardwareIdentity.UNSPECIFIED, CpuLocalTuningBatch.WorkloadCohort.DEFAULT, decision);
    }

    private static CpuPartitionAnalysisInputs qualifiedInputs(CpuPartitionAnalysisInputs source,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig nativeConfig) {
        // Caller inputs are storage-unknown; the CPU owns its native output allocation.
        var callerUnknown = new CpuPartitionAnalysisInputs.BoundaryStorageFact(false, 1);
        var storageFacts = List.of(callerUnknown, callerUnknown,
                new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, 8));
        return new CpuPartitionAnalysisInputs(source.loweringManifestEnabled(), source.carrierPattern(),
                source.portableExecution(), source.materializationPolicy(), source.conv2dMaterializedSuffixUnit(),
                source.partialReductionEvidence(), storageFacts, nativeConfig,
                source.cpuHardwareIdentity(), source.workloadCohort(), source.localTuningDecision());
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, int m, int k, int n,
            CpuPartitionAnalysisInputs inputs) {
        return CpuOpenBlasRouteSelectorTest.context(type, dense(Shape.of(m, k)), dense(Shape.of(k, n)),
                dense(Shape.of(m, n)), inputs);
    }

    private static TensorDescriptor dense(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(
            PrepareContext<CpuPartitionAnalysisInputs> context, CpuPartitionAnalysisInputs inputs) {
        return new PrepareContext<>(context.partitionDag(), context.values(), context.memoryRequirements(),
                context.constants(), inputs, context.scanActivityPlans());
    }
}
