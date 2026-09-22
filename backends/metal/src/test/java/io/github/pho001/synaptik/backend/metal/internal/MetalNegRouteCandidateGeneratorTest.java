package io.github.pho001.synaptik.backend.metal.internal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

class MetalNegRouteCandidateGeneratorTest {
    @Test
    void candidateDomainsAndBudgetPrefixesPreserveTheCurrentSafeChoice() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload singleton = workload(device, 100, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            Generated generated = generated(singleton, 2);
            assertEquals(List.of(
                    MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                    MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated.batch().candidates());
            assertEquals(List.of(MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG),
                    new MetalNegRouteCandidateGenerator().generate(
                            singleton.context(), generated.analysis().plan(), 1).candidates());

            Workload oversized = workload(device, 200, Shape.of(0x1_0000_0000L), false,
                    Optional.empty(), true, false, 1);
            assertEquals(List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(oversized, 2).batch().candidates());
            Workload chain = workload(device, 300, Shape.of(4), false,
                    Optional.empty(), true, false, 2);
            assertEquals(List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(chain, 2).batch().candidates());
            assertThrows(IllegalArgumentException.class,
                    () -> new MetalNegRouteCandidateGenerator().generate(
                            singleton.context(), generated.analysis().plan(), 0));
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    @Test
    void signaturesUseStructuralPositionsAndCoverIndependentCurrentFacts() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload baseline = workload(device, 1, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            Workload differentIds = workload(device, 50_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var baselineCompatibility = generated(baseline, 2).batch().compatibility();
            assertEquals(baselineCompatibility,
                    generated(differentIds, 2).batch().compatibility());

            List<Workload> changed = List.of(
                    workload(device, 2, Shape.of(5), false,
                            Optional.empty(), true, false, 1),
                    workload(device, 3, Shape.of(4), true,
                            Optional.empty(), true, false, 1),
                    workload(device, 4, Shape.of(4), false,
                            Optional.of(ScalarValue.float32(-0.0f)), true, false, 1),
                    workload(device, 5, Shape.of(4), false,
                            Optional.empty(), false, true, 1),
                    workload(device, 6, Shape.of(4), false,
                            Optional.empty(), true, false, 2));
            for (Workload changedWorkload : changed) {
                assertNotEquals(baselineCompatibility,
                        generated(changedWorkload, 2).batch().compatibility());
            }
        }
    }

    @Test
    void codecIsCanonicalBoundedAndRejectsEveryDefensiveMismatch() {
        TestNativeApi api = new TestNativeApi();
        TestNativeApi otherApi = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            Generated current = generated(workload(device, 10, Shape.of(4), false,
                    Optional.empty(), true, false, 1), 2);
            var decision = new MetalNegTuningDecision(
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    current.batch().compatibility(), MetalNegTuningBatch.Candidate.MPSGRAPH);
            var codec = new MetalNegTuningCodec();
            byte[] first = codec.encodeDecision(decision);
            assertArrayEquals(first, codec.encodeDecision(decision));
            assertTrue(first.length <= MetalNegTuningCodec.MAX_DECISION_BYTES);
            assertEquals(decision, codec.decodeDecision(first, current.batch()).orElseThrow());

            byte[] corrupt = first.clone();
            corrupt[20] ^= 1;
            assertTrue(codec.decodeDecision(corrupt, current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(Arrays.copyOf(first, first.length - 1),
                    current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(Arrays.copyOf(first, first.length + 1),
                    current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(rewriteInt(first, 0, 0), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(rewriteInt(first, 4, 99), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(rewriteInt(first, 8, 99), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, first.length - 8, 99), current.batch()).isEmpty());

            Generated foreignSession = generated(workload(otherDevice, 20, Shape.of(4), false,
                    Optional.empty(), true, false, 1), 2);
            assertTrue(codec.decodeDecision(first, foreignSession.batch()).isEmpty());
            Generated changedWorkload = generated(workload(device, 30, Shape.of(5), false,
                    Optional.empty(), true, false, 1), 2);
            assertTrue(codec.decodeDecision(first, changedWorkload.batch()).isEmpty());
        }
    }

    @Test
    void freshAnalysisAuthenticatesSelectionsAndDeclaresOnlyTheirResources() {
        TestNativeApi api = new TestNativeApi();
        TestNativeApi otherApi = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            Workload workload = workload(device, 1000, Shape.of(8), false,
                    Optional.empty(), true, false, 1);
            Generated original = generated(workload, 2);
            var generator = new MetalNegRouteCandidateGenerator();
            BackendPartitionTuningHandoff<MetalNegTuningBatch, MetalNegTuningDecision> absent =
                    generator.absentHandoff(workload.context(), original.analysis().plan(), 2);
            var absentAnalysis = analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(absent))).context());
            assertEquals(MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG,
                    absentAnalysis.plan().route());
            assertEquals(2, absentAnalysis.requirements().size());

            var selected = generator.presentHandoff(
                    workload.context().partition(), original.batch(),
                    MetalNegTuningBatch.Candidate.MPSGRAPH);
            var selectedAnalysis = analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(selected))).context());
            assertEquals(MetalNegPreparationPlan.Route.MPSGRAPH, selectedAnalysis.plan().route());
            assertEquals(3, selectedAnalysis.requirements().size());
            assertTrue(selectedAnalysis.plan().addressWorkspace().isPresent());
            assertEquals(List.of(
                    MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                    MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generator.generate(
                            workload.context(), selectedAnalysis.plan(), 2).candidates());

            Workload foreignFacts = workload(otherDevice, 1000, Shape.of(8), false,
                    Optional.empty(), true, false, 1);
            Generated foreign = generated(foreignFacts, 2);
            var foreignDecision = generator.presentHandoff(
                    workload.context().partition(), foreign.batch(),
                    MetalNegTuningBatch.Candidate.MPSGRAPH);
            assertThrows(IllegalArgumentException.class, () -> analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(foreignDecision))).context()));

            Workload changed = workload(device, 1000, Shape.of(9), false,
                    Optional.empty(), true, false, 1);
            assertThrows(IllegalArgumentException.class, () -> analyze(withInputs(changed,
                    new MetalNegAnalysisInputs(device, Optional.of(selected))).context()));
            assertEquals(0, api.nativeAllocations.get());
            assertEquals(0, otherApi.nativeAllocations.get());
        }
    }

    @Test
    void invalidSemanticsFailBeforeCandidateGenerationAndColdUseIsConcurrent() throws Exception {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload valid = workload(device, 700, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            CompiledNode invalidNode = new CompiledNode(valid.context().nodes().getFirst().id(),
                    new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                    valid.context().nodes().getFirst().inputs(),
                    valid.context().nodes().getFirst().outputs());
            var invalidDag = new PartitionDag(valid.context().partition(), List.of(invalidNode));
            var invalid = new PrepareContext<>(invalidDag, valid.context().values(),
                    valid.context().memoryRequirements(), valid.context().constants(),
                    new MetalNegAnalysisInputs(device));
            assertThrows(IllegalArgumentException.class, () -> new MetalNegPartitionPreparer()
                    .analyze(invalid));

            Generated generated = generated(valid, 2);
            var executor = Executors.newFixedThreadPool(6);
            try {
                var calls = new ArrayList<java.util.concurrent.Callable<MetalNegTuningBatch>>();
                for (int index = 0; index < 48; index++) {
                    calls.add(() -> new MetalNegRouteCandidateGenerator().generate(
                            valid.context(), generated.analysis().plan(), 2));
                }
                for (var future : executor.invokeAll(calls)) {
                    assertEquals(generated.batch(), future.get(5, TimeUnit.SECONDS));
                }
            } finally {
                executor.shutdownNow();
                assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            }
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    private static Generated generated(Workload workload, int budget) {
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(workload.context());
        MetalNegTuningBatch batch = new MetalNegRouteCandidateGenerator().generate(
                workload.context(), analysis.plan(), budget);
        return new Generated(analysis, batch);
    }

    private static BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            PrepareContext<MetalNegAnalysisInputs> context) {
        return new MetalNegPartitionPreparer().analyze(context);
    }

    private static Workload withInputs(Workload workload, MetalNegAnalysisInputs inputs) {
        PrepareContext<MetalNegAnalysisInputs> context = workload.context();
        return new Workload(new PrepareContext<>(context.partitionDag(), context.values(),
                context.memoryRequirements(), context.constants(), inputs));
    }

    private static Workload workload(
            MetalDeviceContext device,
            long identityBase,
            Shape shape,
            boolean requiresGrad,
            Optional<ScalarValue> splat,
            boolean graphOutput,
            boolean externalConsumer,
            int nodeCount) {
        TensorDescriptor descriptor = new TensorDescriptor(
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        var nodes = new ArrayList<CompiledNode>();
        var values = new ArrayList<GraphValue>();
        ValueId feed = new ValueId(identityBase);
        values.add(new GraphValue(feed, descriptor));
        ValueId previous = feed;
        for (int index = 0; index < nodeCount; index++) {
            ValueId output = new ValueId(identityBase + index + 1);
            nodes.add(new CompiledNode(new NodeId(identityBase + index),
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(previous), List.of(output)));
            values.add(new GraphValue(output, descriptor));
            previous = output;
        }
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        PlannedPartition external = new PlannedPartition(
                new io.github.pho001.synaptik.backend.contract.BackendId("external-test"),
                List.of(new NodeId(identityBase + 90_000)));
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        requirements.add(new LogicalMemoryRequirement(feed, descriptor, Optional.empty(),
                List.of(partition), false));
        for (int index = 0; index < nodeCount; index++) {
            ValueId output = nodes.get(index).outputs().getFirst();
            boolean last = index == nodeCount - 1;
            List<PlannedPartition> consumers = last
                    ? externalConsumer ? List.of(external) : List.of()
                    : List.of(partition);
            requirements.add(new LogicalMemoryRequirement(output, descriptor,
                    Optional.of(partition), consumers, last && graphOutput));
        }
        Map<ValueId, ScalarValue> constants = splat
                .<Map<ValueId, ScalarValue>>map(value -> Map.of(feed, value))
                .orElseGet(Map::of);
        var dag = new PartitionDag(partition, nodes);
        return new Workload(new PrepareContext<>(dag, values, requirements, constants,
                new MetalNegAnalysisInputs(device)));
    }

    private static byte[] rewriteInt(byte[] source, int offset, int value) {
        byte[] changed = source.clone();
        changed[offset] = (byte) (value >>> 24);
        changed[offset + 1] = (byte) (value >>> 16);
        changed[offset + 2] = (byte) (value >>> 8);
        changed[offset + 3] = (byte) value;
        CRC32 checksum = new CRC32();
        checksum.update(changed, 0, changed.length - Integer.BYTES);
        int crc = (int) checksum.getValue();
        int checksumOffset = changed.length - Integer.BYTES;
        changed[checksumOffset] = (byte) (crc >>> 24);
        changed[checksumOffset + 1] = (byte) (crc >>> 16);
        changed[checksumOffset + 2] = (byte) (crc >>> 8);
        changed[checksumOffset + 3] = (byte) crc;
        return changed;
    }

    private record Workload(PrepareContext<MetalNegAnalysisInputs> context) { }

    private record Generated(
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis,
            MetalNegTuningBatch batch) { }

    private static final class TestNativeApi extends MetalNativeApi {
        private final AtomicInteger nativeAllocations = new AtomicInteger();
        private long nextHandle = 1;

        @Override synchronized Handle createContext() { return handle(); }
        @Override void releaseContext(Handle context) { }
        @Override Handle createBuffer(Handle context, long logicalByteSize) {
            nativeAllocations.incrementAndGet();
            return handle();
        }
        @Override void releaseBuffer(Handle buffer) { }
        @Override void upload(Handle buffer, long offset, MemorySegment source, long count) { }
        @Override void download(Handle buffer, long offset, MemorySegment target, long count) { }
        @Override NativeCreateResult createNegExecutableNative(
                Handle context, int[] ranks, long[] dimensions, int[] inputs, int[] outputs,
                int[] feeds, int[] targets) {
            nativeAllocations.incrementAndGet();
            return new NativeCreateResult(0, handle());
        }
        @Override int releaseExecutableNative(Handle executable) { return 0; }
        @Override int runExecutableNative(
                Handle executable, int inputCount, MemorySegment inputs,
                int outputCount, MemorySegment outputs) { return 0; }
        @Override NativeCreateResult createNegKernelPipelineNative(
                Handle context, long elementCount) {
            nativeAllocations.incrementAndGet();
            return new NativeCreateResult(0, handle());
        }
        @Override int releaseNegKernelPipelineNative(Handle pipeline) { return 0; }
        @Override int runNegKernelPipelineNative(
                Handle pipeline, Handle input, Handle output) { return 0; }
        @Override public void close() { }

        private synchronized Handle handle() {
            return new Handle(MemorySegment.ofAddress(nextHandle++));
        }
    }
}
