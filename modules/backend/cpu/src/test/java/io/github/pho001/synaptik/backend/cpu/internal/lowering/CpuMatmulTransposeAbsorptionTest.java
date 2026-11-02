package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Private transposes remove resources, not MATMUL arithmetic or observable values. */
public final class CpuMatmulTransposeAbsorptionTest {
    @Test void eitherOrBothInputsUseOriginalBackingAndExistingStridedClassBytes() {
        for (boolean left : List.of(false, true)) for (boolean right : List.of(false, true)) {
            if (!left && !right) continue;
            var context = context(DataType.FLOAT32, 2, 3, 4, left, right, false, true,
                    CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            var plan = analysis.plan();
            assertEquals(1, plan.units().size());
            assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(4)), plan.boundaryValues());
            assertEquals(3, analysis.requirements().size());
            assertEquals(0, plan.affineAddressPairs().length);
            var unit = plan.units().getFirst();
            var backing = context.values().stream().collect(java.util.stream.Collectors.toMap(
                    GraphValue::id, GraphValue::descriptor));
            assertEquals(left ? List.of(2L, 7L) : List.of(9L, 2L),
                    unit.accessBindings().get(0).effectiveStrides());
            assertEquals(right ? List.of(2L, 9L) : List.of(11L, 2L),
                    unit.accessBindings().get(1).effectiveStrides());
            assertEquals(3, unit.accessBindings().get(0).baseElementOffset());
            assertEquals(3, unit.accessBindings().get(1).baseElementOffset());
            var matmul = context.nodes().getLast();
            var bare = CpuScatterLoweringTest.context(matmul.operation(), List.of(0, 1),
                    matmul.inputs().stream().map(backing::get).toList(), backing.get(new ValueId(4)));
            var bareUnit = new CpuPartitionPreparer().analyze(bare).plan().units().getFirst();
            assertEquals(bareUnit.portablePlan().specialization(), unit.portablePlan().specialization());
            var generator = new CpuClassFileKernelGenerator();
            assertArrayEquals(generator.generateClassBytes(bareUnit.portablePlan().specialization(),
                            bareUnit.portablePlan().kernelIr()),
                    generator.generateClassBytes(unit.portablePlan().specialization(),
                            unit.portablePlan().kernelIr()));
        }
    }

    /**
     * Uses small exactly represented integers to isolate addressing from numerical tolerance.
     * Poisoned padding and nonzero offsets catch reads from copied/canonical rather than backing geometry.
     *
     * @throws Throwable if finalization or the exact typed generated entry fails
     */
    @Test void generatedEntriesReadAllNumericBackingsAndEmptyContractions() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.INT32, DataType.INT64)) {
            for (int k : new int[] {0, 3}) {
                for (boolean left : List.of(false, true)) for (boolean right : List.of(false, true)) {
                    if (!left && !right) continue;
                    var context = context(type, 2, k, 4, left, right, false, k != 0,
                            CpuPartitionAnalysisInputs.DEFAULT);
                    if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(context)) continue;
                    var analysis = new CpuPartitionPreparer().analyze(context);
                    assertEquals(1, analysis.plan().units().size(), type + " K=" + k);
                    var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
                    assertEquals(3, executable.memoryPlan().buffers().size());
                    assertTrue(executable.memoryPlan().workspaces().isEmpty());
                    try (var arena = Arena.ofConfined()) {
                        var data = new ArrayList<MemorySegment>();
                        for (var declaration : analysis.plan().bufferDeclarations()) {
                            var segment = arena.allocate(declaration.byteSize(), declaration.byteAlignment());
                            segment.fill((byte) 0x7f);
                            data.add(segment);
                        }
                        TensorDescriptor a = context.values().get(0).descriptor();
                        TensorDescriptor b = context.values().get(1).descriptor();
                        fill(data.get(0), a); fill(data.get(1), b);
                        var bindings = data.stream().map(segment -> List.of(new BufferRepresentationBinding(
                                CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,
                                        segment.byteSize() / type.byteWidth(), segment)),
                                RunResourceOwnership.BORROWED))).toList();
                        try (var run = new RunState(executable.memoryPlan(), bindings, List.of())) {
                            executable.bind(run).execute();
                        }
                        for (int row = 0; row < 2; row++) for (int column = 0; column < 4; column++) {
                            double expected = 0;
                            for (int inner = 0; inner < k; inner++) {
                                expected += value(left ? inner : row, left ? row : inner)
                                        * value(right ? column : inner, right ? inner : column);
                            }
                            assertEquals(expected, read(data.get(2), type, row * 4L + column),
                                    type + " left=" + left + " right=" + right + " K=" + k);
                        }
                    }
                }
            }
        }
    }

    @Test void publicationAndExternalPartitionObligationsKeepTheCopy() {
        var base = context(DataType.FLOAT32, 2, 3, 4, false, true, false, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var outside = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(new NodeId(100)));
        for (boolean published : List.of(false, true)) {
            var requirements = base.memoryRequirements().stream().map(requirement ->
                    requirement.valueId().equals(new ValueId(3))
                            ? new LogicalMemoryRequirement(requirement.valueId(), requirement.descriptor(),
                                    requirement.producerPartition(), published
                                            ? List.of(base.partition()) : List.of(base.partition(), outside),
                                    published)
                            : requirement).toList();
            var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    requirements, base.constants(), base.backendInputs());
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            assertEquals(2, plan.units().size());
            assertTrue(plan.boundaryValues().contains(new ValueId(3)));
        }
    }

    @Test void preparedHeapArraysUseTheOriginalTransposedBacking() {
        var inputs = new CpuPartitionAnalysisInputs(false, Collections.nCopies(3,
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization
                        .CarrierAccess.FLOAT_ARRAY));
        var context = context(DataType.FLOAT32, 2, 3, 4, false, true, false, false, inputs);
        var analysis = new CpuPartitionPreparer().analyze(context);
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        float[] left = {1, 2, 3, 4, 5, 6};
        float[] right = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        float[] output = new float[8];
        var bindings = List.of(MemorySegment.ofArray(left), MemorySegment.ofArray(right),
                MemorySegment.ofArray(output)).stream().map(segment -> List.of(new BufferRepresentationBinding(
                        CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32,
                                segment.byteSize() / Float.BYTES, segment)),
                        RunResourceOwnership.BORROWED))).toList();
        try (var run = new RunState(executable.memoryPlan(), bindings, List.of())) {
            executable.bind(run).execute();
        }
        assertEquals(3, analysis.requirements().size());
        assertArrayEquals(new float[] {50, 68, 86, 104, 122, 167, 212, 257}, output);
    }

    @Test void repeatedTransposeInputOccurrencesKeepTheCopy() {
        var base = context(DataType.FLOAT32, 2, 2, 2, true, false, false, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var node = base.nodes().getLast();
        var repeated = new CompiledNode(node.id(), node.operation(),
                List.of(new ValueId(2), new ValueId(2)), node.outputs());
        var requirements = base.memoryRequirements().stream().map(requirement ->
                requirement.valueId().equals(new ValueId(1))
                        ? new LogicalMemoryRequirement(requirement.valueId(), requirement.descriptor(),
                                Optional.empty(), List.of(), false) : requirement).toList();
        var context = new PrepareContext<>(base.partition(), List.of(base.nodes().getFirst(), repeated),
                base.values(), requirements, base.constants(), base.backendInputs());
        assertEquals(2, new CpuPartitionDagDecomposer().decompose(context,
                new CpuPartitionLowering()).size());
        assertThrows(IllegalArgumentException.class, () -> new CpuMatmulLowering().lower(context));
    }

    @Test void sharedOriginalBackingDoesNotMergeDistinctReadBoundaries() {
        var one = context(DataType.FLOAT32, 2, 2, 2, true, false, false, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var sharedOne = withNodeInputs(one, 1, List.of(new ValueId(2), new ValueId(0)));
        var onePlan = new CpuPartitionPreparer().analyze(sharedOne).plan();
        assertEquals(2, onePlan.units().size());
        assertTrue(onePlan.boundaryValues().contains(new ValueId(2)));

        var both = context(DataType.FLOAT32, 2, 2, 2, true, true, false, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var sharedBoth = withNodeInputs(both, 1, List.of(new ValueId(0)));
        var bothPlan = new CpuPartitionPreparer().analyze(sharedBoth).plan();
        assertEquals(2, bothPlan.units().size());
        assertTrue(bothPlan.boundaryValues().contains(new ValueId(2)));
        assertFalse(bothPlan.boundaryValues().contains(new ValueId(3)));
    }

    @Test void aSecondLocalConsumerKeepsTheTransposeMaterialized() {
        var base = context(DataType.FLOAT32, 2, 3, 4, false, true, false, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var nodes = new ArrayList<>(base.nodes());
        nodes.add(new CompiledNode(new NodeId(2),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(3)), List.of(new ValueId(8))));
        var values = new ArrayList<>(base.values());
        values.add(new GraphValue(new ValueId(8), values.stream().filter(value ->
                value.id().equals(new ValueId(3))).findFirst().orElseThrow().descriptor()));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var memory = values.stream().map(value -> new LogicalMemoryRequirement(value.id(),
                value.descriptor(), nodes.stream().anyMatch(node -> node.outputs().contains(value.id()))
                        ? Optional.of(partition) : Optional.empty(),
                nodes.stream().anyMatch(node -> node.inputs().contains(value.id()))
                        ? List.of(partition) : List.of(),
                Set.of(new ValueId(4), new ValueId(8)).contains(value.id()))).toList();
        var context = new PrepareContext<>(partition, nodes, values, memory, Map.of(), base.backendInputs());
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(3, plan.units().size());
        assertTrue(plan.boundaryValues().contains(new ValueId(3)));
    }

    @Test void biasAndActivationKeepTheirFusedAlternativeWithBothTransposeInputs() {
        var context = context(DataType.FLOAT32, 2, 3, 4, true, true, true, false,
                CpuPartitionAnalysisInputs.DEFAULT);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertFalse(plan.boundaryValues().contains(new ValueId(2)));
        assertFalse(plan.boundaryValues().contains(new ValueId(3)));
        assertTrue(plan.fusionDecisions().stream().filter(CpuFusionDecision.LegalCandidate.class::isInstance)
                .map(CpuFusionDecision.LegalCandidate.class::cast).anyMatch(candidate ->
                        candidate.identity().units().size() == 1
                                && candidate.identity().units().getFirst().memberNodePositions()
                                        .equals(List.of(0, 1, 2, 3, 4))));
    }

    /**
     * Builds exact rank-two forward metadata with optional transposes and a bias/RELU suffix.
     *
     * @param type common numeric type
     * @param m output rows
     * @param k contraction extent
     * @param n output columns
     * @param left whether the left source is stored in transposed shape
     * @param right whether the right source is stored in transposed shape
     * @param suffix whether to append bias and RELU
     * @param padded whether backing layouts use stride two, row gaps and offset three
     * @param inputs exact cold route inputs
     * @return fresh validated context whose only published result is the final output
     */
    public static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, int m, int k, int n,
            boolean left, boolean right, boolean suffix, boolean padded, CpuPartitionAnalysisInputs inputs) {
        var descriptors = new LinkedHashMap<ValueId, TensorDescriptor>();
        descriptors.put(new ValueId(0), descriptor(type, left ? Shape.of(k, m) : Shape.of(m, k), padded));
        descriptors.put(new ValueId(1), descriptor(type, right ? Shape.of(n, k) : Shape.of(k, n), padded));
        var nodes = new ArrayList<CompiledNode>();
        if (left) addTranspose(nodes, descriptors, 0, 2);
        if (right) addTranspose(nodes, descriptors, 1, 3);
        descriptors.put(new ValueId(4), descriptor(type, Shape.of(m, n), false));
        nodes.add(new CompiledNode(new NodeId(nodes.size()),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(left ? 2 : 0), new ValueId(right ? 3 : 1)), List.of(new ValueId(4))));
        if (suffix) {
            descriptors.put(new ValueId(5), descriptor(type, Shape.of(n), false));
            descriptors.put(new ValueId(6), descriptor(type, Shape.of(m, n), false));
            descriptors.put(new ValueId(7), descriptor(type, Shape.of(m, n), false));
            nodes.add(new CompiledNode(new NodeId(nodes.size()),
                    new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                    List.of(new ValueId(4), new ValueId(5)), List.of(new ValueId(6))));
            nodes.add(new CompiledNode(new NodeId(nodes.size()),
                    new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                    List.of(new ValueId(6)), List.of(new ValueId(7))));
        }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        descriptors.forEach((id, descriptor) -> {
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), id.equals(new ValueId(suffix ? 7 : 4))));
        });
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    /**
     * Appends one semantic transpose while retaining the original storage offset and strides.
     *
     * @param nodes mutable fixture node list receiving the next stable node identity
     * @param descriptors mutable descriptors containing the static rank-two source
     * @param source existing source value number
     * @param result unused result value number to insert
     */
    private static void addTranspose(List<CompiledNode> nodes, Map<ValueId, TensorDescriptor> descriptors,
            int source, int result) {
        var original = descriptors.get(new ValueId(source));
        var shape = Shape.of(original.shape().toLongArray()[1], original.shape().toLongArray()[0]);
        var layout = original.layout().orElseThrow();
        descriptors.put(new ValueId(result), new TensorDescriptor(original.dataType(), shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {layout.stride(1), layout.stride(0)},
                        layout.storageOffset(), true)), false));
        nodes.add(new CompiledNode(new NodeId(nodes.size()),
                new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))),
                List.of(new ValueId(source)), List.of(new ValueId(result))));
    }

    /**
     * Rebinds one fixture input list and its exact local consumer obligations without new outputs.
     *
     * @param base non-null fixture whose node/output identities stay fixed
     * @param index valid node position to replace
     * @param inputs non-null replacement semantic input identities
     * @return fresh context with recomputed consumer partitions
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> withNodeInputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, int index, List<ValueId> inputs) {
        var nodes = new ArrayList<>(base.nodes());
        var original = nodes.get(index);
        nodes.set(index, new CompiledNode(original.id(), original.operation(), inputs, original.outputs()));
        var memory = base.memoryRequirements().stream().map(requirement ->
                new LogicalMemoryRequirement(requirement.valueId(), requirement.descriptor(),
                        requirement.producerPartition(), nodes.stream().anyMatch(node ->
                                node.inputs().contains(requirement.valueId()))
                                        ? List.of(base.partition()) : List.of(),
                        requirement.graphOutput())).toList();
        return new PrepareContext<>(base.partition(), nodes, base.values(), memory,
                base.constants(), base.backendInputs());
    }

    /**
     * Supplies either canonical storage or a gapped matrix layout that exposes lost read geometry.
     *
     * @param type non-null represented numeric type
     * @param shape non-null static shape, rank two when padded
     * @param padded whether to use column stride two, a row gap and offset three
     * @return resolved non-gradient fixture descriptor
     */
    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean padded) {
        var layout = padded ? LayoutDescriptor.of(shape,
                new long[] {shape.toLongArray()[1] * 2 + 3, 2}, 3, true)
                : LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    /**
     * Chooses small signed integers whose products and short sums are exact in every tested type.
     *
     * @param row non-negative source row
     * @param column non-negative source column
     * @return an exactly represented value between minus two and two
     */
    private static double value(long row, long column) { return (row * 3 + column) % 5 - 2; }

    /**
     * Writes only logical elements, leaving poisoned padding intact to expose incorrect addressing.
     *
     * @param segment live writable segment large enough for the descriptor's referenced span
     * @param descriptor non-null resolved static matrix descriptor with one tested numeric type
     * @throws AssertionError if a fixture supplies an unsupported type
     */
    private static void fill(MemorySegment segment, TensorDescriptor descriptor) {
        long[] shape = descriptor.shape().toLongArray();
        var layout = descriptor.layout().orElseThrow();
        for (long row = 0; row < shape[0]; row++) for (long column = 0; column < shape[1]; column++) {
            long offset = (layout.storageOffset() + row * layout.stride(0) + column * layout.stride(1))
                    * descriptor.dataType().byteWidth();
            double value = value(row, column);
            switch (descriptor.dataType()) {
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, offset, (float) value);
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, offset, value);
                case BFLOAT16 -> segment.set(ValueLayout.JAVA_SHORT, offset, BFloat16Bits.fromFloat((float) value));
                case FLOAT16 -> segment.set(ValueLayout.JAVA_SHORT, offset, Float.floatToFloat16((float) value));
                case INT32 -> segment.set(ValueLayout.JAVA_INT, offset, (int) value);
                case INT64 -> segment.set(ValueLayout.JAVA_LONG, offset, (long) value);
                default -> throw new AssertionError(descriptor.dataType());
            }
        }
    }

    /**
     * Decodes one stored result for exact comparison of the small fixture integers.
     *
     * @param segment live readable result segment
     * @param type non-null represented numeric type
     * @param element valid physical element index, not a byte offset
     * @return the stored numeric value, exactly representable as double for these fixtures
     * @throws AssertionError if a fixture supplies an unsupported type
     */
    private static double read(MemorySegment segment, DataType type, long element) {
        long offset = element * type.byteWidth();
        return switch (type) {
            case FLOAT32 -> segment.get(ValueLayout.JAVA_FLOAT, offset);
            case FLOAT64 -> segment.get(ValueLayout.JAVA_DOUBLE, offset);
            case BFLOAT16 -> BFloat16Bits.toFloat(segment.get(ValueLayout.JAVA_SHORT, offset));
            case FLOAT16 -> Float.float16ToFloat(segment.get(ValueLayout.JAVA_SHORT, offset));
            case INT32 -> segment.get(ValueLayout.JAVA_INT, offset);
            case INT64 -> segment.get(ValueLayout.JAVA_LONG, offset);
            default -> throw new AssertionError(type);
        };
    }
}
