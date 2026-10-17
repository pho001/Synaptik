package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Provider-to-invocation regressions for producer-ordered scans and loss occurrence geometry. */
class CpuComposedValueValidationTest {
    @Test void producedSoftmaxInfinityRejectsBeforeAnyConsumerWriteOnEveryInvocation() {
        var descriptor = descriptor(DataType.FLOAT32, Shape.of(2, 2));
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            var analysis = analyze(List.of(descriptor, descriptor, descriptor), List.of(
                    node(0, plain(UnaryElementwiseKind.EXP), 1, 0),
                    node(1, new Operation(kind, new SoftmaxAttrs(1)), 2, 1)));
            assertEquals(2, analysis.plan().units().getLast().selectedRangeCount());
            try (var fixture = new Fixture(analysis)) {
                fixture.floats(0, 0, 0, 0, 0);
                fixture.floats(1, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
                BoundInvocation bound = assertDoesNotThrow(() -> fixture.executable.bind(fixture.state));
                bound.execute();
                assertArrayEquals(kind == SoftmaxKind.SOFTMAX
                        ? new float[] {.5f, .5f, .5f, .5f}
                        : new float[] {-(float) Math.log(2), -(float) Math.log(2),
                                -(float) Math.log(2), -(float) Math.log(2)},
                        fixture.floats(2), 1e-7f);
                for (int repetition = 0; repetition < 2; repetition++) {
                    fixture.floats(0, 0, 0, 0, 1000);
                    fixture.floats(2, 71, 71, 71, 71);
                    assertThrows(IllegalArgumentException.class, bound::execute);
                    assertTrue(Float.isInfinite(fixture.floats(1)[3]));
                    assertArrayEquals(new float[] {71, 71, 71, 71}, fixture.floats(2));
                    fixture.floats(0, 0, 0, 0, 0);
                    bound.execute();
                }
            }
        }
    }

    @Test void computedIndexTargetsAreCheckedAfterProducerAndBeforeParallelLossWrites() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            var target = descriptor(type, Shape.of(2));
            var output = descriptor(DataType.FLOAT32, Shape.of(2));
            ScalarValue ignored = type == DataType.INT32
                    ? ScalarValue.int32(-1) : ScalarValue.int64(-1);
            var analysis = analyze(List.of(output, target,
                    descriptor(DataType.FLOAT32, Shape.of(2, 3)), output), List.of(
                    node(0, new Operation(CastKind.CAST, new CastAttrs(type)), 1, 0),
                    node(1, new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                                    Optional.of(ignored))), 3, 2, 1)));
            assertEquals(2, analysis.plan().units().getLast().selectedRangeCount());
            try (var fixture = new Fixture(analysis)) {
                fixture.floats(0, 1, 2);
                fixture.indices(1, type, 99, 99);
                fixture.floats(2, 0, 0, 0, 0, 0, 0);
                var bound = assertDoesNotThrow(() -> fixture.executable.bind(fixture.state));
                bound.execute();
                assertArrayEquals(new float[] {(float) Math.log(3), (float) Math.log(3)},
                        fixture.floats(3), 1e-6f);
                fixture.floats(0, 1, 3);
                fixture.floats(3, 71, 71);
                assertThrows(IndexOutOfBoundsException.class, bound::execute);
                assertArrayEquals(new float[] {71, 71}, fixture.floats(3));
                fixture.floats(0, -1, 2);
                bound.execute();
                assertArrayEquals(new float[] {0, (float) Math.log(3)}, fixture.floats(3), 1e-6f);
            }
        }
    }

    @Test void computedMasksReplacePoisonedStorageBeforeMaskedReductionAndAttentionScans() {
        for (boolean attention : List.of(false, true)) {
            var mask = descriptor(DataType.BOOL, Shape.of(2, 2));
            List<TensorDescriptor> descriptors;
            CompiledNode consumer;
            if (attention) {
                var matrix = descriptor(DataType.FLOAT32, Shape.of(2, 1));
                descriptors = List.of(mask, mask, matrix, matrix, matrix, matrix);
                consumer = node(1, new Operation(
                        ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)), 5, 2, 3, 4, 1);
            } else {
                descriptors = List.of(mask, mask, descriptor(DataType.FLOAT32, Shape.of(2, 2)),
                        descriptor(DataType.FLOAT32, Shape.of(2)));
                consumer = node(1, new Operation(AggregateReductionKind.SUM,
                        new MaskedReductionAttrs(1)), 3, 2, 1);
            }
            var analysis = analyze(descriptors, List.of(
                    node(0, plain(BooleanLogicalKind.NOT), 1, 0), consumer));
            try (var fixture = new Fixture(analysis)) {
                fixture.bytes(0, 0, 1, 1, 0);
                fixture.bytes(1, 9, 9, 9, 9);
                if (attention) {
                    fixture.floats(2, 0, 0);
                    fixture.floats(3, 0, 0);
                    fixture.floats(4, 10, 20);
                } else fixture.floats(2, 10, 20, 30, 40);
                int result = attention ? 5 : 3;
                var bound = assertDoesNotThrow(() -> fixture.executable.bind(fixture.state));
                bound.execute();
                assertArrayEquals(attention ? new float[] {10, 20} : new float[] {10, 40},
                        fixture.floats(result));
                fixture.bytes(0, 1, 0, 0, 1);
                bound.execute();
                assertArrayEquals(attention ? new float[] {20, 10} : new float[] {20, 30},
                        fixture.floats(result));
                fixture.bytes(0, 0, 0, 0, 2);
                fixture.floats(result, 71, 71);
                assertThrows(IllegalArgumentException.class, bound::execute);
                assertArrayEquals(new float[] {71, 71}, fixture.floats(result));
            }
        }
    }

    @Test void composedStridedLossesRetainOccurrenceBindingsAndRankZeroCacheProjection()
            throws Exception {
        for (LossKind kind : List.of(LossKind.MEAN_SQUARED_ERROR,
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS)) {
            for (boolean strided : List.of(false, true)) {
                var shape = Shape.of(2, 3);
                var input = new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(strided
                        ? LayoutDescriptor.of(shape, new long[] {8, 2}, 1, true)
                        : LayoutDescriptor.contiguous(shape)), false);
                Shape resultShape = kind == LossKind.MEAN_SQUARED_ERROR ? shape : Shape.of(2);
                var result = descriptor(DataType.FLOAT32, resultShape);
                Operation loss = kind == LossKind.MEAN_SQUARED_ERROR
                        ? new Operation(kind, new MeanSquaredErrorAttrs(LossReduction.NONE))
                        : new Operation(kind, new DenseCategoricalCrossEntropyWithLogitsAttrs(
                                1, LossReduction.NONE));
                var analysis = analyze(List.of(input, input, result, result), List.of(
                        node(0, loss, 2, 0, 1), node(1, plain(UnaryElementwiseKind.NEG), 3, 2)));
                var unit = analysis.plan().units().getFirst();
                var ir = assertInstanceOf(CpuLossIr.class, unit.portablePlan().portableKernelIr());
                assertEquals(ir.boundaryAccesses(),
                        unit.accessBindings().stream().map(CpuAccessPlan.Binding::plan).toList());
                assertTrue(ir.encodedKernelIr().values().stream()
                        .allMatch(value -> value.accessPlan().iterationRank() == 0));
                // Replacing an occurrence access with the cache projection must still fail.
                var accesses = new ArrayList<>(ir.boundaryAccesses());
                accesses.set(0, ir.encodedKernelIr().values().getFirst().accessPlan());
                CpuLossIr forgedIr = copyRecord(ir, "boundaryAccesses", accesses);
                var forgedPortable = copyRecord(unit.portablePlan(), "portableKernelIr", forgedIr);
                var forgedUnit = copyRecord(unit, "portablePlan", forgedPortable);
                var units = new ArrayList<>(analysis.plan().units());
                units.set(0, forgedUnit);
                var failure = assertThrows(InvocationTargetException.class,
                        () -> copyRecord(analysis.plan(), "units", units));
                assertInstanceOf(IllegalArgumentException.class, failure.getCause());
                try (var fixture = new Fixture(analysis)) {
                    fixture.logicalFloats(0, input, 1, 2, 3, 3, 2, 1);
                    fixture.logicalFloats(1, input, 0, 0, 1, 1, 0, 0);
                    fixture.executable.bind(fixture.state).execute();
                    assertArrayEquals(kind == LossKind.MEAN_SQUARED_ERROR
                            ? new float[] {-1, -4, -4, -4, -4, -1}
                            : new float[] {-(float) Math.log(1 + Math.exp(-1) + Math.exp(-2)),
                                    -(float) Math.log(1 + Math.exp(-1) + Math.exp(-2))},
                            fixture.floats(3), 1e-6f);
                }
            }
        }
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            List<TensorDescriptor> descriptors, List<CompiledNode> nodes) {
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            var id = new ValueId(index);
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            values.add(new GraphValue(id, descriptors.get(index)));
            memory.add(new LogicalMemoryRequirement(id, descriptors.get(index),
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), produced && !consumed));
        }
        var provider = new CpuCapabilityProvider();
        for (var node : nodes) assertTrue(provider.supports(new OperationCapabilityQuery(
                node.operation(), node.inputs().stream()
                        .map(id -> descriptors.get(Math.toIntExact(id.value()))).toList(),
                node.outputs().stream().map(id -> descriptors.get(Math.toIntExact(id.value()))).toList())));
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR, 2, 2, 1);
        return new CpuPartitionPreparer().analyze(new PrepareContext<>(partition, nodes, values,
                memory, Map.of(), new CpuPartitionAnalysisInputs(false, List.of(), config)));
    }

    private static Operation plain(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static CompiledNode node(int ordinal, Operation operation, int output, int... inputs) {
        return new CompiledNode(new NodeId(ordinal), operation,
                Arrays.stream(inputs).mapToObj(ValueId::new).toList(), List.of(new ValueId(output)));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    @SuppressWarnings("unchecked")
    private static <T extends Record> T copyRecord(T source, String component, Object replacement)
            throws Exception {
        var components = source.getClass().getRecordComponents();
        Class<?>[] types = new Class<?>[components.length];
        Object[] values = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            types[i] = components[i].getType();
            values[i] = components[i].getName().equals(component)
                    ? replacement : components[i].getAccessor().invoke(source);
        }
        assertTrue(Arrays.stream(components).anyMatch(value -> value.getName().equals(component)));
        return (T) source.getClass().getDeclaredConstructor(types).newInstance(values);
    }

    private static final class Fixture implements AutoCloseable {
        final Arena arena = Arena.ofShared();
        final CpuWorkerGroup workers = new CpuWorkerGroup(2);
        final Map<ValueId, MemorySegment> storage = new HashMap<>();
        final io.github.pho001.synaptik.runtime.execution.PreparedExecutable executable;
        final RunState state;

        Fixture(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
            var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
            var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
            var assignments = new ArrayList<PreparationResourceAssignment>();
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            var scratch = new ArrayList<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation>();
            for (var requirement : analysis.requirements()) {
                if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                    var slot = new BufferSlot(buffers.size());
                    buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                            buffer.byteAlignment()));
                    assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot,
                            buffers.size() - 1, 0));
                    MemorySegment segment = arena.allocate(buffer.byteSize(), buffer.byteAlignment());
                    storage.put(buffer.valueId(), segment);
                    var unit = analysis.plan().units().stream().filter(candidate ->
                            candidate.boundaryValues().contains(buffer.valueId())).findFirst().orElseThrow();
                    var type = unit.portablePlan().specialization().boundaryDataTypes().get(
                            unit.boundaryValues().indexOf(buffer.valueId()));
                    bindings.add(List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(
                            new MemorySegmentStorage(type, buffer.byteSize() / type.byteWidth(),
                                    segment)), RunResourceOwnership.BORROWED)));
                } else if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                    var slot = new WorkspaceSlot(workspaces.size());
                    workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, workspace.byteSize(),
                            workspace.byteAlignment()));
                    assignments.add(new PreparationResourceAssignment.Workspace(workspace, slot,
                            workspaces.size() - 1));
                    scratch.add(CpuContiguousWorkspace.allocate(workspace.byteSize(),
                            workspace.byteAlignment()));
                }
            }
            var memory = new PreparedMemoryPlan(buffers, workspaces);
            executable = new CpuPartitionFinalizer(Optional.empty(), Optional.of(workers))
                    .finalizePartition(new BackendPartitionFinalization<>(analysis, memory, assignments))
                    .executable();
            state = new RunState(memory, bindings, scratch);
        }

        void floats(int id, float... values) {
            MemorySegment.copy(MemorySegment.ofArray(values), 0, storage.get(new ValueId(id)), 0,
                    values.length * (long) Float.BYTES);
        }
        float[] floats(int id) { return storage.get(new ValueId(id)).toArray(ValueLayout.JAVA_FLOAT); }
        void indices(int id, DataType type, long... values) {
            for (int i = 0; i < values.length; i++) {
                var segment = storage.get(new ValueId(id));
                if (type == DataType.INT32) segment.setAtIndex(ValueLayout.JAVA_INT, i, (int) values[i]);
                else segment.setAtIndex(ValueLayout.JAVA_LONG, i, values[i]);
            }
        }
        void bytes(int id, int... values) {
            for (int i = 0; i < values.length; i++)
                storage.get(new ValueId(id)).set(ValueLayout.JAVA_BYTE, i, (byte) values[i]);
        }
        void logicalFloats(int id, TensorDescriptor descriptor, float... values) {
            var layout = descriptor.layout().orElseThrow();
            for (int i = 0; i < values.length; i++) {
                long address = layout.storageOffset() + i / 3 * layout.strides()[0]
                        + i % 3 * layout.strides()[1];
                storage.get(new ValueId(id)).setAtIndex(ValueLayout.JAVA_FLOAT, address, values[i]);
            }
        }
        @Override public void close() {
            try { state.close(); } finally { try { workers.close(); } finally { arena.close(); } }
        }
    }
}
