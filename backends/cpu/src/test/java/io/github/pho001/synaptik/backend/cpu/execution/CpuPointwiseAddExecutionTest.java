package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CpuPointwiseAddExecutionTest {
    @TempDir Path artifactRoot;

    @Test void executesOrderedFloatAndIntegralChainsWithExactBoundaryBehavior() {
        assertArrayEquals(new double[]{4.0, 0.0, Double.POSITIVE_INFINITY},
                executeDoubles(new double[]{1.5, -0.0, Double.MAX_VALUE},
                        new double[]{1.0, 0.0, Double.MAX_VALUE}));
        assertArrayEquals(new float[]{4.0f, 0.0f, Float.POSITIVE_INFINITY},
                executeFloats(new float[]{1.5f, -0.0f, Float.MAX_VALUE},
                        new float[]{1.0f, 0.0f, Float.MAX_VALUE}));
        assertArrayEquals(new int[]{-1, 1, -3},
                executeInts(new int[]{Integer.MAX_VALUE, 1, -2}, new int[]{1, -1, 1}));
        assertArrayEquals(new long[]{-1, 1, -3},
                executeLongs(new long[]{Long.MAX_VALUE, 1, -2}, new long[]{1, -1, 1}));
    }

    @Test void supportsZeroElementsRepeatedBindingAndOnePartitionStateGuard() {
        Prepared prepared = prepare(DataType.FLOAT32, 0);
        try (prepared.workers) {
            BoundInvocation invocation = prepared.executable.bind(prepared.state);
            assertDoesNotThrow(invocation::execute);
            assertDoesNotThrow(invocation::execute);
            prepared.state.close();
            assertEquals("run state is closed", assertThrows(IllegalStateException.class,
                    invocation::execute).getMessage());
        }
    }

    @Test void executesRepeatedInputFanOutAndIndependentNodesInPartitionOrder() {
        Operation add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        var nodes = List.of(
                new CompiledNode(new NodeId(0), add,
                        List.of(new ValueId(0), new ValueId(0)), List.of(new ValueId(2))),
                new CompiledNode(new NodeId(1), add,
                        List.of(new ValueId(2), new ValueId(1)), List.of(new ValueId(3))),
                new CompiledNode(new NodeId(2), add,
                        List.of(new ValueId(2), new ValueId(0)), List.of(new ValueId(4))),
                new CompiledNode(new NodeId(3), add,
                        List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(5))));
        Prepared prepared = prepare(CpuPointwiseAddCandidateSourceTest.graph(
                DataType.INT32, io.github.pho001.synaptik.model.shape.Shape.of(1), nodes, 6));
        try (prepared.workers) {
            prepared.byValue.get(new ValueId(0)).segment().set(INT, 0, 3);
            prepared.byValue.get(new ValueId(1)).segment().set(INT, 0, 5);
            prepared.executable.bind(prepared.state).execute();
            assertAll(
                    () -> assertEquals(6, prepared.byValue.get(new ValueId(2)).segment().get(INT, 0)),
                    () -> assertEquals(11, prepared.byValue.get(new ValueId(3)).segment().get(INT, 0)),
                    () -> assertEquals(9, prepared.byValue.get(new ValueId(4)).segment().get(INT, 0)),
                    () -> assertEquals(8, prepared.byValue.get(new ValueId(5)).segment().get(INT, 0)),
                    () -> assertEquals(BufferAccess.READ_WRITE,
                            prepared.executable.bufferAccess(prepared.selectionByValue.get(
                                    new ValueId(2)))));
        } finally { prepared.state.close(); }
    }

    private double[] executeDoubles(double[] left, double[] right) {
        Prepared prepared = prepare(DataType.FLOAT64, left.length);
        try (prepared.workers) {
            for (int index = 0; index < left.length; index++) {
                prepared.buffers.get(0).segment().set(DOUBLE, (long) index * Double.BYTES, left[index]);
                prepared.buffers.get(1).segment().set(DOUBLE, (long) index * Double.BYTES, right[index]);
            }
            prepared.executable.bind(prepared.state).execute();
            double[] result = new double[left.length];
            for (int index = 0; index < result.length; index++) result[index] =
                    prepared.buffers.get(3).segment().get(DOUBLE, (long) index * Double.BYTES);
            return result;
        } finally { prepared.state.close(); }
    }

    private float[] executeFloats(float[] left, float[] right) {
        Prepared prepared = prepare(DataType.FLOAT32, left.length);
        try (prepared.workers) {
            for (int index = 0; index < left.length; index++) {
                prepared.buffers.get(0).segment().set(FLOAT, (long) index * Float.BYTES, left[index]);
                prepared.buffers.get(1).segment().set(FLOAT, (long) index * Float.BYTES, right[index]);
            }
            prepared.executable.bind(prepared.state).execute();
            float[] result = new float[left.length];
            for (int index = 0; index < result.length; index++) result[index] =
                    prepared.buffers.get(3).segment().get(FLOAT, (long) index * Float.BYTES);
            return result;
        } finally { prepared.state.close(); }
    }

    private int[] executeInts(int[] left, int[] right) {
        Prepared prepared = prepare(DataType.INT32, left.length);
        try (prepared.workers) {
            for (int index = 0; index < left.length; index++) {
                prepared.buffers.get(0).segment().set(INT, (long) index * Integer.BYTES, left[index]);
                prepared.buffers.get(1).segment().set(INT, (long) index * Integer.BYTES, right[index]);
            }
            prepared.executable.bind(prepared.state).execute();
            int[] result = new int[left.length];
            for (int index = 0; index < result.length; index++) result[index] =
                    prepared.buffers.get(3).segment().get(INT, (long) index * Integer.BYTES);
            return result;
        } finally { prepared.state.close(); }
    }

    private long[] executeLongs(long[] left, long[] right) {
        Prepared prepared = prepare(DataType.INT64, left.length);
        try (prepared.workers) {
            for (int index = 0; index < left.length; index++) {
                prepared.buffers.get(0).segment().set(LONG, (long) index * Long.BYTES, left[index]);
                prepared.buffers.get(1).segment().set(LONG, (long) index * Long.BYTES, right[index]);
            }
            prepared.executable.bind(prepared.state).execute();
            long[] result = new long[left.length];
            for (int index = 0; index < result.length; index++) result[index] =
                    prepared.buffers.get(3).segment().get(LONG, (long) index * Long.BYTES);
            return result;
        } finally { prepared.state.close(); }
    }

    private Prepared prepare(DataType type, long elementCount) {
        return prepare(CpuPointwiseAddCandidateSourceTest.chain(
                type, io.github.pho001.synaptik.model.shape.Shape.of(elementCount)));
    }

    private Prepared prepare(
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<CpuPortableAnalysisInputs>
                    context) {
        DataType type = context.values().getFirst().descriptor().dataType();
        BackendPartitionAnalysis<CpuPortablePreparationPlan> analysis =
                new CpuPortablePartitionPreparer(new CpuPointwiseAddCandidateSource())
                        .analyze(context);
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (int index = 0; index < analysis.requirements().size(); index++) {
            var requirement = (PreparationResourceRequirement.Buffer)
                    analysis.requirements().get(index);
            var slot = new BufferSlot(index);
            entries.add(new PreparedMemoryPlan.BufferEntry(
                    slot, requirement.byteSize(), requirement.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(requirement, slot, index));
        }
        var memoryPlan = new PreparedMemoryPlan(entries, List.of());
        var workers = new CpuWorkerGroup(1);
        var executable = (CpuPortablePreparedExecutable) new CpuPortablePartitionFinalizer(
                artifactRoot, workers).finalizePartition(
                        new BackendPartitionFinalization<>(analysis, memoryPlan, assignments));
        var buffers = new ArrayList<CpuNativeBuffer>();
        var byValue = new java.util.HashMap<ValueId, CpuNativeBuffer>();
        var selectionByValue = new java.util.HashMap<ValueId, Integer>();
        var bindings = new ArrayList<List<BufferRepresentationBinding>>();
        for (int index = 0; index < memoryPlan.buffers().size(); index++) {
            var entry = memoryPlan.buffers().get(index);
            var buffer = CpuNativeBuffer.allocate(type, entry.byteSize(), entry.byteAlignment());
            buffers.add(buffer);
            var requirement = (PreparationResourceRequirement.Buffer)
                    analysis.requirements().get(index);
            byValue.put(requirement.valueId(), buffer);
            selectionByValue.put(requirement.valueId(), index);
            bindings.add(List.of(new BufferRepresentationBinding(
                    buffer, RunResourceOwnership.RUN_OWNED)));
        }
        var state = new RunState(memoryPlan, bindings, List.of());
        return new Prepared(executable, state, buffers, byValue, selectionByValue, workers);
    }

    private record Prepared(CpuPortablePreparedExecutable executable, RunState state,
            List<CpuNativeBuffer> buffers, java.util.Map<ValueId, CpuNativeBuffer> byValue,
            java.util.Map<ValueId, Integer> selectionByValue, CpuWorkerGroup workers) {}

    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());
}
