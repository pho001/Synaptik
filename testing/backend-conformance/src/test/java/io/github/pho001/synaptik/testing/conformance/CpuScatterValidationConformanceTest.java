package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Direct CPU preparation and invocation of rank-mismatched Scatter Add, without NN or Engine. */
final class CpuScatterValidationConformanceTest {
    @Test void rankOneInt32AndInt64IndicesConformAcrossHeapAndNativeStorage() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64))
            for (boolean nativeStorage : List.of(false, true)) execute(type, nativeStorage, false);
    }

    @Test void lateInvalidIndicesLeaveAllTargetBytesUntouchedAndPermitReuseAfterRepair() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64))
            for (boolean nativeStorage : List.of(false, true)) execute(type, nativeStorage, true);
    }

    private static void execute(DataType indexType, boolean nativeStorage, boolean checkFailures) {
        PreparedExecutable executable = prepare(indexType, nativeStorage);
        try (Arena arena = Arena.ofConfined(); CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            float[] base = {10, 20, 30, 40, 50, 60}, updates = {1, 2, 3, 4, 5, 6};
            MemorySegment indexBacking = nativeStorage
                    ? arena.allocate(8L * indexType.byteWidth(), indexType.byteWidth())
                    : indexType == DataType.INT32 ? MemorySegment.ofArray(new int[8])
                            : MemorySegment.ofArray(new long[8]);
            indexBacking.fill((byte) 0x7f);
            MemorySegment indices = indexBacking.asSlice(indexType.byteWidth(), 6L * indexType.byteWidth());
            int[] logical = {2, 0, 2};
            for (int i = 0; i < logical.length; i++) putIndex(indexType, indices, 1 + i * 2, logical[i]);
            byte[] originalIndices = indexBacking.toArray(ValueLayout.JAVA_BYTE);
            MemorySegment outputBacking = nativeStorage ? arena.allocate(8L * Float.BYTES, Float.BYTES)
                    : MemorySegment.ofArray(new float[8]);
            outputBacking.fill((byte) 0x5a);
            MemorySegment output = outputBacking.asSlice(Float.BYTES, 6L * Float.BYTES);
            var segments = List.of(MemorySegment.ofArray(base), indices, MemorySegment.ofArray(updates), output);
            var types = List.of(DataType.FLOAT32, indexType, DataType.FLOAT32, DataType.FLOAT32);
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            for (int i = 0; i < segments.size(); i++) {
                var segment = segments.get(i);
                bindings.add(List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                        types.get(i), segment.byteSize() / types.get(i).byteWidth(), segment)), RunResourceOwnership.BORROWED)));
            }
            try (RunState state = new RunState(executable.memoryPlan(), bindings, List.of())) {
                var bound = executable.bind(state);
                float[] expected = base.clone();
                // Independent direct scatter oracle: each update contributes once, duplicates included.
                for (int row = 0; row < 2; row++)
                    for (int i = 0; i < logical.length; i++)
                        expected[row * 3 + logical[i]] += updates[row * logical.length + i];
                for (int repeat = 0; repeat < 2; repeat++) {
                    bound.execute();
                    assertArrayEquals(expected, output.toArray(ValueLayout.JAVA_FLOAT));
                }
                if (checkFailures) for (int bad : new int[] {-1, 3}) {
                    putIndex(indexType, indices, 5, bad);
                    outputBacking.fill((byte) 0x5a);
                    byte[] sentinel = outputBacking.toArray(ValueLayout.JAVA_BYTE);
                    var failure = assertThrows(IndexOutOfBoundsException.class, bound::execute);
                    assertEquals("SCATTER_ADD index at logical position 2 for data axis 1 is out of bounds: value="
                            + bad + ", extent=3", failure.getMessage());
                    assertArrayEquals(sentinel, outputBacking.toArray(ValueLayout.JAVA_BYTE));
                    putIndex(indexType, indices, 5, 2);
                    bound.execute();
                    assertArrayEquals(expected, output.toArray(ValueLayout.JAVA_FLOAT));
                }
                assertEquals(0x5a5a5a5a, outputBacking.getAtIndex(ValueLayout.JAVA_INT, 0));
                assertEquals(0x5a5a5a5a, outputBacking.getAtIndex(ValueLayout.JAVA_INT, 7));
                assertArrayEquals(originalIndices, indexBacking.toArray(ValueLayout.JAVA_BYTE));
                assertArrayEquals(new float[] {10, 20, 30, 40, 50, 60}, base);
                assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, updates);
            }
        }
    }

    private static PreparedExecutable prepare(DataType indexType, boolean nativeStorage) {
        var operation = new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1));
        var data = descriptor(DataType.FLOAT32, Shape.of(2, 3), new long[] {3, 1}, 0);
        var indices = descriptor(indexType, Shape.of(3), new long[] {2}, 1);
        var descriptors = List.of(data, indices, data, data);
        assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(operation,
                descriptors.subList(0, 3), List.of(data))));
        var node = new CompiledNode(new NodeId(0), operation, List.of(new ValueId(0), new ValueId(1),
                new ValueId(2)), List.of(new ValueId(3)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var values = new ArrayList<GraphValue>();
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < descriptors.size(); i++) {
            var id = new ValueId(i);
            values.add(new GraphValue(id, descriptors.get(i)));
            requirements.add(new LogicalMemoryRequirement(id, descriptors.get(i),
                    i == 3 ? Optional.of(partition) : Optional.empty(), i == 3 ? List.of() : List.of(partition), i == 3));
        }
        var context = new PrepareContext<>(partition, List.of(node), values, requirements, Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.FLOAT_ARRAY,
                        nativeStorage ? CarrierAccess.MEMORY_SEGMENT
                                : indexType == DataType.INT32 ? CarrierAccess.INT_ARRAY : CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, nativeStorage ? CarrierAccess.MEMORY_SEGMENT : CarrierAccess.FLOAT_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            var buffer = assertInstanceOf(PreparationResourceRequirement.Buffer.class, requirement);
            var slot = new BufferSlot(buffers.size());
            buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(), buffer.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot, buffers.size() - 1, 0));
        }
        return new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(
                analysis, new PreparedMemoryPlan(buffers, List.of()), assignments)).executable();
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, long[] strides, long offset) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

    private static void putIndex(DataType type, MemorySegment segment, int position, int value) {
        if (type == DataType.INT32) segment.setAtIndex(ValueLayout.JAVA_INT, position, value);
        else segment.setAtIndex(ValueLayout.JAVA_LONG, position, value);
    }
}
