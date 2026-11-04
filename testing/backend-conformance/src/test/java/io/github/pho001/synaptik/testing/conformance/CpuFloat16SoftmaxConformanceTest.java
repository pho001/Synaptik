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
import io.github.pho001.synaptik.model.operation.normalization.*;
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
import java.util.*;
import org.junit.jupiter.api.Test;

/** Low floating stable softmax formulas and exact heap/native strided storage guards. */
final class CpuFloat16SoftmaxConformanceTest {
    private static final Shape SHAPE = Shape.of(2, 3);
    private static final float[] VALUES = {-1, 0, 1, 1, 2, 3};

    @Test void bothKindsExecuteEveryCarrierPairAndBothStridedAxes() {
        for (var type : LowPrecisionValues.TYPES)
            for (SoftmaxKind kind : SoftmaxKind.values()) for (int axis = 0; axis < 2; axis++)
            for (boolean nativeInput : List.of(false, true))
                for (boolean nativeOutput : List.of(false, true))
                    execute(type, kind, axis, nativeInput, nativeOutput);
    }

    private static void execute(DataType type, SoftmaxKind kind, int axis, boolean nativeInput, boolean nativeOutput) {
        var input = descriptor(type, new long[] {7, 2});
        var output = descriptor(type, new long[] {9, 2});
        var operation = new Operation(kind, new SoftmaxAttrs(axis));
        var provider = new CpuCapabilityProvider();
        assertTrue(provider.supports(new OperationCapabilityQuery(operation, List.of(input), List.of(output))));
        var node = new CompiledNode(new NodeId(0), operation,
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var heap = type.byteWidth() == 1 ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY;
        var context = new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(new ValueId(0), input), new GraphValue(new ValueId(1), output)),
                List.of(new LogicalMemoryRequirement(new ValueId(0), input, Optional.empty(),
                                List.of(partition), false),
                        new LogicalMemoryRequirement(new ValueId(1), output, Optional.of(partition),
                                List.of(), true)), Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(nativeInput ? CarrierAccess.MEMORY_SEGMENT
                        : heap, nativeOutput ? CarrierAccess.MEMORY_SEGMENT : heap)));
        var executable = finalizePortable(new CpuPartitionPreparer().analyze(context));
        try (Arena arena = Arena.ofConfined(); var integration = CpuBackendIntegration.open()) {
            int inputSpan = Math.toIntExact(input.layout().orElseThrow().referencedElementSpan());
            int outputSpan = Math.toIntExact(output.layout().orElseThrow().referencedElementSpan());
            MemorySegment inBacking = backing(arena, type, nativeInput, inputSpan + 2);
            MemorySegment outBacking = backing(arena, type, nativeOutput, outputSpan + 2);
            MemorySegment in = inBacking.asSlice(type.byteWidth(), inputSpan * (long) type.byteWidth());
            MemorySegment out = outBacking.asSlice(type.byteWidth(), outputSpan * (long) type.byteWidth());
            for (int i = 0; i < inputSpan + 2; i++) LowPrecisionValues.put(type, inBacking, i, Float.NaN);
            for (int i = 0; i < VALUES.length; i++) LowPrecisionValues.put(type, in, address(input, i), VALUES[i]);
            reset(type, outBacking);
            byte[] unchanged = inBacking.toArray(ValueLayout.JAVA_BYTE);
            try (RunState state = new RunState(executable.memoryPlan(), List.of(
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            type, inputSpan, in)), RunResourceOwnership.RUN_OWNED)),
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            type, outputSpan, out)), RunResourceOwnership.RUN_OWNED))), List.of())) {
                var bound = executable.bind(state);
                bound.execute();
                for (int i = 0; i < VALUES.length; i++) {
                    double actual = LowPrecisionValues.decode(type, LowPrecisionValues.raw(type, out, address(output, i)));
                    // Double stable-formula oracle, independently narrowed once to the public dtype.
                    assertEquals(expected(type, kind, axis, i), actual, type == DataType.FLOAT16 ? 1e-3 : 0,
                            type + " " + kind + " axis=" + axis + " native=" + nativeInput + "/" + nativeOutput);
                }
                for (int i = 0; i < outputSpan + 2; i++) {
                    final int address = i - 1;
                    if (java.util.stream.IntStream.range(0, VALUES.length)
                            .noneMatch(logical -> address(output, logical) == address))
                        assertEquals(LowPrecisionValues.encode(type, -71), LowPrecisionValues.raw(type, outBacking, i));
                }
                assertArrayEquals(unchanged, inBacking.toArray(ValueLayout.JAVA_BYTE));
            }
        }
    }

    private static double expected(DataType type, SoftmaxKind kind, int axis, int position) {
        int row = position / 3, column = position % 3;
        double maximum = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < (axis == 0 ? 2 : 3); c++)
            maximum = Math.max(maximum, VALUES[axis == 0 ? c * 3 + column : row * 3 + c]);
        double sum = 0;
        for (int c = 0; c < (axis == 0 ? 2 : 3); c++)
            sum += Math.exp(VALUES[axis == 0 ? c * 3 + column : row * 3 + c] - maximum);
        double result = kind == SoftmaxKind.SOFTMAX
                ? Math.exp(VALUES[position] - maximum) / sum : VALUES[position] - maximum - Math.log(sum);
        return LowPrecisionValues.decode(type, LowPrecisionValues.encode(type, (float) result));
    }

    private static TensorDescriptor descriptor(DataType type, long[] strides) {
        return new TensorDescriptor(type, SHAPE,
                Optional.of(LayoutDescriptor.of(SHAPE, strides, 1, true)), false);
    }
    private static long address(TensorDescriptor descriptor, int logical) {
        var layout = descriptor.layout().orElseThrow();
        return layout.storageOffset() + logical / 3 * layout.strides()[0] + logical % 3 * layout.strides()[1];
    }
    private static MemorySegment backing(Arena arena, DataType type, boolean nativeStorage, int count) {
        return nativeStorage ? arena.allocate(count * (long) type.byteWidth(), type.byteWidth())
                : type.byteWidth() == 1 ? MemorySegment.ofArray(new byte[count]) : MemorySegment.ofArray(new short[count]);
    }
    private static void reset(DataType type, MemorySegment output) {
        for (long i = 0; i < output.byteSize() / type.byteWidth(); i++)
            LowPrecisionValues.put(type, output, i, -71);
    }

    private static PreparedExecutable finalizePortable(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                var slot = new BufferSlot(buffers.size());
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                        buffer.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(
                        buffer, slot, buffers.size() - 1, 0));
            } else if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                var slot = new WorkspaceSlot(workspaces.size());
                workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, workspace.byteSize(),
                        workspace.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(
                        workspace, slot, workspaces.size() - 1));
            }
        }
        var memory = new PreparedMemoryPlan(buffers, workspaces);
        return new CpuPartitionFinalizer().finalizePartition(
                new BackendPartitionFinalization<>(analysis, memory, assignments)).executable();
    }

}
