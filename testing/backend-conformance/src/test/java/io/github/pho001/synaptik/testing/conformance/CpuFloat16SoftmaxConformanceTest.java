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

/** FLOAT16 softmax provider, prepare, binding, and execution on exact heap/native strided regions. */
final class CpuFloat16SoftmaxConformanceTest {
    private static final Shape SHAPE = Shape.of(2, 3);
    private static final float[] VALUES = {-1, 0, 1, 1, 2, 3};
    private static final short SENTINEL = Float.floatToFloat16(-71);

    @Test void bothKindsExecuteEveryCarrierPairAndBothStridedAxes() {
        for (SoftmaxKind kind : SoftmaxKind.values()) for (int axis = 0; axis < 2; axis++)
            for (boolean nativeInput : List.of(false, true))
                for (boolean nativeOutput : List.of(false, true))
                    execute(kind, axis, nativeInput, nativeOutput);
    }

    private static void execute(SoftmaxKind kind, int axis, boolean nativeInput, boolean nativeOutput) {
        var input = descriptor(new long[] {7, 2});
        var output = descriptor(new long[] {9, 2});
        var operation = new Operation(kind, new SoftmaxAttrs(axis));
        var provider = new CpuCapabilityProvider();
        assertTrue(provider.supports(new OperationCapabilityQuery(operation, List.of(input), List.of(output))));
        var node = new CompiledNode(new NodeId(0), operation,
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var context = new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(new ValueId(0), input), new GraphValue(new ValueId(1), output)),
                List.of(new LogicalMemoryRequirement(new ValueId(0), input, Optional.empty(),
                                List.of(partition), false),
                        new LogicalMemoryRequirement(new ValueId(1), output, Optional.of(partition),
                                List.of(), true)), Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(nativeInput ? CarrierAccess.MEMORY_SEGMENT
                        : CarrierAccess.SHORT_ARRAY, nativeOutput ? CarrierAccess.MEMORY_SEGMENT
                        : CarrierAccess.SHORT_ARRAY)));
        var executable = finalizePortable(new CpuPartitionPreparer().analyze(context));
        try (Arena arena = Arena.ofConfined(); var integration = CpuBackendIntegration.open()) {
            int inputSpan = Math.toIntExact(input.layout().orElseThrow().referencedElementSpan());
            int outputSpan = Math.toIntExact(output.layout().orElseThrow().referencedElementSpan());
            MemorySegment inBacking = backing(arena, nativeInput, inputSpan + 2);
            MemorySegment outBacking = backing(arena, nativeOutput, outputSpan + 2);
            MemorySegment in = inBacking.asSlice(Short.BYTES, inputSpan * (long) Short.BYTES);
            MemorySegment out = outBacking.asSlice(Short.BYTES, outputSpan * (long) Short.BYTES);
            for (int i = 0; i < inputSpan + 2; i++) inBacking.setAtIndex(ValueLayout.JAVA_SHORT, i,
                    Float.floatToFloat16(Float.NaN));
            for (int i = 0; i < VALUES.length; i++) in.setAtIndex(ValueLayout.JAVA_SHORT,
                    address(input, i), Float.floatToFloat16(VALUES[i]));
            reset(outBacking);
            try (RunState state = new RunState(executable.memoryPlan(), List.of(
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            DataType.FLOAT16, inputSpan, in)), RunResourceOwnership.BORROWED)),
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            DataType.FLOAT16, outputSpan, out)), RunResourceOwnership.BORROWED))), List.of())) {
                var bound = executable.bind(state);
                bound.execute();
                for (int i = 0; i < VALUES.length; i++) {
                    double actual = Float.float16ToFloat(out.getAtIndex(ValueLayout.JAVA_SHORT, address(output, i)));
                    // Independent stable formula; one final half rounding, with a 1e-3 finite oracle.
                    assertEquals(expected(kind, axis, i), actual, 1e-3,
                            kind + " axis=" + axis + " native=" + nativeInput + "/" + nativeOutput);
                }
                for (int i = 0; i < outputSpan + 2; i++) {
                    final int address = i - 1;
                    if (java.util.stream.IntStream.range(0, VALUES.length)
                            .noneMatch(logical -> address(output, logical) == address))
                        assertEquals(SENTINEL, outBacking.getAtIndex(ValueLayout.JAVA_SHORT, i));
                }
                for (float invalid : new float[] {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
                    reset(outBacking);
                    in.setAtIndex(ValueLayout.JAVA_SHORT, address(input, 5), Float.floatToFloat16(invalid));
                    assertThrows(IllegalArgumentException.class, bound::execute);
                    for (short value : outBacking.toArray(ValueLayout.JAVA_SHORT)) assertEquals(SENTINEL, value);
                }
                in.setAtIndex(ValueLayout.JAVA_SHORT, address(input, 5), Float.floatToFloat16(VALUES[5]));
                bound.execute();
                assertEquals(expected(kind, axis, 5),
                        Float.float16ToFloat(out.getAtIndex(ValueLayout.JAVA_SHORT, address(output, 5))), 1e-3);
            }
        }
    }

    private static double expected(SoftmaxKind kind, int axis, int position) {
        int row = position / 3, column = position % 3;
        double maximum = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < (axis == 0 ? 2 : 3); c++)
            maximum = Math.max(maximum, VALUES[axis == 0 ? c * 3 + column : row * 3 + c]);
        double sum = 0;
        for (int c = 0; c < (axis == 0 ? 2 : 3); c++)
            sum += Math.exp(VALUES[axis == 0 ? c * 3 + column : row * 3 + c] - maximum);
        double result = kind == SoftmaxKind.SOFTMAX
                ? Math.exp(VALUES[position] - maximum) / sum : VALUES[position] - maximum - Math.log(sum);
        return Float.float16ToFloat(Float.floatToFloat16((float) result));
    }

    private static TensorDescriptor descriptor(long[] strides) {
        return new TensorDescriptor(DataType.FLOAT16, SHAPE,
                Optional.of(LayoutDescriptor.of(SHAPE, strides, 1, true)), false);
    }
    private static long address(TensorDescriptor descriptor, int logical) {
        var layout = descriptor.layout().orElseThrow();
        return layout.storageOffset() + logical / 3 * layout.strides()[0] + logical % 3 * layout.strides()[1];
    }
    private static MemorySegment backing(Arena arena, boolean nativeStorage, int count) {
        return nativeStorage ? arena.allocate(count * (long) Short.BYTES, Short.BYTES)
                : MemorySegment.ofArray(new short[count]);
    }
    private static void reset(MemorySegment output) {
        for (long i = 0; i < output.byteSize() / Short.BYTES; i++)
            output.setAtIndex(ValueLayout.JAVA_SHORT, i, SENTINEL);
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
