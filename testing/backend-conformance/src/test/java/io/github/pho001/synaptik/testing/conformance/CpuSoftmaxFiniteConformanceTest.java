package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Qualifies finite stable softmax results across supported types and represented layouts. */
final class CpuSoftmaxFiniteConformanceTest {
    @Test void bothKindsAndFourDtypesHandleOverflowUnderflowAndOffsets() {
        for (SoftmaxKind kind : SoftmaxKind.values()) for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var fixture = fixture(kind, type);
            try (var integration = CpuBackendIntegration.open()) {
                MemorySegment input = storage(type), output = storage(type);
                try (RunState state = state(fixture.executable(), integration, type, input, output)) {
                    var bound = fixture.executable().bind(state);
                    for (double[] values : new double[][] {{1000, 1000}, {0, -1000}, {1000, 1008}}) {
                        write(type, input, 0, values[0]); write(type, input, 1, values[1]);
                        bound.execute();
                        double first = read(type, output, 0), second = read(type, output, 1);
                        if (values[0] == 1000 && values[1] == 1000) {
                            double expected = kind == SoftmaxKind.SOFTMAX ? 0.5 : -Math.log(2);
                            assertEquals(expected, first, tolerance(type), kind + "/" + type);
                            assertEquals(expected, second, tolerance(type), kind + "/" + type);
                        } else if (values[0] == 0) {
                            assertEquals(kind == SoftmaxKind.SOFTMAX ? 1 : 0, first);
                            assertEquals(kind == SoftmaxKind.SOFTMAX ? 0 : -1000, second);
                        } else {
                            double denominator = 1 + Math.exp(-8);
                            double expectedFirst = kind == SoftmaxKind.SOFTMAX
                                    ? Math.exp(-8) / denominator : -8 - Math.log(denominator);
                            double expectedSecond = kind == SoftmaxKind.SOFTMAX
                                    ? 1 / denominator : -Math.log(denominator);
                            assertEquals(expectedFirst, first, tolerance(type),
                                    kind + "/" + type + " offset first");
                            assertEquals(expectedSecond, second, tolerance(type),
                                    kind + "/" + type + " offset second");
                        }
                    }
                    if (type == DataType.FLOAT32 || type == DataType.FLOAT64) {
                        double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                        write(type, input, 0, -maximum);
                        write(type, input, 1, maximum);
                        bound.execute();
                        assertEquals(kind == SoftmaxKind.SOFTMAX ? 0 : Double.NEGATIVE_INFINITY,
                                read(type, output, 0));
                        assertEquals(kind == SoftmaxKind.SOFTMAX ? 1 : 0, read(type, output, 1));
                    }
                    write(type, input, 0, 0); write(type, input, 1, 0);
                    bound.execute();
                    assertEquals(kind == SoftmaxKind.SOFTMAX ? 0.5 : -Math.log(2),
                            read(type, output, 0), tolerance(type));
                }
            }
        }
    }

    @Test void parallelStridedNonLastSlicesIgnoreBackingHolesAndReadFreshValuesEveryRun() {
        Shape shape = Shape.of(3, 8);
        var layout = LayoutDescriptor.of(shape, new long[] {2, 8}, 1, true);
        int count = Math.toIntExact(layout.referencedElementSpan());
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (SoftmaxKind kind : SoftmaxKind.values()) for (boolean nativeCarrier : List.of(false, true)) {
                try (var arena = Arena.ofShared(); var workers = new CpuWorkerGroup(4);
                        var integration = CpuBackendIntegration.open()) {
                    CarrierAccess carrier = nativeCarrier ? CarrierAccess.MEMORY_SEGMENT
                            : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
                    var fixture = fixture(kind, type, shape, 0, layout, carrier, Optional.of(workers));
                    MemorySegment input = nativeCarrier
                            ? arena.allocate((long) count * type.byteWidth(), type.byteWidth())
                            : storage(type, count);
                    MemorySegment output = nativeCarrier
                            ? arena.allocate((long) count * type.byteWidth(), type.byteWidth())
                            : storage(type, count);
                    double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                    // Unaddressed cells are deliberately invalid; only represented contributors count.
                    for (int i = 0; i < count; i++) write(type, input, i, Double.NaN);
                    for (int slice = 0; slice < 8; slice++) {
                        write(type, input, 1 + 8 * slice, -maximum);
                        write(type, input, 3 + 8 * slice, maximum);
                        write(type, input, 5 + 8 * slice, maximum);
                    }
                    try (RunState state = state(fixture.executable(), integration, type, input, output)) {
                        var bound = fixture.executable().bind(state);
                        bound.execute();
                        assertTiedSlices(kind, type, output);
                        for (int pass = 0; pass < 2; pass++) {
                            for (int slice = 0; slice < 8; slice++) {
                                write(type, input, 1 + 8 * slice, 0);
                                write(type, input, 3 + 8 * slice, 0);
                                write(type, input, 5 + 8 * slice, 0);
                            }
                            bound.execute();
                            double expected = kind == SoftmaxKind.SOFTMAX ? 1.0 / 3 : -Math.log(3);
                            for (int slice = 0; slice < 8; slice++) for (int member = 0; member < 3; member++)
                                assertEquals(expected, read(type, output, 1 + 8 * slice + 2 * member),
                                        tolerance(type));
                            for (int slice = 0; slice < 8; slice++) {
                                write(type, input, 1 + 8 * slice, -maximum);
                                write(type, input, 3 + 8 * slice, maximum);
                                write(type, input, 5 + 8 * slice, maximum);
                            }
                            bound.execute();
                            assertTiedSlices(kind, type, output);
                        }
                    }
                }
            }
    }

    private static void assertTiedSlices(SoftmaxKind kind, DataType type, MemorySegment output) {
        for (int slice = 0; slice < 8; slice++) {
            double first = read(type, output, 1 + 8 * slice);
            if (kind == SoftmaxKind.SOFTMAX) assertEquals(0L, Double.doubleToRawLongBits(first));
            else assertEquals(Double.NEGATIVE_INFINITY, first);
            double expected = kind == SoftmaxKind.SOFTMAX ? 0.5 : -Math.log(2);
            assertEquals(expected, read(type, output, 3 + 8 * slice), tolerance(type));
            assertEquals(expected, read(type, output, 5 + 8 * slice), tolerance(type));
        }
    }

    private static Fixture fixture(SoftmaxKind kind, DataType type) {
        CarrierAccess carrier = type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY;
        return fixture(kind, type, Shape.of(2), 0, LayoutDescriptor.contiguous(Shape.of(2)),
                carrier, Optional.empty());
    }

    private static Fixture fixture(SoftmaxKind kind, DataType type, Shape shape, int axis,
            LayoutDescriptor layout, CarrierAccess carrier, Optional<CpuWorkerGroup> workers) {
        var descriptor = new TensorDescriptor(type, shape, Optional.of(layout), false);
        var operation = new Operation(kind, new SoftmaxAttrs(axis));
        var node = new CompiledNode(new NodeId(0), operation,
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var context = new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(new ValueId(0), descriptor),
                        new GraphValue(new ValueId(1), descriptor)),
                List.of(new LogicalMemoryRequirement(new ValueId(0), descriptor,
                                Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(new ValueId(1), descriptor,
                                Optional.of(partition), List.of(), true)), Map.of(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(carrier, carrier), workers.isPresent()
                                ? new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1)
                                : PortableExecutionConfig.DEFAULT));
        var analysis = new CpuPartitionPreparer().analyze(context);
        if (workers.isPresent()) assertEquals(4, analysis.plan().selectedRangeCount());
        return new Fixture(finalizePortable(analysis, workers));
    }

    private static RunState state(PreparedExecutable executable, CpuBackendIntegration integration,
            DataType type, MemorySegment input, MemorySegment output) {
        return new RunState(executable.memoryPlan(), List.of(
                List.of(new BufferRepresentationBinding(integration.borrow(
                        new MemorySegmentStorage(type, input.byteSize() / type.byteWidth(), input)),
                        RunResourceOwnership.BORROWED)),
                List.of(new BufferRepresentationBinding(integration.borrow(
                        new MemorySegmentStorage(type, output.byteSize() / type.byteWidth(), output)),
                        RunResourceOwnership.BORROWED))),
                List.of());
    }

    private static MemorySegment storage(DataType type) {
        return storage(type, 2);
    }

    private static MemorySegment storage(DataType type, int count) {
        return switch (type) {
            case FLOAT64 -> MemorySegment.ofArray(new double[count]);
            case FLOAT32 -> MemorySegment.ofArray(new float[count]);
            case BFLOAT16, FLOAT16 -> MemorySegment.ofArray(new short[count]);
            default -> throw new AssertionError(type);
        };
    }

    private static void write(DataType type, MemorySegment storage, int index, double value) {
        switch (type) {
            case FLOAT64 -> storage.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
            case FLOAT32 -> storage.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) value);
            case FLOAT16 -> storage.setAtIndex(ValueLayout.JAVA_SHORT, index,
                    Float.floatToFloat16((float) value));
            case BFLOAT16 -> {
                int bits = Float.floatToRawIntBits((float) value);
                storage.setAtIndex(ValueLayout.JAVA_SHORT, index,
                        (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16));
            }
            default -> throw new AssertionError(type);
        }
    }

    private static double read(DataType type, MemorySegment storage, int index) {
        return switch (type) {
            case FLOAT64 -> storage.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            case FLOAT32 -> storage.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case FLOAT16 -> Float.float16ToFloat(storage.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case BFLOAT16 -> Float.intBitsToFloat(storage.getAtIndex(ValueLayout.JAVA_SHORT, index)
                    << 16);
            default -> throw new AssertionError(type);
        };
    }

    private static double tolerance(DataType type) {
        return type == DataType.BFLOAT16 ? 0.005 : type == DataType.FLOAT16 ? 0.001
                : type == DataType.FLOAT32 ? 1e-6 : 2e-7;
    }

    private static PreparedExecutable finalizePortable(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            Optional<CpuWorkerGroup> workers) {
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
        return new CpuPartitionFinalizer(Optional.empty(), workers).finalizePartition(
                new BackendPartitionFinalization<>(analysis, memory, assignments)).executable();
    }

    private record Fixture(PreparedExecutable executable) { }
}
