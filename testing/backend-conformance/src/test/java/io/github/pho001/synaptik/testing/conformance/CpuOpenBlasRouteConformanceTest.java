package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasInvocation;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
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

/** Native-free staged-boundary conformance for the borrowed OpenBLAS CPU route. */
final class CpuOpenBlasRouteConformanceTest {
    @Test void preparesFinalizesBindsAndExecutesDirectFloat32AndFloat64Routes() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                    new CpuPartitionPreparer().analyze(context(type));
            assertEquals(CpuPartitionPreparationPlan.Route.OPENBLAS, analysis.plan().route());
            var fake = new ComputingInvocation(type);
            PreparedExecutable executable = finalize(analysis, fake);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment left = arena.allocate(6L * type.byteWidth(), type.byteWidth());
                MemorySegment right = arena.allocate(6L * type.byteWidth(), type.byteWidth());
                MemorySegment output = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                double[] leftValues = {1, 2, 3, 4, 5, 6};
                double[] rightValues = {7, 8, 9, 10, 11, 12};
                put(type, left, leftValues); put(type, right, rightValues);
                try (RunState state = state(executable.memoryPlan(), type, left, right, output)) {
                    executable.bind(state).execute();
                }
                assertAll(() -> assertArrayEquals(new double[] {58, 64, 139, 154},
                                get(type, output)),
                        () -> assertEquals(1, fake.calls), () -> assertEquals(2, fake.m),
                        () -> assertEquals(2, fake.n), () -> assertEquals(3, fake.k),
                        () -> assertEquals(1.0, fake.alpha), () -> assertEquals(0.0, fake.beta),
                        () -> assertSame(left, fake.left), () -> assertSame(right, fake.right),
                        () -> assertSame(output, fake.output));
            }
        }
    }

    @Test void executesOneCopyBeforeTheSameTypedNativeCallForBothFloatTypes() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                    new CpuPartitionPreparer().analyze(context(type, true));
            assertEquals(CpuOpenBlasRoutePlan.Representation.COPY_LEFT,
                    analysis.plan().openBlasPlan().orElseThrow().representation());
            var fake = new ComputingInvocation(type);
            PreparedExecutable executable = finalize(analysis, fake);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment left = type == DataType.FLOAT32
                        ? MemorySegment.ofArray(new float[] {1, 4, 2, 5, 3, 6})
                        : MemorySegment.ofArray(new double[] {1, 4, 2, 5, 3, 6});
                MemorySegment right = arena.allocate(6L * type.byteWidth(), type.byteWidth());
                MemorySegment output = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                put(type, right, new double[] {7, 8, 9, 10, 11, 12});
                var workspaceEntry = executable.memoryPlan().workspaces().getFirst();
                var workspace = CpuContiguousWorkspace.allocate(workspaceEntry.byteSize(),
                        workspaceEntry.byteAlignment());
                try (RunState state = state(executable.memoryPlan(), type, left, right, output,
                        List.of(workspace))) {
                    executable.bind(state).execute();
                }
                assertAll(() -> assertArrayEquals(new double[] {58, 64, 139, 154},
                                get(type, output)),
                        () -> assertEquals(1, fake.calls), () -> assertTrue(fake.left.isNative()),
                        () -> assertNotSame(left, fake.left), () -> assertSame(right, fake.right),
                        () -> assertSame(output, fake.output));
            }
        }
    }

    @Test void rejectsCopyCarrierDriftBeforeCopyOrProviderMutation() {
        DataType type = DataType.FLOAT32;
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                new CpuPartitionPreparer().analyze(context(type, true));
        var fake = new ComputingInvocation(type);
        PreparedExecutable executable = finalize(analysis, fake);
        try (Arena arena = Arena.ofConfined()) {
            long leftBytes = 6L * type.byteWidth();
            long rightBytes = 6L * type.byteWidth();
            long outputBytes = 4L * type.byteWidth();
            MemorySegment left = arena.allocate(leftBytes, type.byteWidth());
            MemorySegment right = arena.allocate(rightBytes, type.byteWidth());
            MemorySegment output = arena.allocate(outputBytes, type.byteWidth());
            var entry = executable.memoryPlan().workspaces().getFirst();
            var workspace = CpuContiguousWorkspace.allocate(entry.byteSize(),
                    entry.byteAlignment());
            try (RunState state = state(executable.memoryPlan(), type, left, right, output,
                    List.of(workspace))) {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
            }
            assertAll(() -> assertEquals(0, fake.calls),
                    () -> assertArrayEquals(new double[] {0, 0, 0, 0}, get(type, output)));
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type) {
        return context(type, false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,
            boolean copyLeft) {
        var node = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        Shape leftShape = Shape.of(2, 3);
        TensorDescriptor left = copyLeft ? new TensorDescriptor(type, leftShape, Optional.of(
                LayoutDescriptor.of(leftShape, new long[] {1, 2}, 0, true)), false)
                : descriptor(type, leftShape);
        List<TensorDescriptor> descriptors = List.of(left,
                descriptor(type, Shape.of(3, 2)), descriptor(type, Shape.of(2, 2)));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = descriptors.get(index);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    index == 2 ? Optional.of(partition) : Optional.empty(),
                    index < 2 ? List.of(partition) : List.of(), index == 2));
        }
        int width = type.byteWidth();
        var carriers = copyLeft ? List.of(type == DataType.FLOAT32
                        ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : java.util.Collections.nCopies(3,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        var storage = copyLeft ? List.of(CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN,
                new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width),
                new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width))
                : java.util.Collections.nCopies(3,
                        new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width));
        var policy = copyLeft ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                1, 1, 10, 0, 2, 1_000_000, 0, 0)
                : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED;
        var inputs = new CpuPartitionAnalysisInputs(false, carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE,
                storage,
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        100, 2, 10, 1, 1, 1, 1, 1));
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(), inputs);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static PreparedExecutable finalize(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuOpenBlasInvocation invocation) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                var slot = new BufferSlot(buffers.size());
                buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                        buffer.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot,
                        buffers.size() - 1));
            } else if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                var slot = new WorkspaceSlot(workspaces.size());
                workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, workspace.byteSize(),
                        workspace.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(workspace, slot,
                        workspaces.size() - 1));
            }
        }
        var memory = new PreparedMemoryPlan(buffers, workspaces);
        return new CpuPartitionFinalizer(Optional.empty(), Optional.empty(),
                Optional.of(invocation)).finalizePartition(
                        new BackendPartitionFinalization<>(analysis, memory, assignments));
    }

    private static RunState state(PreparedMemoryPlan plan, DataType type, MemorySegment left,
            MemorySegment right, MemorySegment output) {
        return state(plan, type, left, right, output, List.of());
    }

    private static RunState state(PreparedMemoryPlan plan, DataType type, MemorySegment left,
            MemorySegment right, MemorySegment output,
            List<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation> workspaces) {
        return new RunState(plan, List.of(binding(type, left), binding(type, right),
                binding(type, output)), workspaces);
    }

    private static List<BufferRepresentationBinding> binding(DataType type,
            MemorySegment segment) {
        var storage = new MemorySegmentStorage(type, segment.byteSize() / type.byteWidth(), segment);
        return List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(storage),
                RunResourceOwnership.BORROWED));
    }

    private static void put(DataType type, MemorySegment segment, double[] values) {
        for (int index = 0; index < values.length; index++) {
            long offset = Math.multiplyExact((long) index, type.byteWidth());
            if (type == DataType.FLOAT32) segment.set(ValueLayout.JAVA_FLOAT, offset,
                    (float) values[index]);
            else segment.set(ValueLayout.JAVA_DOUBLE, offset, values[index]);
        }
    }

    private static double[] get(DataType type, MemorySegment segment) {
        int count = Math.toIntExact(segment.byteSize() / type.byteWidth());
        double[] values = new double[count];
        for (int index = 0; index < count; index++) {
            long offset = Math.multiplyExact((long) index, type.byteWidth());
            values[index] = type == DataType.FLOAT32
                    ? segment.get(ValueLayout.JAVA_FLOAT, offset)
                    : segment.get(ValueLayout.JAVA_DOUBLE, offset);
        }
        return values;
    }

    private static final class ComputingInvocation implements CpuOpenBlasInvocation {
        private final DataType type;
        int calls;
        int m;
        int n;
        int k;
        double alpha;
        double beta;
        MemorySegment left;
        MemorySegment right;
        MemorySegment output;

        ComputingInvocation(DataType type) { this.type = type; }
        @Override public boolean isOpen() { return true; }
        @Override public int threadCount() { return 1; }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            assertEquals(DataType.FLOAT32, type); execute(m, n, k, alpha, a, b, beta, c);
        }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            assertEquals(DataType.FLOAT64, type); execute(m, n, k, alpha, a, b, beta, c);
        }
        private void execute(int m, int n, int k, double alpha, MemorySegment left,
                MemorySegment right, double beta, MemorySegment output) {
            calls++; this.m = m; this.n = n; this.k = k; this.alpha = alpha; this.beta = beta;
            this.left = left; this.right = right; this.output = output;
            double[] a = get(type, left), b = get(type, right), result = new double[m * n];
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                double sum = 0;
                for (int inner = 0; inner < k; inner++) {
                    sum += a[row * k + inner] * b[inner * n + column];
                }
                result[row * n + column] = alpha * sum + beta * result[row * n + column];
            }
            put(type, output, result);
        }
    }
}
