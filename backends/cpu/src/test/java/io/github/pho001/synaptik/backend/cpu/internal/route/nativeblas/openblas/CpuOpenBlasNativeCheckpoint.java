package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;
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
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Explicit real-native checkpoint for one caller-supplied CPU-qualified OpenBLAS binary. */
public final class CpuOpenBlasNativeCheckpoint {
    private CpuOpenBlasNativeCheckpoint() { }

    /**
     * Runs direct and one-copy prepared routes and restores the caller-coordinated thread count.
     *
     * @param args exactly one absolute compatible OpenBLAS shared-library path
     * @throws Throwable if input, loading, thread control, preparation, execution, numerical
     *     validation, restoration, verification, or cleanup fails
     */
    public static void main(String[] args) throws Throwable {
        if (args.length != 1) throw new IllegalArgumentException(
                "expected exactly one absolute OpenBLAS library path argument");
        Path path = Path.of(args[0]);
        if (!path.isAbsolute() || !Files.isRegularFile(path)) throw new IllegalArgumentException(
                "OpenBLAS library path must be an existing absolute file: " + path);
        try (OpenBlasLibrary library = OpenBlasLibrary.open(path)) {
            int original = library.threadCount();
            Throwable primary = null;
            try {
                library.setThreadCount(1);
                require(library.threadCount() == 1, "OpenBLAS thread count did not become one");
                CpuOpenBlasInvocation invocation =
                        CpuOpenBlasRouteSelector.borrowedInvocation(library);
                for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                    checkFinite(type, false, invocation);
                    checkFinite(type, true, invocation);
                    checkSpecial(type, invocation);
                }
            } catch (Throwable failure) {
                primary = failure;
                throw failure;
            } finally {
                try {
                    library.setThreadCount(original);
                    require(library.threadCount() == original,
                            "OpenBLAS thread count was not restored");
                } catch (Throwable restorationFailure) {
                    if (primary == null) throw restorationFailure;
                    if (restorationFailure != primary) primary.addSuppressed(restorationFailure);
                }
            }
            System.out.println("CPU OpenBLAS native checkpoint passed; restored thread count "
                    + original);
        }
    }

    private static void checkFinite(DataType type, boolean copyLeft,
            CpuOpenBlasInvocation invocation) {
        double[] left = {1, -2, 3, 4, 5, -6};
        double[] right = {7, 8, -9, 10, 11, 12};
        double[] actual = execute(type, 2, 2, 3, left, right, copyLeft, invocation);
        checkFiniteOracle(type, 2, 2, 3, left, right, actual);
    }

    private static void checkSpecial(DataType type, CpuOpenBlasInvocation invocation) {
        double nan = execute(type, 1, 1, 3, new double[] {Double.NaN, 0, 0},
                new double[] {1, 1, 1}, false, invocation)[0];
        double positiveInfinity = execute(type, 1, 1, 2,
                new double[] {Double.POSITIVE_INFINITY, 1}, new double[] {1, 1}, false,
                invocation)[0];
        double negativeInfinity = execute(type, 1, 1, 2,
                new double[] {Double.NEGATIVE_INFINITY, 1}, new double[] {1, 1}, false,
                invocation)[0];
        double positiveZero = execute(type, 1, 1, 2, new double[] {0, 0},
                new double[] {1, 2}, false, invocation)[0];
        require(Double.isNaN(nan), type + " did not preserve the one-NaN product class");
        require(positiveInfinity == Double.POSITIVE_INFINITY,
                type + " did not preserve sole positive infinity");
        require(negativeInfinity == Double.NEGATIVE_INFINITY,
                type + " did not preserve sole negative infinity");
        require(Double.doubleToRawLongBits(positiveZero) == 0L,
                type + " did not produce unambiguous positive zero");
    }

    private static double[] execute(DataType type, int m, int n, int k, double[] leftValues,
            double[] rightValues, boolean copyLeft, CpuOpenBlasInvocation invocation) {
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                new CpuPartitionPreparer().analyze(context(type, m, n, k, copyLeft));
        require(analysis.plan().route() == CpuPartitionPreparationPlan.Route.OPENBLAS,
                "analysis did not select OpenBLAS");
        require(analysis.plan().openBlasPlan().orElseThrow().representation()
                        == (copyLeft ? CpuOpenBlasRoutePlan.Representation.COPY_LEFT
                                : CpuOpenBlasRoutePlan.Representation.DIRECT),
                "analysis selected the wrong representation");
        PreparedExecutable executable = finalize(analysis, invocation);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left;
            if (copyLeft) {
                double[] physical = stridedPhysical(m, k, leftValues);
                left = type == DataType.FLOAT32 ? MemorySegment.ofArray(toFloats(physical))
                        : MemorySegment.ofArray(physical);
            } else {
                left = arena.allocate(Math.multiplyExact((long) m * k, type.byteWidth()),
                        type.byteWidth());
                put(type, left, leftValues);
            }
            MemorySegment right = arena.allocate(Math.multiplyExact((long) k * n,
                    type.byteWidth()), type.byteWidth());
            MemorySegment output = arena.allocate(Math.multiplyExact((long) m * n,
                    type.byteWidth()), type.byteWidth());
            put(type, right, rightValues);
            List<WorkspaceRepresentation> workspaces = executable.memoryPlan().workspaces().stream()
                    .map(entry -> (WorkspaceRepresentation) CpuContiguousWorkspace.allocate(
                            entry.byteSize(), entry.byteAlignment())).toList();
            try (RunState state = state(executable.memoryPlan(), type, left, right, output,
                    workspaces)) {
                executable.bind(state).execute();
            }
            return get(type, output);
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,
            int m, int n, int k, boolean copyLeft) {
        var node = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        Shape leftShape = Shape.of(m, k);
        TensorDescriptor left = new TensorDescriptor(type, leftShape, Optional.of(copyLeft
                ? LayoutDescriptor.of(leftShape, new long[] {1, m}, 0, true)
                : LayoutDescriptor.contiguous(leftShape)), false);
        List<TensorDescriptor> descriptors = List.of(left, descriptor(type, Shape.of(k, n)),
                descriptor(type, Shape.of(m, n)));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < 3; index++) {
            ValueId id = new ValueId(index);
            values.add(new GraphValue(id, descriptors.get(index)));
            memory.add(new LogicalMemoryRequirement(id, descriptors.get(index),
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
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, storage,
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        1_000, 10, 100, 1, 1, 1, 1, 1));
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
        PreparedMemoryPlan memory = new PreparedMemoryPlan(buffers, workspaces);
        return new CpuPartitionFinalizer(Optional.empty(), Optional.empty(),
                Optional.of(invocation)).finalizePartition(
                        new BackendPartitionFinalization<>(analysis, memory, assignments));
    }

    private static RunState state(PreparedMemoryPlan plan, DataType type, MemorySegment left,
            MemorySegment right, MemorySegment output, List<WorkspaceRepresentation> workspaces) {
        return new RunState(plan, List.of(binding(type, left), binding(type, right),
                binding(type, output)), workspaces);
    }

    private static List<BufferRepresentationBinding> binding(DataType type,
            MemorySegment segment) {
        return List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(
                new MemorySegmentStorage(type, segment.byteSize() / type.byteWidth(), segment)),
                RunResourceOwnership.BORROWED));
    }

    private static void checkFiniteOracle(DataType type, int m, int n, int k,
            double[] left, double[] right, double[] actual) {
        double unitRoundoff = type == DataType.FLOAT32 ? Math.scalb(1.0, -24)
                : Math.scalb(1.0, -53);
        double gamma = (2.0 * k * unitRoundoff) / (1.0 - 2.0 * k * unitRoundoff);
        require(2.0 * k * unitRoundoff < 1.0, "checkpoint gamma is undefined");
        for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
            BigDecimal exact = BigDecimal.ZERO;
            BigDecimal sumAbs = BigDecimal.ZERO;
            for (int inner = 0; inner < k; inner++) {
                BigDecimal product = BigDecimal.valueOf(left[row * k + inner])
                        .multiply(BigDecimal.valueOf(right[inner * n + column]));
                exact = exact.add(product); sumAbs = sumAbs.add(product.abs());
            }
            double rounded = type == DataType.FLOAT32 ? (float) exact.doubleValue()
                    : exact.doubleValue();
            double ulp = type == DataType.FLOAT32 ? Math.ulp((float) rounded) : Math.ulp(rounded);
            double tolerance = Math.max(gamma * sumAbs.doubleValue(), 4.0 * ulp);
            double error = Math.abs(actual[row * n + column] - rounded);
            require(error <= tolerance, type + " finite result exceeded checkpoint tolerance: "
                    + error + " > " + tolerance);
        }
    }

    private static double[] stridedPhysical(int rows, int columns, double[] logical) {
        double[] physical = new double[logical.length];
        for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
            physical[column * rows + row] = logical[row * columns + column];
        }
        return physical;
    }

    private static float[] toFloats(double[] values) {
        float[] result = new float[values.length];
        for (int index = 0; index < values.length; index++) result[index] = (float) values[index];
        return result;
    }

    private static void put(DataType type, MemorySegment segment, double[] values) {
        for (int index = 0; index < values.length; index++) {
            if (type == DataType.FLOAT32) segment.setAtIndex(ValueLayout.JAVA_FLOAT, index,
                    (float) values[index]);
            else segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, values[index]);
        }
    }

    private static double[] get(DataType type, MemorySegment segment) {
        int count = Math.toIntExact(segment.byteSize() / type.byteWidth());
        double[] result = new double[count];
        for (int index = 0; index < count; index++) result[index] = type == DataType.FLOAT32
                ? segment.getAtIndex(ValueLayout.JAVA_FLOAT, index)
                : segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
        return result;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
