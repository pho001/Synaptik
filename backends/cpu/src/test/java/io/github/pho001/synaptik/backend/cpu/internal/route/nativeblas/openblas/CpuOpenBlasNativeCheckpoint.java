package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicReference;

/** Explicit real-native checkpoint for one caller-supplied exact OpenBLAS 0.3.34 binary. */
public final class CpuOpenBlasNativeCheckpoint {
    /** Prevents construction of the command-line checkpoint namespace. */
    private CpuOpenBlasNativeCheckpoint() { }

    /**
     * Runs direct and one-copy prepared routes and restores the caller-coordinated thread count.
     *
     * @param args exactly one absolute OpenBLAS 0.3.34 shared-library path
     * @throws Throwable if input, loading, thread control, preparation, execution, numerical
     *     validation, restoration, verification, or cleanup fails
     */
    public static void main(String[] args) throws Throwable {
        if (args.length != 1 || args[0].equals("--auto")) throw new IllegalArgumentException(
                "expected exactly one absolute OpenBLAS 0.3.34 library path argument");
        Path path = ((CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath)
                exactPathSelection(args[0])).value();
        require(path.toRealPath().toString().contains("0.3.34"),
                "checkpoint path does not resolve to OpenBLAS 0.3.34: " + path.toRealPath());
        CpuOpenBlasDiscoverySession session = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.exactAbsolutePath(path));
        require(session.result().status() == CpuOpenBlasDiscoveryResult.Status.LOADED,
                "exact OpenBLAS library did not load");
        CpuOpenBlasInvocation invocation = session.invocation().orElseThrow();
        int original = invocation.threadCount();
        require(original > 0, "OpenBLAS original thread count was not positive");
        CpuConcurrencyBudget budget = new CpuConcurrencyBudget(2);
        CpuOpenBlasCoordinator coordinator = session.transferToCoordinator(budget);
        try {
                CpuOpenBlasQualification qualification = CpuOpenBlasQualifier.qualify(
                        session.result(), coordinator);
                require(qualification.scope()
                                == CpuOpenBlasQualification.Scope.PERSISTENT_BINARY,
                        "absolute-path qualification was not persistent-binary scoped");
                System.out.println("resolved OpenBLAS path: " + path.toRealPath());
                System.out.println("target fingerprint: " + qualification.targetFingerprint());
                System.out.println("binary identity: "
                        + qualification.binaryIdentity().orElseThrow());
                System.out.println("qualified scope: " + qualification.scope());
                require(invocation.threadCount() == 1, "OpenBLAS thread count did not become one");
                for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                    checkFinite(type, coordinator, budget, 1, qualification);
                    checkSpecial(type, coordinator, budget, 1, qualification);
                }
                checkOverlappingCountOneCallsAndWriterExclusion(coordinator, invocation);
                coordinator.configure(2);
                require(invocation.threadCount() == 2, "OpenBLAS thread count did not become two");
                for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                    checkFinite(type, coordinator, budget, 2, qualification);
                }
        } finally {
            coordinator.close();
            System.out.println("CPU OpenBLAS native checkpoint passed; restored thread count "
                    + original);
        }
    }

    private static void checkOverlappingCountOneCallsAndWriterExclusion(
            CpuOpenBlasCoordinator coordinator, CpuOpenBlasInvocation invocation) throws Exception {
        try (Arena arena = Arena.ofShared()) {
            var barrier = new CyclicBarrier(2);
            var admitted = new CountDownLatch(2);
            var release = new CountDownLatch(1);
            var failure = new AtomicReference<Throwable>();
            Runnable call = () -> coordinator.execute(1, 1, () -> {
                try {
                    MemorySegment left = arena.allocate(Float.BYTES, Float.BYTES);
                    MemorySegment right = arena.allocate(Float.BYTES, Float.BYTES);
                    MemorySegment output = arena.allocate(Float.BYTES, Float.BYTES);
                    left.set(ValueLayout.JAVA_FLOAT, 0, 2.0f);
                    right.set(ValueLayout.JAVA_FLOAT, 0, 3.0f);
                    admitted.countDown();
                    barrier.await();
                    release.await();
                    invocation.sgemm(1, 1, 1, 1.0f, left, right, 0.0f, output);
                    require(output.get(ValueLayout.JAVA_FLOAT, 0) == 6.0f,
                            "overlapping count-one GEMM result disagrees");
                } catch (Throwable actual) { throw new RuntimeException(actual); }
            });
            Thread first = Thread.ofPlatform().start(() -> runChecked(call, failure));
            Thread second = Thread.ofPlatform().start(() -> runChecked(call, failure));
            require(admitted.await(5, java.util.concurrent.TimeUnit.SECONDS),
                    "compatible count-one calls did not overlap admission");
            Thread writer = Thread.ofPlatform().start(() -> {
                try { coordinator.configure(2); }
                catch (Throwable actual) { failure.compareAndSet(null, actual); }
            });
            long deadline = System.nanoTime() + 5_000_000_000L;
            while (writer.getState() != Thread.State.WAITING && System.nanoTime() < deadline) {
                Thread.onSpinWait();
            }
            require(writer.getState() == Thread.State.WAITING,
                    "OpenBLAS configuration writer did not wait for admitted calls");
            release.countDown();
            first.join(); second.join(); writer.join();
            if (failure.get() != null) throw new AssertionError("overlap checkpoint failed",
                    failure.get());
            require(invocation.threadCount() == 2,
                    "writer did not install count two after admitted calls quiesced");
            coordinator.configure(1);
        }
    }

    private static void runChecked(Runnable action, AtomicReference<Throwable> failure) {
        try { action.run(); }
        catch (Throwable actual) { failure.compareAndSet(null, actual); }
    }

    /**
     * Validates and preserves the checkpoint's exact absolute-path argument.
     *
     * @param argument the caller-supplied path text
     * @return the exact existing absolute-path selection; never {@code null}
     * @throws java.nio.file.InvalidPathException if {@code argument} has invalid path syntax
     * @throws IllegalArgumentException if the path is relative or not a regular file
     */
    private static CpuOpenBlasDiscoveryResult.Selection exactPathSelection(String argument) {
        Path path = Path.of(argument);
        if (!path.isAbsolute() || !Files.isRegularFile(path)) throw new IllegalArgumentException(
                "OpenBLAS library path must be an existing absolute file: " + path);
        return new CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath(path);
    }

    private static void checkFinite(DataType type, CpuOpenBlasCoordinator coordinator,
            CpuConcurrencyBudget budget, int threadCount,
            CpuOpenBlasQualification qualification) {
        double[] left = {1, -2, 3, 4, 5, -6};
        double[] right = {7, 8, -9, 10, 11, 12};
        boolean[][] cases = {{false, false, false}, {true, false, false},
                {false, true, false}, {true, true, false}, {false, false, true},
                {true, false, true}, {false, true, true}, {true, true, true}};
        for (boolean[] copy : cases) {
            double[] actual = execute(type, 2, 2, 3, left, right,
                    copy[0], copy[1], copy[2], coordinator, budget, threadCount, qualification);
            checkFiniteOracle(type, 2, 2, 3, left, right, actual);
        }
    }

    private static void checkSpecial(DataType type, CpuOpenBlasCoordinator coordinator,
            CpuConcurrencyBudget budget, int threadCount,
            CpuOpenBlasQualification qualification) {
        double nan = execute(type, 1, 1, 3, new double[] {Double.NaN, 0, 0},
                new double[] {1, 1, 1}, false, false, false, coordinator, budget, threadCount,
                qualification)[0];
        double positiveInfinity = execute(type, 1, 1, 2,
                new double[] {Double.POSITIVE_INFINITY, 1}, new double[] {1, 1}, false, false, false,
                coordinator, budget, threadCount, qualification)[0];
        double negativeInfinity = execute(type, 1, 1, 2,
                new double[] {Double.NEGATIVE_INFINITY, 1}, new double[] {1, 1}, false, false, false,
                coordinator, budget, threadCount, qualification)[0];
        double positiveZero = execute(type, 1, 1, 2, new double[] {0, 0},
                new double[] {1, 2}, false, false, false, coordinator, budget, threadCount,
                qualification)[0];
        require(Double.isNaN(nan), type + " did not preserve the one-NaN product class");
        require(positiveInfinity == Double.POSITIVE_INFINITY,
                type + " did not preserve sole positive infinity");
        require(negativeInfinity == Double.NEGATIVE_INFINITY,
                type + " did not preserve sole negative infinity");
        require(Double.doubleToRawLongBits(positiveZero) == 0L,
                type + " did not produce unambiguous positive zero");
    }

    private static double[] execute(DataType type, int m, int n, int k, double[] leftValues,
            double[] rightValues, boolean copyLeft, boolean copyRight, boolean copyOutput,
            CpuOpenBlasCoordinator coordinator, CpuConcurrencyBudget budget, int threadCount,
            CpuOpenBlasQualification qualification) {
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                new CpuPartitionPreparer().analyze(context(type, m, n, k, copyLeft, copyRight,
                        copyOutput, threadCount, qualification));
        require(analysis.plan().route() == CpuPartitionPreparationPlan.Route.OPENBLAS,
                "analysis did not select OpenBLAS");
        require(analysis.plan().openBlasPlan().orElseThrow().representation()
                        == representation(copyLeft, copyRight, copyOutput),
                "analysis selected the wrong representation");
        PreparedExecutable executable = finalize(analysis, coordinator, budget);
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
            MemorySegment right;
            if (copyRight) {
                double[] physical = rightAffinePhysical(k, n, rightValues);
                right = type == DataType.FLOAT32 ? MemorySegment.ofArray(toFloats(physical))
                        : MemorySegment.ofArray(physical);
            } else {
                right = arena.allocate(Math.multiplyExact((long) k * n, type.byteWidth()),
                        type.byteWidth());
                put(type, right, rightValues);
            }
            MemorySegment output = copyOutput
                    ? type == DataType.FLOAT32 ? MemorySegment.ofArray(new float[m * n])
                            : MemorySegment.ofArray(new double[m * n])
                    : arena.allocate(Math.multiplyExact((long) m * n, type.byteWidth()),
                            type.byteWidth());
            List<WorkspaceRepresentation> workspaces = executable.memoryPlan().workspaces().stream()
                    .map(entry -> (WorkspaceRepresentation) CpuContiguousWorkspace.allocate(
                            entry.byteSize(), entry.byteAlignment())).toList();
            try (RunState state = state(executable.memoryPlan(), type, left, right, output,
                    workspaces)) {
                executable.bind(state).execute();
            }
            double[] physical = get(type, output);
            return copyOutput ? stridedLogical(m, n, physical) : physical;
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,
            int m, int n, int k, boolean copyLeft, boolean copyRight, boolean copyOutput,
            int threadCount, CpuOpenBlasQualification qualification) {
        var node = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        Shape leftShape = Shape.of(m, k);
        TensorDescriptor left = new TensorDescriptor(type, leftShape, Optional.of(copyLeft
                ? LayoutDescriptor.of(leftShape, new long[] {1, m}, 0, true)
                : LayoutDescriptor.contiguous(leftShape)), false);
        Shape rightShape = Shape.of(k, n);
        Shape outputShape = Shape.of(m, n);
        TensorDescriptor right = new TensorDescriptor(type, rightShape, Optional.of(copyRight
                ? LayoutDescriptor.of(rightShape, new long[] {n + 1L, 1}, 0, true)
                : LayoutDescriptor.contiguous(rightShape)), false);
        TensorDescriptor output = new TensorDescriptor(type, outputShape, Optional.of(copyOutput
                ? LayoutDescriptor.of(outputShape, new long[] {1, m}, 0, true)
                : LayoutDescriptor.contiguous(outputShape)), false);
        List<TensorDescriptor> descriptors = List.of(left, right, output);
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
        CpuKernelSpecialization.CarrierAccess array = type == DataType.FLOAT32
                ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        var carriers = List.of(copyLeft ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                copyRight ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                copyOutput ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        var nativeFact = new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width);
        var storage = List.of(copyLeft ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact,
                copyRight ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact,
                copyOutput ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact);
        var policy = copyLeft || copyRight || copyOutput
                ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                1, 1, 10, 0, 2, 1_000_000, 0, 0)
                : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED;
        var inputs = new CpuPartitionAnalysisInputs(false, carriers,
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        2, 2, 1), policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, storage,
                new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(
                        Optional.of(qualification),
                        List.of(new CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate(
                                threadCount, CpuPartitionAnalysisInputs.CostTerms.complete(1, 1, 1))),
                        CpuPartitionAnalysisInputs.CostTerms.complete(1_000, 10, 100),
                        CpuPartitionAnalysisInputs.RepresentationCostTerms.ZERO,
                        java.util.OptionalLong.of(1), java.util.OptionalInt.of(1)));
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(), inputs);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static PreparedExecutable finalize(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuOpenBlasCoordinator coordinator, CpuConcurrencyBudget budget) {
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
        return new CpuPartitionFinalizer(Optional.empty(), Optional.empty(), budget,
                Optional.of(coordinator)).finalizePartition(
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

    private static double[] stridedLogical(int rows, int columns, double[] physical) {
        double[] logical = new double[physical.length];
        for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
            logical[row * columns + column] = physical[column * rows + row];
        }
        return logical;
    }

    private static double[] rightAffinePhysical(int rows, int columns, double[] logical) {
        int rowStride = columns + 1;
        double[] physical = new double[(rows - 1) * rowStride + columns];
        for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
            physical[row * rowStride + column] = logical[row * columns + column];
        }
        return physical;
    }

    private static CpuOpenBlasRoutePlan.Representation representation(boolean left,
            boolean right, boolean output) {
        if (left && right && output) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT_OUTPUT;
        if (left && right) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT;
        if (left && output) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_OUTPUT;
        if (right && output) return CpuOpenBlasRoutePlan.Representation.COPY_RIGHT_OUTPUT;
        if (left) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT;
        if (right) return CpuOpenBlasRoutePlan.Representation.COPY_RIGHT;
        if (output) return CpuOpenBlasRoutePlan.Representation.COPY_OUTPUT;
        return CpuOpenBlasRoutePlan.Representation.DIRECT;
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
