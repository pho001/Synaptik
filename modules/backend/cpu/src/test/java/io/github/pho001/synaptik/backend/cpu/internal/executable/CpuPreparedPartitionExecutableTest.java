package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.MemorySegment;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuPreparedPartitionExecutableTest {
    @Test void nonfixedContainerSelectionsMustBelongToAnActualDynamicChild() {
        var fixed = io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
                .finalizeExecutable(Shape.of(4), Optional.empty());
        for (boolean extraWorkspace : List.of(false, true)) {
            var buffers = new ArrayList<>(fixed.memoryPlan().buffers());
            var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
            var batchBytes = new io.github.pho001.synaptik.backend.cpu.internal.memory
                    .CpuRepresentationRecipes.BatchByteSize(0, 0, Double.BYTES);
            if (extraWorkspace) workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(
                    new WorkspaceSlot(0), batchBytes, Double.BYTES));
            else buffers.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(buffers.size()),
                    batchBytes, Double.BYTES));
            var plan = new PreparedMemoryPlan(buffers, workspaces, true);
            var childSelections = java.util.stream.IntStream.range(0, fixed.bufferSelectionCount())
                    .mapToObj(fixed::bufferSelection).toList();
            var child = new CpuPreparedExecutable(plan, childSelections, fixed.artifact(),
                    fixed.accessBindings(), fixed.artifact().specialization().carrierPattern(), 0, 4);
            var selections = new ArrayList<>(childSelections);
            var accesses = new ArrayList<PreparedExecutable.BufferAccess>();
            for (int p = 0; p < childSelections.size(); p++) accesses.add(child.bufferAccess(p));
            if (!extraWorkspace) {
                selections.add(new PreparedExecutable.BufferSelection(buffers.size() - 1, 0));
                accesses.add(PreparedExecutable.BufferAccess.READ_ONLY);
            }
            var selectedWorkspaces = extraWorkspace
                    ? List.of(new PreparedExecutable.WorkspaceSelection(0))
                    : List.<PreparedExecutable.WorkspaceSelection>of();
            var executable = new CpuPreparedPartitionExecutable(plan, selections, selectedWorkspaces,
                    accesses, List.of(child), List.of(List.of()));
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            for (int p = 0; p < buffers.size(); p++) {
                var carrier = new double[p < childSelections.size() ? 4 : 0];
                var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64,
                        carrier.length, MemorySegment.ofArray(carrier)));
                bindings.add(List.of(new BufferRepresentationBinding(borrowed, RunResourceOwnership.BORROWED)));
            }
            List<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation> scratch = extraWorkspace
                    ? List.of(io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace.allocate(0, 8))
                    : List.of();
            try (var run = new RunState(plan, bindings, scratch,
                    RunMemoryGeometry.resolve(plan, OptionalLong.of(0)))) {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(run));
            }
        }
    }

    @Test void fixedCompositeKeepsBothChildrenFixedAcrossCompleteBatchRuns() {
        var executable = finalizeSequence(new CpuPartitionPreparer().analyze(context()), true);
        var first = assertInstanceOf(CpuPreparedExecutable.class, executable.children().get(0));
        var second = assertInstanceOf(CpuPreparedExecutable.class, executable.children().get(1));
        var firstArtifact = first.artifact();
        var secondArtifact = second.artifact();
        for (long batch : new long[] {0, 13, 0, 1}) {
            float[] intermediate = new float[4], output = new float[4];
            try (var run = state(executable, List.of(new float[] {1, -2, 3, 4},
                    new float[] {2}, new float[] {.5f}, intermediate,
                    new float[] {-3, 2, -10, 1}, output), OptionalLong.of(batch))) {
                var bound = executable.bind(run);
                bound.execute();
                bound.execute();
                assertArrayEquals(new float[] {2.5f, -3.5f, 6.5f, 8.5f}, intermediate);
                assertArrayEquals(new float[] {0, 0, 0, 9.5f}, output);
                assertSame(firstArtifact, first.artifact());
                assertSame(secondArtifact, second.artifact());
            }
        }
        float[] shared = {1, -2, 3, 4};
        try (var run = state(executable, List.of(shared, new float[] {2}, new float[] {.5f},
                new float[4], new float[] {-3, 2, -10, 1}, shared), OptionalLong.of(0))) {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(run));
            assertArrayEquals(new float[] {1, -2, 3, 4}, shared);
        }
        var staticExecutable = finalizeSequence(new CpuPartitionPreparer().analyze(context()));
        try (var foreign = state(staticExecutable, List.of(new float[4], new float[1], new float[1],
                new float[4], new float[4], new float[4]))) {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(foreign));
        }
    }

    @Test void realizesTwoArtifactsAndExecutesStrictlyThroughOneComposite() {
        var analysis = new CpuPartitionPreparer().analyze(context());
        assertAll(() -> assertEquals(CpuPartitionPreparationPlan.PlanForm
                        .GENERAL_PARTITION, analysis.plan().form()),
                () -> assertEquals(2, analysis.plan().units().size()),
                () -> assertEquals(6, analysis.plan().bufferDeclarations().size()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isEmpty()),
                () -> assertEquals(analysis.plan().units().get(0).boundaryValues().getLast(),
                        analysis.plan().units().get(1).boundaryValues().getFirst()));
        var executable = finalizeSequence(analysis);
        assertAll(() -> assertNotSame(assertInstanceOf(io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable.class, executable.children().get(0)).artifact().hiddenClass(),
                        assertInstanceOf(io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable.class, executable.children().get(1)).artifact().hiddenClass()),
                () -> assertEquals(6, executable.bufferSelectionCount()),
                () -> assertEquals(PreparedExecutable.BufferAccess.WRITE_ONLY,
                        executable.bufferAccess(3)),
                () -> assertEquals(PreparedExecutable.BufferAccess.WRITE_ONLY,
                        executable.bufferAccess(5)));

        float[] input = {1, -2, 3, 4};
        float[] weight = {2};
        float[] intrinsic = {.5f};
        float[] intermediate = new float[4];
        float[] external = {-3, 2, -10, 1};
        float[] output = new float[4];
        var run = state(executable, List.of(input, weight, intrinsic, intermediate, external, output));
        try {
            executable.bind(run).execute();
            assertArrayEquals(new float[] {2.5f, -3.5f, 6.5f, 8.5f}, intermediate);
            assertArrayEquals(new float[] {0, 0, 0, 9.5f}, output);
        } finally { run.close(); }
    }

    @Test void defaultSegmentChildrenBindHeapInputsAcrossComposedUnits() {
        var original = context();
        var defaultContext = new PrepareContext<>(original.partition(), original.nodes(),
                original.values(), original.memoryRequirements(), original.constants(),
                CpuPartitionAnalysisInputs.DEFAULT);
        var executable = finalizeSequence(new CpuPartitionPreparer().analyze(defaultContext));
        float[] intermediate = new float[4];
        float[] output = new float[4];
        var run = state(executable, List.of(new float[] {1, -2, 3, 4},
                new float[] {2}, new float[] {.5f}, intermediate,
                new float[] {-3, 2, -10, 1}, output));
        try {
            executable.bind(run).execute();
            assertArrayEquals(new float[] {2.5f, -3.5f, 6.5f, 8.5f}, intermediate);
            assertArrayEquals(new float[] {0, 0, 0, 9.5f}, output);
        } finally { run.close(); }
    }

    @Test void rejectsCrossUnitAliasBeforeTheConv2dWrite() {
        var executable = finalizeSequence(new CpuPartitionPreparer().analyze(context()));
        float[] shared = {1, -2, 3, 4};
        float[] unchanged = shared.clone();
        var run = state(executable, List.of(shared, new float[] {2}, new float[] {.5f},
                new float[4], new float[] {-3, 2, -10, 1}, shared));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(run));
            assertArrayEquals(unchanged, shared);
        } finally { run.close(); }
    }

    @Test void acceptsEitherAddOperandOrderAndRetainsEveryPublishedSuffixResult() {
        var analysis = new CpuPartitionPreparer().analyze(context(true, true));
        var plan = analysis.plan();
        assertAll(() -> assertEquals(CpuPartitionPreparationPlan.PlanForm
                        .GENERAL_PARTITION, plan.form()),
                () -> assertEquals(3, plan.units().size()),
                () -> assertEquals(1, plan.units().get(1).outputCount()),
                () -> assertEquals(7, plan.bufferDeclarations().size()),
                () -> assertEquals(plan.units().get(0).boundaryValues().getLast(),
                        plan.units().get(1).boundaryValues().get(1)));
        var executable = finalizeSequence(analysis);
        assertAll(() -> assertEquals(PreparedExecutable.BufferAccess.WRITE_ONLY,
                        executable.bufferAccess(3)),
                () -> assertEquals(PreparedExecutable.BufferAccess.WRITE_ONLY,
                        executable.bufferAccess(5)),
                () -> assertEquals(PreparedExecutable.BufferAccess.WRITE_ONLY,
                        executable.bufferAccess(6)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context() {
        return context(false, false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(
            boolean externalFirst, boolean publishAdd) {
        Shape shape = Shape.of(1, 1, 2, 2), weightShape = Shape.of(1, 1, 1, 1);
        TensorDescriptor tensor = descriptor(shape), weight = descriptor(weightShape);
        TensorDescriptor bias = descriptor(Shape.of(1));
        List<ValueId> ids = java.util.stream.LongStream.range(0, 7)
                .mapToObj(ValueId::new).toList();
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(Conv2dKind.CONV2D,
                        Conv2dAttrs.defaults()), ids.subList(0, 3), List.of(ids.get(4))),
                new CompiledNode(new NodeId(1), new Operation(BinaryArithmeticKind.ADD,
                        NoOperationAttrs.INSTANCE), externalFirst
                            ? List.of(ids.get(3), ids.get(4)) : List.of(ids.get(4), ids.get(3)),
                        List.of(ids.get(5))),
                new CompiledNode(new NodeId(2), new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE), List.of(ids.get(5)), List.of(ids.get(6))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var descriptors = List.of(tensor, weight, bias, tensor, tensor, tensor, tensor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < ids.size(); i++) {
            values.add(new GraphValue(ids.get(i), descriptors.get(i)));
            boolean produced = i >= 4;
            boolean published = i == 4 || i == 6 || publishAdd && i == 5;
            memory.add(new LogicalMemoryRequirement(ids.get(i), descriptors.get(i),
                    produced ? Optional.of(partition) : Optional.empty(),
                    published ? List.of() : List.of(partition), published));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), new CpuPartitionAnalysisInputs(false,
                Collections.nCopies(publishAdd ? 7 : 6, CarrierAccess.FLOAT_ARRAY)));
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static CpuPreparedPartitionExecutable finalizeSequence(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
        return finalizeSequence(analysis, false);
    }

    private static CpuPreparedPartitionExecutable finalizeSequence(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis, boolean requiresBatch) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (var requirement : analysis.requirements()) {
            var buffer = (PreparationResourceRequirement.Buffer) requirement;
            var slot = new BufferSlot(buffers.size());
            buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                    buffer.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot, buffers.size() - 1, 0));
        }
        var memory = new PreparedMemoryPlan(buffers, List.of(), requiresBatch);
        return (CpuPreparedPartitionExecutable) new CpuPartitionFinalizer().finalizePartition(
                new BackendPartitionFinalization<>(analysis, memory, assignments)).executable();
    }

    private static RunState state(CpuPreparedPartitionExecutable executable,
            List<float[]> carriers) {
        return state(executable, carriers, OptionalLong.empty());
    }

    private static RunState state(CpuPreparedPartitionExecutable executable,
            List<float[]> carriers, OptionalLong batch) {
        var bindings = new ArrayList<List<BufferRepresentationBinding>>();
        for (int i = 0; i < carriers.size(); i++) {
            float[] carrier = carriers.get(i);
            var storage = new MemorySegmentStorage(DataType.FLOAT32, carrier.length,
                    MemorySegment.ofArray(carrier));
            bindings.add(List.of(new BufferRepresentationBinding(CpuBorrowedBuffer.borrow(storage),
                    RunResourceOwnership.BORROWED)));
        }
        return new RunState(executable.memoryPlan(), bindings, List.of(),
                RunMemoryGeometry.resolve(executable.memoryPlan(), batch));
    }
}
