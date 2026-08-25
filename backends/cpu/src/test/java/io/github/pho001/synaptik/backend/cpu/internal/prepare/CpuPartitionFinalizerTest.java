package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import java.nio.file.Files;

public class CpuPartitionFinalizerTest {
    @TempDir Path root;

    @Test void maskedReductionFinalizesThreeBuffersExactWorkspaceAndOneSchema45Artifact()
            throws Exception {
        var base = CpuMaskedReductionLoweringTest.context(
                io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                DataType.FLOAT64, Shape.of(2, 4), Shape.of(4), 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.BYTE_ARRAY,
                                CarrierAccess.DOUBLE_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        Path artifactRoot = root.resolve("masked-reduction");
        var executable = finalizeExecutable(analysis, Optional.of(artifactRoot));
        try (var files = Files.list(artifactRoot)) {
            assertAll(() -> assertEquals(3, executable.bufferSelectionCount()),
                    () -> assertEquals(3, executable.memoryPlan().buffers().size()),
                    () -> assertEquals(1, executable.memoryPlan().workspaces().size()),
                    () -> assertEquals(45, io.github.pho001.synaptik.backend.cpu.internal.cache
                            .CpuGeneratorSchema.CURRENT_VERSION),
                    () -> assertEquals(1, files.filter(path -> path.getFileName().toString()
                            .endsWith(".artifact")).count()));
        }
    }

    @Test void finalizesZeroInputInitializerWithOneBufferAndNoWorkspace() {
        var analysis = new CpuPartitionPreparer().analyze(
                CpuRandomLoweringTest.initialContext(Long.MIN_VALUE, Long.MAX_VALUE));
        var executable = finalizeExecutable(analysis, Optional.of(root.resolve("random")));
        assertAll(() -> assertEquals(1, executable.bufferSelectionCount()),
                () -> assertEquals(1, executable.memoryPlan().buffers().size()),
                () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                () -> assertNotNull(executable.artifact().entryPoint()));
    }

    @Test void finalizesOneArtifactAndOnePartitionRecipeAfterExactAssignment() {
        var executable = finalizeExecutable(Shape.of(4), Optional.of(root));
        assertAll(
                () -> assertEquals(4, executable.bufferSelectionCount()),
                () -> assertEquals(4, executable.memoryPlan().buffers().size()),
                () -> assertEquals(4, executable.binding().end()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void parallelFinalizationRequiresAnOpenSufficientWorkerGroup() {
        var context = CpuPartitionPreparerTest.context(Shape.of(16));
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1));
        var analysis = new CpuPartitionPreparer().analyze(new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                context.partition(), context.nodes(), context.values(), context.memoryRequirements(),
                context.constants(), inputs));
        assertThrows(IllegalArgumentException.class,
                () -> finalizeExecutable(analysis, Optional.empty()));
        var undersized = new CpuWorkerGroup(1);
        try {
            assertThrows(IllegalArgumentException.class, () -> finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(undersized)));
        } finally { undersized.close(); }
        var closed = new CpuWorkerGroup(2);
        closed.close();
        assertThrows(IllegalArgumentException.class, () -> finalizeExecutable(analysis,
                Optional.empty(), Optional.of(closed)));
    }

    @Test void resolvesMaterializationWorkspaceAndWorkersBeforeArtifactPersistence() {
        var shape = Shape.of(2, 3);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, 48, 1, 1);
        var parallel = new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1);
        var analysis = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), parallel, policy));
        Path artifactRoot = root.resolve("must-remain-absent");
        assertAll(
                () -> assertEquals(5, analysis.requirements().size()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isPresent()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> finalizeExecutable(analysis, Optional.of(artifactRoot))),
                () -> assertFalse(Files.exists(artifactRoot)));
    }

    @Test void finalizesMovementAsOneArtifactWithoutWorkspace() {
        var analysis = new CpuPartitionPreparer().analyze(
                CpuNonAffineMovementLoweringTest.context(
                        new Operation(WindowTransformKind.UNFOLD_AXIS,
                                new UnfoldAxisAttrs(0, 2, 1)),
                        List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(3))),
                        CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(2, 2))));
        var executable = finalizeExecutable(analysis, Optional.of(root.resolve("movement")));
        assertAll(
                () -> assertEquals(1, analysis.plan().units().size()),
                () -> assertEquals(2, executable.bufferSelectionCount()),
                () -> assertEquals(2, executable.accessBindings().size()),
                () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void finalizesSliceUpdateAsOneArtifactAfterExactAssignments() {
        var analysis = new CpuPartitionPreparer().analyze(
                CpuNonAffineMovementLoweringTest.context(
                        new Operation(SliceKind.SLICE_UPDATE,
                                new SliceAttrs(List.of(3L), List.of(2L), List.of(0),
                                        List.of(-2L))),
                        List.of(0, 1),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(4)),
                                CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(4))));
        Path artifactRoot = root.resolve("slice-update");
        var executable = finalizeExecutable(analysis, Optional.of(artifactRoot));
        assertAll(
                () -> assertEquals(1, analysis.plan().units().size()),
                () -> assertEquals(3, executable.bufferSelectionCount()),
                () -> assertEquals(3, executable.accessBindings().size()),
                () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void indexingValidatesWorkersBeforeCreatingItsOnlyArtifact() throws Exception {
        var base = CpuIndexingLoweringTest.context(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4)), List.of(0),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(4))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(4, 4)));
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), inputs);
        var analysis = new CpuPartitionPreparer().analyze(context);
        Path artifactRoot = root.resolve("indexing");
        assertAll(
                () -> assertEquals(2, analysis.requirements().size()),
                () -> assertEquals(1, analysis.plan().units().size()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isEmpty()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> finalizeExecutable(analysis, Optional.of(artifactRoot))),
                () -> assertFalse(Files.exists(artifactRoot)));
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = finalizeExecutable(analysis, Optional.of(artifactRoot),
                    Optional.of(workers));
            try (var files = Files.list(artifactRoot)) {
                assertAll(
                        () -> assertNotNull(executable.artifact().hiddenClass()),
                        () -> assertEquals(2, executable.bufferSelectionCount()),
                        () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                        () -> assertEquals(1, files.filter(path -> path.getFileName().toString()
                                .endsWith(".artifact")).count()));
            }
        } finally { workers.close(); }
    }

    @Test void scatterProductFinalizesOneArtifactAndRejectsMalformedWorkspaceBeforeLookup()
            throws Exception {
        var base = CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MUL)), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(3)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(4))),
                CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(3)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, List.of(
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        Path validRoot = root.resolve("scatter-valid");
        var executable = finalizeExecutable(analysis, Optional.of(validRoot));
        try (var files = Files.list(validRoot)) {
            assertAll(
                    () -> assertEquals(1, executable.memoryPlan().workspaces().size()),
                    () -> assertEquals(analysis.plan().workspaceDeclaration().orElseThrow()
                                    .byteSize(),
                            executable.memoryPlan().workspaces().getFirst().byteSize()),
                    () -> assertEquals(1, files.filter(path -> path.getFileName().toString()
                            .endsWith(".artifact")).count()));
        }

        Path malformedRoot = root.resolve("scatter-malformed");
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () ->
                        finalizeWithMalformedWorkspace(analysis, malformedRoot)),
                () -> assertFalse(Files.exists(malformedRoot)));
    }

    @Test void foldFinalizesExactlyTwoAssignedBuffersAndOneSchema17Artifact() throws Exception {
        var base = CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT64, Shape.of(3, 3), Shape.of(5));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        Path artifactRoot = root.resolve("fold");
        var executable = finalizeExecutable(analysis, Optional.of(artifactRoot));
        try (var files = Files.list(artifactRoot)) {
            assertAll(() -> assertEquals(2, executable.bufferSelectionCount()),
                    () -> assertEquals(2, executable.memoryPlan().buffers().size()),
                    () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                    () -> assertEquals(45, io.github.pho001.synaptik.backend.cpu.internal.cache
                            .CpuGeneratorSchema.CURRENT_VERSION),
                    () -> assertEquals(1, files.filter(path -> path.getFileName().toString()
                            .endsWith(".artifact")).count()));
        }
    }

    @Test void topKFinalizesThreeBuffersExactScratchAndOneArtifact() throws Exception {
        var base = CpuOrderingLoweringTest.context(new Operation(
                io.github.pho001.synaptik.model.operation.ordering.TopKKind.TOP_K,
                new io.github.pho001.synaptik.model.operation.ordering.TopKAttrs(1, 2, true, false)),
                DataType.FLOAT32, Shape.of(3, 5), Shape.of(3, 2), true);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                                CarrierAccess.LONG_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var executable = finalizeExecutable(analysis, Optional.of(root.resolve("ordering")));
        assertAll(() -> assertEquals(3, executable.bufferSelectionCount()),
                () -> assertEquals(3, executable.memoryPlan().buffers().size()),
                () -> assertEquals(1, executable.memoryPlan().workspaces().size()),
                () -> assertEquals(80, executable.memoryPlan().workspaces().getFirst().byteSize()),
                () -> assertTrue(executable.artifact().specialization().scratchParameter()));
    }

    @Test void argExtremaFinalizesExactlyTwoBuffersNoWorkspaceAndOneCurrentArtifact()
            throws Exception {
        var base = CpuArgExtremaLoweringTest.context(
                io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.ARG_MIN,
                DataType.FLOAT64, Shape.of(2, 3), 1, false,
                io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy.FIRST_INDEX);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.LONG_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        Path artifactRoot = root.resolve("arg-extrema");
        var executable = finalizeExecutable(analysis, Optional.of(artifactRoot));
        try (var files = Files.list(artifactRoot)) {
            assertAll(() -> assertEquals(2, executable.bufferSelectionCount()),
                    () -> assertEquals(2, executable.memoryPlan().buffers().size()),
                    () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()),
                    () -> assertEquals(45, io.github.pho001.synaptik.backend.cpu.internal.cache
                            .CpuGeneratorSchema.CURRENT_VERSION),
                    () -> assertEquals(1, files.filter(path -> path.getFileName().toString()
                            .endsWith(".artifact")).count()));
        }
    }

    private static void finalizeWithMalformedWorkspace(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Path artifactRoot) {
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (int i = 0; i < analysis.requirements().size(); i++) {
            var requirement = analysis.requirements().get(i);
            if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                var slot = new WorkspaceSlot(workspaces.size());
                workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot,
                        Math.addExact(workspace.byteSize(), 8), workspace.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(workspace, slot,
                        workspaces.size() - 1));
            } else {
                var buffer = (PreparationResourceRequirement.Buffer) requirement;
                var slot = new BufferSlot(entries.size());
                entries.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(),
                        buffer.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot,
                        entries.size() - 1));
            }
        }
        new CpuPartitionFinalizer(Optional.of(artifactRoot), Optional.empty()).finalizePartition(
                new BackendPartitionFinalization<>(analysis,
                        new PreparedMemoryPlan(entries, workspaces), assignments));
    }

    public static CpuPreparedExecutable finalizeExecutable(Shape shape, Optional<Path> root) {
        return finalizeExecutable(CpuPartitionPreparerTest.analyze(shape), root);
    }

    public static CpuPreparedExecutable finalizeExecutable(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Optional<Path> root) {
        return finalizeExecutable(analysis, root, Optional.empty());
    }

    public static CpuPreparedExecutable finalizeExecutable(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    CpuPartitionPreparationPlan> analysis, Optional<Path> root,
            Optional<CpuWorkerGroup> workerGroup) {
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        var workspaceEntries = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        for (int i = 0; i < analysis.requirements().size(); i++) {
            var any = analysis.requirements().get(i);
            if (any instanceof PreparationResourceRequirement.Workspace requirement) {
                var slot = new WorkspaceSlot(workspaceEntries.size());
                workspaceEntries.add(new PreparedMemoryPlan.WorkspaceEntry(slot,
                        requirement.byteSize(), requirement.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(requirement, slot,
                        workspaceEntries.size() - 1));
                continue;
            }
            var requirement = (PreparationResourceRequirement.Buffer) any;
            var slot = new BufferSlot(i);
            entries.add(new PreparedMemoryPlan.BufferEntry(slot, requirement.byteSize(),
                    requirement.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(requirement, slot, i));
        }
        var memoryPlan = new PreparedMemoryPlan(entries, workspaceEntries);
        return (CpuPreparedExecutable) new CpuPartitionFinalizer(root, workerGroup).finalizePartition(
                new BackendPartitionFinalization<>(analysis, memoryPlan, assignments));
    }
}
