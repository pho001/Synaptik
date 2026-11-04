package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAdvancedReductionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable whole-partition recipe with exact cold geometry and one direct generated artifact.
 * A parallel recipe borrows, but never closes, its explicitly supplied {@link CpuWorkerGroup};
 * cold binding creates deterministic disjoint chunk calls, while a scalar or one-chunk range
 * executes directly on the invoking thread. When analysis selected materialization, cold binding
 * retains the original read-only source for one canonical copy and substitutes the workspace
 * segment only in the generated consumer arguments. Execution completes that copy on the invoking
 * thread before any inline or worker consumer call begins.
 * A representation-adjusted child may instead retain one or two explicitly chosen copy plans and
 * their workspace positions. The enclosing partition composite performs each generated copy once
 * before invoking any child, while this child binds every compatible repeated or cross-unit
 * consumer directly to the completed workspace. Ordinary preparation selects direct children and
 * never asks Runtime to choose among retained representation candidates.
 * A prepared segment carrier accepts the exact selected heap segment as well as native storage;
 * explicit array-carrier specializations retain their direct array arguments. The choice is
 * resolved during cold binding and never changes the generated entry or its hot loop. Cold overlap
 * checks compare exact selected storage even when one boundary uses a segment and another uses
 * its observable primitive-array carrier.
 * For an affine plan, cold binding additionally validates the exact two boundary carriers and
 * represented address spans, rejects source/result overlap, and retains the composed address
 * pairs directly. Runtime invokes only the prepared range and never interprets the view chain.
 * For an indexing plan, cold binding creates one direct typed scalar validator. Each execution
 * scans the complete logical index domain in deterministic row-major order before any generated
 * call or worker submission. A zero output still validates a non-empty index domain, while a
 * valid zero-output execution makes no generated call and submits no worker work.
 * For scatter, each invocation validates every index and then replacement uniqueness before any
 * output call. Successful ranges own disjoint output coordinates and, when floating multiplication is
 * selected, disjoint slices of the exact declared scratch workspace.
 * For fold, binding rejects every physical input/output overlap before any generated call or
 * worker submission. Each scalar or parallel-scalar range owns disjoint output coordinates and
 * uses only its invocation-private packed coordinate state.
 * For explicit-state random execution, binding validates all six dropout input/output and all
 * three output/output span pairs before creating calls. One generated {@code [0,0)} prologue
 * writes initializer or next-state words exactly once before any dropout element range.
 * For cumulative scans, binding rejects complete input/output physical overlap before creating
 * calls or submitting workers. Each selected range owns complete independent logical slices;
 * one slice is never divided, and every range receives invocation-private coordinate state.
 * For ordinary aggregate execution, binding validates complete input/output spans and any
 * exact-state workspace size, alignment, accessibility, and physical
 * non-overlap before mutation. Floating numerical parallel ranges receive disjoint workspace
 * slices. All ranges own disjoint complete output cells; no range splits, partially reduces, or
 * combines a selected domain.
 * Ordinary fused SUM/MEAN uses one such range and binds every producer/tail read plus the sole
 * final write explicitly. Alias checks cover all complete physical spans before writes; only
 * proved private intermediate carriers are absent. Exact-state resources remain run-owned.
 * Fixed softmax derivative units also use the aggregate state header, but their final binding
 * spans the full gradient, not the reduced SUM. Cold binding checks every actual element base
 * plus the baked referenced span in bytes against the int-address proof before building the
 * invocation-private geometry. One range reuses one exact-state slice across rows; no forward
 * softmax finite-input validator is attached to these ordinary derivative primitives.
 * Masked SUM/MEAN invocation additionally validates the complete canonical Boolean mask before any
 * output write or worker submission. Binding permits only physical input/input aliasing and rejects the
 * output against both inputs and the exact-state workspace. Every call receives one already-sliced
 * private exact state region and retains its selected count in primitive invocation-local state.
 * Trailing Layer/RMS normalization binding validates every typed carrier, resolved span,
 * output/input overlap, and optional Layer exact-state slice before mutation or submission.
 * Ranges always own complete trailing slices; Layer ranges receive disjoint scratch, while RMS
 * requires none.
 * For MATMUL, binding validates exact typed carriers and packed normalized geometry, rejects every
 * output/input or output/bias overlap before mutation or worker submission, and creates disjoint
 * half-open work-unit ranges. Depending on the cold-selected realization, one work unit owns one
 * output cell, one complete M row, or one bounded M/N tile; K is never divided among ranges.
 * Pool2d binding validates exact typed input/output spans and rejects physical overlap before any
 * output mutation or worker submission. Each range owns complete NCHW output cells and receives
 * immutable packed layout and window geometry; a pooling window is never divided among ranges.
 * Direct loss binding validates every represented carrier, full logical geometry, injective
 * output, and complete output/input span before binding an index-target scan. At invocation the
 * scan compares ignore before bounds, before any generated call or worker submission;
 * MSE and dense categorical loss require no target-value scan. Losses declare no workspace.
 * Scaled-dot-product-attention binding validates all input/output/workspace overlap and prebinds
 * its optional canonical Boolean mask scan. Each range owns complete broadcast-batch/query
 * rows and a disjoint aligned score slice; zero-score and zero-work forms retain the scratch-shaped
 * entry without requiring a workspace.
 * All value-dependent scans (softmax inputs/shifts, loss targets, and canonical BOOL inputs or
 * masks) are cold-bound without reading storage. Each invocation scans current producer results
 * once on the orchestrator before its prologue, consumer writes, or worker submission. Traversal
 * state and typed readers are prebound, so successful scans allocate no hot-path state. Metadata,
 * carrier, span, accessibility, and overlap checks remain in cold binding.
 * An ordinary row-map/multiple-terminal region owns one unsplittable full-body work item:
 * [0,1) executes every row and publishes shared zeros even at B=0; [0,0) is empty. Other
 * ranges reject before writes. Binding checks all output spans/aliases, not only the last
 * output, and resolves invocation bases/strides once without changing prepared requirements.
 */
public final class CpuPreparedExecutable extends PreparedExecutable {
    private final io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan batchRoute;
    private final CpuGeneratedKernel artifact;
    private final Optional<CpuGeneratedKernel.PartialReductionArtifact> partialReductionArtifact;
    private final List<CpuAccessPlan.Binding> bindings;
    private final List<CarrierAccess> carrierPattern;
    private final List<CarrierAccess> generatedCarrierPattern;
    private final long start;
    private final long end;
    private final int selectedRangeCount;
    private final long minimumElementsPerWorker;
    private final CpuWorkerGroup workerGroup;
    private final CpuConcurrencyBudget concurrencyBudget;
    private final Optional<CpuMaterializationPlan> materialization;
    private final Optional<WorkspaceSelection> workspaceSelection;
    private final List<CpuMaterializationPlan> representationMaterializations;
    private final List<Integer> representationMaterializationPositions;
    private final List<WorkspaceSelection> representationWorkspaceSelections;
    private final boolean affineCopy;
    private final long[] affineAddressPairs;
    private final Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry;
    private final Optional<CpuIndexingLowering.Geometry> indexingGeometry;
    private final Optional<CpuScatterLowering.Geometry> scatterGeometry;
    private final Optional<CpuFoldLowering.Geometry> foldGeometry;
    private final Optional<CpuOrderingLowering.Geometry> orderingGeometry;
    private final Optional<CpuRandomLowering.Geometry> randomGeometry;
    private final Optional<CpuScanLowering.Geometry> scanGeometry;
    private final Optional<CpuAggregateLowering.Geometry> aggregateGeometry;
    private final Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry;
    private final Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry;
    private final Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry;
    private final Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry;
    private final Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry;
    private final Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry;
    private final Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry;
    private final Optional<CpuConv2dLowering.Geometry> conv2dGeometry;
    private final Optional<CpuConv3dLowering.Geometry> conv3dGeometry;
    private final Optional<CpuMatmulLowering.Geometry> matmulGeometry;
    private final Optional<CpuPool2dLowering.Geometry> pool2dGeometry;
    private final Optional<CpuPool3dLowering.Geometry> pool3dGeometry;
    private final Optional<CpuAttentionLowering.Geometry> attentionGeometry;
    private final Optional<CpuLossLowering.Geometry> lossGeometry;
    private final int outputCount;

    /**
     * Retains one fully prepared ordinary-batch scalar unit without resolving any actual B.
     * Every mutable access/packed array is created by cold binding, never retained here.
     * Boundary dtypes determine element bases, spans and alignment independently, including
     * low byte/short storage and separate eight-byte length CAST outputs. Generic family geometry
     * reuses the ordinary ABI and snapshots invocation-specific cursor seeds for repeated calls.
     * @param memoryPlan non-null exact batch-required prepared plan
     * @param selections non-null ordered unique-input-then-output selections
     * @param artifact non-null verified generated artifact matching route
     * @param route non-null immutable numeric batch route
     * @param workspace non-null optional assigned exact-state scratch selection
     * @param budget borrowed CPU admission budget, or null for low-level compatibility tests
     * @throws NullPointerException if a required argument is null
     * @throws IllegalArgumentException if mode, artifact, selections or scratch declarations disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            Optional<WorkspaceSelection> workspace, CpuConcurrencyBudget budget) {
        this(memoryPlan,selections,artifact,route,workspace,budget,1,1,null);
    }

    /**
     * Binds ordinary symbolic selected-row policy without resolving B or selecting another
     * artifact. Every run later owns its numeric ranges, staged values, calls and partials.
     * @param memoryPlan exact batch-required memory plan
     * @param selections ordered unique reads followed by all retained writes
     * @param artifact verified matching generated artifact
     * @param route immutable symbolic access recipes
     * @param workspace exact optional assigned scratch selection
     * @param budget borrowed inline budget, or coordinated group budget; null in compatibility mode
     * @param selectedRangeCount positive prepared maximum number of row ranges
     * @param minimumElementsPerWorker positive minimum row work, fixed across runs
     * @param workerGroup borrowed group exactly when selectedRangeCount exceeds one
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if artifact, policy, resources or borrowed budgets disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            Optional<WorkspaceSelection> workspace, CpuConcurrencyBudget budget,
            int selectedRangeCount,long minimumElementsPerWorker,CpuWorkerGroup workerGroup) {
        super(memoryPlan, selections, workspace.stream().toList(), accesses(selections.size(),
                route.specialization().rowMapReductionIr().map(r -> r.arithmetic().stores().size()).orElse(1)));
        this.batchRoute = Objects.requireNonNull(route, "route");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
        boolean selectedRow = route.specialization().rowFusionEmission().isPresent();
        if (!memoryPlan.requiresBatchSize()
                || !route.specialization().ordinaryBatch() && route.ordinaryBatchGeometry().isEmpty()
                || !artifact.specialization().equals(route.specialization())
                || selections.size() != ordinaryBatchBindings(route).size()
                || workspace.isPresent() != (route.ordinaryBatchGeometry().isPresent()
                        ? route.specialization().scratchParameter()
                        : route.batchAggregate().isPresent() || route.specialization().rowFusionEmission().isPresent())
                || selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                || (selectedRangeCount > 1) != (workerGroup != null)
                || !selectedRow && (selectedRangeCount != 1 || minimumElementsPerWorker != 1 || workerGroup != null
                    || !route.specialization().executionStrategy().equals(io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR))
                || selectedRow && (selectedRangeCount != route.specialization().rowFusionEmission().orElseThrow().maximumPartialCount()
                    || (selectedRangeCount > 1) != (route.specialization().executionStrategy().orchestration()
                        == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL))
                || workerGroup != null && budget != null && workerGroup.concurrencyBudget() != budget) {
            throw new IllegalArgumentException("ordinary batch executable declarations disagree");
        }
        partialReductionArtifact = Optional.empty(); bindings = List.of();
        carrierPattern = route.specialization().carrierPattern(); generatedCarrierPattern = carrierPattern;
        start = 0; end = 0; this.selectedRangeCount = selectedRangeCount; this.minimumElementsPerWorker = minimumElementsPerWorker;
        this.workerGroup = workerGroup; concurrencyBudget = workerGroup == null ? budget : null; materialization = Optional.empty();
        workspaceSelection = workspace; representationMaterializations = List.of();
        representationMaterializationPositions = List.of(); representationWorkspaceSelections = List.of();
        affineCopy = false; affineAddressPairs = new long[0]; movementGeometry = Optional.empty();
        indexingGeometry = Optional.empty(); scatterGeometry = Optional.empty(); foldGeometry = Optional.empty();
        orderingGeometry = Optional.empty(); randomGeometry = Optional.empty(); scanGeometry = Optional.empty();
        aggregateGeometry = Optional.empty(); argExtremaGeometry = Optional.empty(); maskedReductionGeometry = Optional.empty();
        advancedReductionGeometry = Optional.empty(); softmaxGeometry = Optional.empty(); trailingNormalizationGeometry = Optional.empty();
        batchNormInferenceGeometry = Optional.empty(); batchNormTrainingGeometry = Optional.empty();
        conv2dGeometry = Optional.empty(); conv3dGeometry = Optional.empty(); matmulGeometry = Optional.empty();
        pool2dGeometry = Optional.empty(); pool3dGeometry = Optional.empty(); attentionGeometry = Optional.empty();
        lossGeometry = Optional.empty(); outputCount = route.specialization().rowMapReductionIr()
                .map(r -> r.arithmetic().stores().size()).orElse(1);
    }

    /**
     * Selects the complete boundary recipes from their owning ordinary geometry representation.
     * Generic family geometry owns its bindings; retained routes keep their original legacy list.
     * @param route non-null prepared ordinary numeric route
     * @return non-null immutable ordered boundary recipes without copying or resolving B
     */
    private static List<CpuAccessPlan.BatchBinding> ordinaryBatchBindings(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        return route.ordinaryBatchGeometry().isPresent()
                ? route.ordinaryBatchGeometry().orElseThrow().bindings() : route.batchBindings();
    }

    /**
     * Creates a direct derived-boundary recipe for one exact half-open logical range.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections; copied by the superclass
     * @param artifact non-null verified artifact matching {@code carrierPattern}
     * @param bindings non-null full-range normalized boundary bindings; copied defensively
     * @param carrierPattern non-null ordered direct carrier pattern; copied defensively
     * @param generatedCarrierPattern non-null generated-entry carrier pattern; copied defensively
     * @param start non-negative inclusive logical element or slice bound
     * @param end exclusive logical element or slice bound no greater than the iteration count
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed open worker group for a parallel plan, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned workspace selection
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional static movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional functional-scatter geometry
     * @param foldGeometry non-null optional overlap-fold geometry
     * @param orderingGeometry non-null optional stable-ordering geometry
     * @param randomGeometry non-null optional explicit-state initializer/dropout geometry
     * @param scanGeometry non-null optional cumulative-scan geometry
     * @param aggregateGeometry non-null optional ordinary aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional logarithmic, statistical, or norm
     *     geometry
     * @param softmaxGeometry non-null optional stable-softmax geometry
     * @param trailingNormalizationGeometry non-null optional Layer/RMS geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if counts, specialization, or range disagree
     * @throws ArithmeticException if exact range or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry,
                advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry,
                batchNormInferenceGeometry, batchNormTrainingGeometry, Optional.empty());
    }

    /**
     * Creates an established-family recipe through optional ordering geometry.
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered boundary selections
     * @param artifact verified generated artifact retained strongly by this recipe
     * @param bindings complete boundary geometry; copied defensively
     * @param carrierPattern Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern generated-entry carrier forms; copied defensively
     * @param start inclusive logical range bound
     * @param end exclusive logical range bound
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum work per submitted range
     * @param workerGroup borrowed worker group for parallel execution, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned route workspace
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if memory, boundary, carrier, range, worker, route,
     *     workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    /**
     * Creates a direct scalar recipe with no optional family geometry.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections; copied by the superclass
     * @param artifact non-null verified artifact matching {@code carrierPattern}
     * @param bindings non-null full-range normalized boundary bindings; copied defensively
     * @param carrierPattern non-null ordered direct carrier pattern; copied defensively
     * @param start non-negative inclusive logical element bound
     * @param end exclusive logical element bound no greater than the iteration element count
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if memory, boundary, carrier, or range facts disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, long start, long end) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, start, end, 1, 1, null);
    }

    /**
     * Creates a direct recipe with cold-selected range geometry and an optional borrowed worker group.
     * The worker group must be present exactly when two or more ranges were selected. This object
     * retains the group without taking ownership of its lifecycle.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections; copied by the superclass
     * @param artifact non-null generated scalar or vector artifact matching
     *     {@code carrierPattern}
     * @param bindings non-null full-range normalized boundary bindings; copied defensively
     * @param carrierPattern non-null ordered direct carrier pattern; copied defensively
     * @param start non-negative inclusive logical element bound
     * @param end exclusive logical element bound no greater than the iteration element count
     * @param selectedRangeCount positive maximum chunk count selected during analysis
     * @param minimumElementsPerWorker positive minimum logical elements per submitted chunk
     * @param workerGroup borrowed open group for a parallel recipe, or {@code null} for a
     *     single-thread recipe; never closed by this executable
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if counts, specialization, range, or worker presence are
     *     inconsistent
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, long start, long end, int selectedRangeCount,
            long minimumElementsPerWorker, CpuWorkerGroup workerGroup) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, start, end,
                selectedRangeCount, minimumElementsPerWorker, workerGroup,
                Optional.empty(), Optional.empty());
    }

    /**
     * Creates a materialization-aware recipe whose original and generated carrier patterns are
     * the same.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections
     * @param artifact non-null verified generated artifact
     * @param bindings non-null generated-consumer access bindings; copied defensively
     * @param carrierPattern non-null direct and generated carrier pattern; copied defensively
     * @param start non-negative inclusive logical bound
     * @param end exclusive logical bound no greater than the element count
     * @param selectedRangeCount positive maximum selected chunk count
     * @param minimumElementsPerWorker positive minimum elements per submitted chunk
     * @param workerGroup borrowed worker group for a parallel recipe, otherwise {@code null}
     * @param materialization non-null optional selected one-input copy
     * @param workspaceSelection non-null optional assigned workspace position, present exactly
     *     when materialization is present
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if range, carrier, worker, materialization, workspace, or
     *     specialization facts disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, long start, long end, int selectedRangeCount,
            long minimumElementsPerWorker, CpuWorkerGroup workerGroup,
            Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, carrierPattern, start, end,
                selectedRangeCount, minimumElementsPerWorker, workerGroup, materialization,
                workspaceSelection, null);
    }

    /**
     * Creates the complete recipe with separate original and adjusted generated carrier patterns.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections
     * @param artifact non-null verified generated artifact matching
     *     {@code generatedCarrierPattern}
     * @param bindings non-null generated-consumer access bindings; copied defensively
     * @param carrierPattern non-null original source/output carrier pattern; copied defensively
     * @param generatedCarrierPattern non-null adjusted generated entry pattern; copied defensively
     * @param start non-negative inclusive logical bound
     * @param end exclusive logical bound no greater than the element count
     * @param selectedRangeCount positive maximum selected chunk count
     * @param minimumElementsPerWorker positive minimum elements per submitted chunk
     * @param workerGroup borrowed worker group for a parallel recipe, otherwise {@code null}
     * @param materialization non-null optional selected copy whose consumer binding appears in
     *     {@code bindings}
     * @param workspaceSelection non-null optional assigned workspace position, present exactly
     *     when materialization is present
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selection counts, range, carrier patterns, worker,
     *     materialization, workspace, or specialization facts disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount,
            long minimumElementsPerWorker, CpuWorkerGroup workerGroup,
            Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, null);
    }

    /**
     * Creates the complete recipe, optionally with cold-composed affine address pairs.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered boundary selections; copied by the superclass
     * @param artifact non-null generated artifact matching {@code generatedCarrierPattern}
     * @param bindings non-null full-range boundary bindings; copied defensively
     * @param carrierPattern non-null ordered Runtime carrier pattern; copied defensively
     * @param generatedCarrierPattern non-null ordered generated-entry carrier pattern; copied
     *     defensively
     * @param start non-negative inclusive logical or distinct-address bound
     * @param end exclusive bound no greater than the prepared copy or computation count
     * @param selectedRangeCount positive maximum chunk count selected during analysis
     * @param minimumElementsPerWorker positive minimum elements per submitted worker chunk
     * @param workerGroup borrowed open group for a parallel recipe, or {@code null} for a
     *     single-thread recipe; never closed by this executable
     * @param materialization non-null optional pointwise contiguous-input copy plan
     * @param workspaceSelection non-null optional workspace selection, present exactly with
     *     {@code materialization}
     * @param affineAddressPairs alternating source/result element addresses for an affine copy,
     *     or {@code null} for pointwise execution; copied defensively
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selections, bindings, carrier patterns, ranges,
     *     parallel facts, materialization, workspace, or affine address geometry disagree
     * @throws ArithmeticException if affine range validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount,
            long minimumElementsPerWorker, CpuWorkerGroup workerGroup,
            Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, Optional.empty());
    }

    /**
     * Creates the complete recipe with optional affine or compact movement geometry.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered unique-input then output selections
     * @param artifact non-null generated artifact matching {@code generatedCarrierPattern}
     * @param bindings non-null full-range boundary bindings; copied defensively
     * @param carrierPattern non-null ordered Runtime carrier pattern; copied defensively
     * @param generatedCarrierPattern non-null ordered generated-entry carrier pattern; copied
     *     defensively
     * @param start non-negative inclusive logical bound
     * @param end exclusive bound no greater than the prepared output count
     * @param selectedRangeCount positive maximum chunk count selected during analysis
     * @param minimumElementsPerWorker positive minimum elements per submitted worker chunk
     * @param workerGroup borrowed open group for a parallel recipe, or {@code null} for a
     *     single-thread recipe; never closed by this executable
     * @param materialization non-null optional pointwise contiguous-input copy plan
     * @param workspaceSelection non-null optional workspace selection, present exactly with
     *     {@code materialization}
     * @param affineAddressPairs alternating source/result element addresses for affine copying,
     *     or {@code null} for pointwise and movement execution; copied defensively
     * @param movementGeometry non-null optional compact movement geometry, present only for a
     *     matching movement artifact and mutually exclusive with affine address pairs
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selections, bindings, carrier patterns, ranges,
     *     parallel facts, materialization, workspace, affine addresses, or movement geometry
     *     disagree
     * @throws ArithmeticException if prepared range or movement element-count validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount,
            long minimumElementsPerWorker, CpuWorkerGroup workerGroup,
            Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                Optional.empty(), Optional.empty());
    }

    /**
     * Creates the complete recipe with mutually exclusive affine, movement, or indexing geometry.
     * Indexing geometry adds deterministic run-bound validation and never permits materialization
     * or workspace selection.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered unique-input then output selections
     * @param artifact non-null generated artifact matching {@code generatedCarrierPattern}
     * @param bindings non-null full-range boundary bindings; copied defensively
     * @param carrierPattern non-null ordered Runtime carrier pattern; copied defensively
     * @param generatedCarrierPattern non-null ordered generated-entry carrier pattern; copied
     *     defensively
     * @param start non-negative inclusive output logical bound
     * @param end exclusive output logical bound no greater than the prepared output count
     * @param selectedRangeCount positive maximum chunk count selected during analysis
     * @param minimumElementsPerWorker positive minimum elements per submitted worker chunk
     * @param workerGroup borrowed open group for a parallel recipe, or {@code null} for a
     *     single-thread recipe; never closed by this executable
     * @param materialization non-null optional pointwise contiguous-input copy plan; empty for
     *     indexing
     * @param workspaceSelection non-null optional workspace selection, present exactly with
     *     {@code materialization}; empty for indexing
     * @param affineAddressPairs alternating source/result addresses for affine copying, or
     *     {@code null} for pointwise, movement, and indexing execution; copied defensively
     * @param movementGeometry non-null optional compact movement geometry
     * @param indexingGeometry non-null optional compact indexing geometry, mutually exclusive
     *     with affine and movement geometry
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selections, bindings, carrier patterns, ranges,
     *     parallel facts, materialization, workspace, affine addresses, or geometry disagree
     * @throws ArithmeticException if prepared count or range geometry overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, Optional.empty());
    }

    /**
     * Creates the complete direct recipe with optional scatter geometry and declared scratch.
     *
     * @param memoryPlan exact prepared memory plan
     * @param selections ordered unique-input then output buffer selections
     * @param artifact verified generated artifact
     * @param bindings complete boundary access geometry
     * @param carrierPattern Runtime carrier forms
     * @param generatedCarrierPattern direct generated-entry carrier forms
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum chunk size
     * @param workerGroup borrowed workers for a parallel plan, otherwise {@code null}
     * @param materialization optional pointwise materialization; empty for scatter
     * @param workspaceSelection optional materialization or scatter-product workspace selection
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry optional movement geometry
     * @param indexingGeometry optional gather or one-hot geometry
     * @param scatterGeometry non-null optional compact functional-scatter geometry
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if resource, carrier, range, worker, or geometry facts
     *     disagree
     * @throws ArithmeticException if exact range or geometry arithmetic overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, Optional.empty());
    }

    /**
     * Creates the complete direct recipe with mutually exclusive fold geometry.
     * @param memoryPlan exact prepared memory plan
     * @param selections ordered input then output selections
     * @param artifact verified generated artifact
     * @param bindings complete boundary access geometry
     * @param carrierPattern Runtime carrier forms
     * @param generatedCarrierPattern direct generated-entry carrier forms
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum chunk size
     * @param workerGroup borrowed workers for a parallel plan, otherwise {@code null}
     * @param materialization optional pointwise materialization
     * @param workspaceSelection optional declared workspace selection
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry optional movement geometry
     * @param indexingGeometry optional indexing geometry
     * @param scatterGeometry optional functional-scatter geometry
     * @param foldGeometry non-null optional overlap-fold geometry
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if resource, range, worker, or geometry facts disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, Optional.empty());
    }

    /**
     * Creates the complete direct recipe including mutually exclusive ordering, random,
     * cumulative-scan, or ordinary-aggregate geometry.
     *
     * <p>For SORT and ARGSORT the ordered selections are input then output. TOP_K uses input,
     * values output, then INT64 indices output and declares both outputs write-only. Ordering
     * requires one assigned run-owned workspace sized for disjoint per-range merge regions.
     * Cold binding rejects every input/output and output/output physical overlap before it
     * mutates scratch, writes output, creates a generated call, or submits worker work.</p>
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered input then one or two output buffer selections
     * @param artifact non-null verified generated artifact retained strongly by this recipe
     * @param bindings complete boundary access geometry in selection order; copied defensively
     * @param carrierPattern Runtime carrier forms in selection order; copied defensively
     * @param generatedCarrierPattern direct generated-entry carrier forms; copied defensively
     * @param start inclusive logical element or independent-slice ordinal
     * @param end exclusive logical element or independent-slice ordinal
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed open worker group for a parallel plan, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned materialization, scatter, or ordering
     *     workspace selection
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional static movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional functional-scatter geometry
     * @param foldGeometry non-null optional overlap-fold geometry
     * @param orderingGeometry non-null optional stable ordering geometry
     * @param randomGeometry non-null optional explicit-state initializer/dropout geometry
     * @param scanGeometry non-null optional cumulative-scan slice/layout geometry
     * @param aggregateGeometry non-null optional ordinary aggregate output/domain geometry
     * @param argExtremaGeometry non-null optional one-axis logical-index output/domain geometry
     * @param maskedReductionGeometry non-null optional axis-removing directional masked SUM/MEAN
     *     broadcast, output-cell, and exact-state geometry
     * @param advancedReductionGeometry non-null optional logarithmic, statistical, or norm
     *     complete-output-cell and exact-state geometry
     * @param softmaxGeometry non-null optional zero-workspace complete-slice normalization geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if memory, boundary, carrier, range, worker, geometry,
     *     workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry,
                advancedReductionGeometry, softmaxGeometry, Optional.empty());
    }

    /**
     * Creates the complete direct recipe including optional trailing-normalization geometry.
     *
     * <p>Exactly one optional route geometry may be present. For trailing normalization,
     * selections are unique logical inputs in first-occurrence order followed by one output;
     * Layer may pair with one assigned exact-state workspace and RMS must not. The recipe retains
     * immutable geometry and a borrowed worker group but does not own carrier instances or
     * per-run workspace.</p>
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered unique inputs followed by output buffer selections
     * @param artifact non-null verified generated artifact retained strongly by this recipe
     * @param bindings complete boundary access geometry in selection order; copied defensively
     * @param carrierPattern Runtime carrier forms in selection order; copied defensively
     * @param generatedCarrierPattern direct generated-entry carrier forms; copied defensively
     * @param start inclusive logical element, output-cell, or complete-slice ordinal
     * @param end exclusive logical element, output-cell, or complete-slice ordinal
     * @param selectedRangeCount positive selected maximum simultaneous range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed open worker group for a parallel plan, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned route workspace; Layer normalization
     *     uses exact-state scratch and RMS normalization uses none
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional static movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional functional-scatter geometry
     * @param foldGeometry non-null optional overlap-fold geometry
     * @param orderingGeometry non-null optional stable ordering geometry
     * @param randomGeometry non-null optional explicit-state initializer/dropout geometry
     * @param scanGeometry non-null optional cumulative-scan slice/layout geometry
     * @param aggregateGeometry non-null optional ordinary aggregate output/domain geometry
     * @param argExtremaGeometry non-null optional logical-index reduction geometry
     * @param maskedReductionGeometry non-null optional directional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional stable-softmax complete-slice geometry
     * @param trailingNormalizationGeometry non-null optional Layer/RMS trailing-slice geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if memory, boundaries, carriers, range, worker, route,
     *     workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range, span, or workspace validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry,
                advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry,
                Optional.empty(), Optional.empty());
    }

    /**
     * Creates the complete direct recipe including optional batch normalization or Conv2d
     * geometry.
     *
     * <p>Exactly one specialized-family geometry may be present. Batch-normalization inference
     * retains unique inputs in first-occurrence order followed by output, owns no carrier or
     * workspace, and interprets {@code start}/{@code end} in the selected channel or flattened
     * non-channel execution domain. Batch-normalization training retains unique inputs followed
     * by five outputs, owns complete-channel ranges, and uses the declared exact-state workspace
     * without retaining cross-run state. Conv2d retains exact resolved NCHW boundary geometry,
     * owns complete output-cell ranges, and uses neither packing nor hidden scratch.</p>
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered unique inputs followed by output selections
     * @param artifact verified generated artifact retained strongly by this recipe
     * @param bindings complete boundary geometry; copied defensively
     * @param carrierPattern Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern generated-entry carrier forms; copied defensively
     * @param start inclusive selected execution-domain bound
     * @param end exclusive selected execution-domain bound
     * @param selectedRangeCount positive selected maximum simultaneous range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed open worker group for parallel execution, otherwise {@code null}
     * @param materialization non-null optional pointwise materialization
     * @param workspaceSelection non-null optional route workspace
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional random geometry
     * @param scanGeometry non-null optional scan geometry
     * @param aggregateGeometry non-null optional aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @param conv2dGeometry non-null optional grouped NCHW Conv2d boundary geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if memory, boundary, carrier, range, worker, route,
     *     workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range, span, or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry,
                advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry,
                batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry,
                Optional.empty(), Optional.empty(),
                establishedOutputCount(batchNormTrainingGeometry, randomGeometry,
                        orderingGeometry));
    }

    /**
     * Creates one direct generated unit with an explicit count of trailing output boundaries.
     *
     * <p>The output count defines which final selections are declared write-only at the Runtime
     * boundary; every preceding selection is read-only. It is normally derived from the family,
     * but the exact two-unit Conv2d exception supplies it explicitly so its intermediate and final
     * writes remain visible to the outer atomic sequence. This recipe owns no representation,
     * workspace, worker-group lifecycle, or validity transition.</p>
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered input then trailing output buffer selections
     * @param artifact verified generated artifact retained strongly by this recipe
     * @param bindings complete boundary geometry; copied defensively
     * @param carrierPattern Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern generated-entry carrier forms; copied defensively
     * @param start inclusive logical element, cell, slice, or channel bound
     * @param end exclusive logical element, cell, slice, or channel bound
     * @param selectedRangeCount positive selected maximum simultaneous range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed worker group for parallel execution, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned route workspace
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional explicit-state geometry
     * @param scanGeometry non-null optional scan geometry
     * @param aggregateGeometry non-null optional aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @param conv2dGeometry non-null optional grouped NCHW Conv2d boundary geometry
     * @param conv3dGeometry non-null optional grouped NCDHW Conv3d boundary geometry
     * @param matmulGeometry non-null optional normalized full-K MATMUL geometry
     * @param outputCount positive number of trailing selections written by this unit
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if memory, boundary, carrier, range, worker, route,
     *     output-count, workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range, span, or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<CpuMatmulLowering.Geometry> matmulGeometry, int outputCount) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization.isPresent() ? Optional.empty() : workspaceSelection,
                materialization.stream().toList(), materialization.isPresent()
                        ? workspaceSelection.stream().toList() : List.of(),
                affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulGeometry,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), outputCount,
                Optional.empty(), null);
    }

    /**
     * Creates a computation recipe whose copied inputs are already-filled representation
     * workspaces. This constructor never performs the copies.
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered input then trailing output buffer selections
     * @param artifact verified generated consumer artifact retained strongly by this recipe
     * @param bindings complete represented boundary geometry; copied defensively
     * @param carrierPattern Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern generated-entry carrier forms with copied inputs replaced by
     *     workspace segments; copied defensively
     * @param start inclusive logical element, cell, slice, or channel bound
     * @param end exclusive logical element, cell, slice, or channel bound
     * @param selectedRangeCount positive selected maximum simultaneous range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param workerGroup borrowed worker group for parallel execution, otherwise {@code null}
     * @param workspaceSelection non-null optional family-owned route workspace
     * @param representationMaterializations non-null ordered copies consumed by this unit; copied
     *     defensively
     * @param representationWorkspaceSelections non-null assigned copy workspaces aligned with the
     *     copies; copied defensively
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional explicit-state geometry
     * @param scanGeometry non-null optional scan geometry
     * @param aggregateGeometry non-null optional aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @param conv2dGeometry non-null optional grouped NCHW Conv2d boundary geometry
     * @param conv3dGeometry non-null optional grouped NCDHW Conv3d boundary geometry
     * @param matmulGeometry non-null optional normalized full-K MATMUL geometry
     * @param pool2dGeometry non-null optional NCHW Pool2d boundary geometry
     * @param pool3dGeometry non-null optional NCDHW Pool3d boundary geometry
     * @param attentionGeometry non-null optional scaled-dot-product-attention row geometry
     * @param lossGeometry non-null optional direct loss rank/layout/base-packing geometry
     * @param outputCount positive number of trailing selections written by this unit
     * @param partialReductionArtifact non-null optional cold-bound partial/combine artifact;
     *     when present it requires the matching private aligned workspace and worker geometry
     * @param concurrencyBudget borrowed coordinated budget for inline execution, or {@code null}
     *     for compatibility or when {@code workerGroup} owns parallel acquisition
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if representation plans/workspaces, memory, boundary,
     *     carrier, range, worker, route, output-count, workspace, or specialization facts disagree
     * @throws ArithmeticException if exact range, span, or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<WorkspaceSelection> workspaceSelection,
            List<CpuMaterializationPlan> representationMaterializations,
            List<WorkspaceSelection> representationWorkspaceSelections,
            long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<CpuMatmulLowering.Geometry> matmulGeometry,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
            Optional<CpuAttentionLowering.Geometry> attentionGeometry,
            Optional<CpuLossLowering.Geometry> lossGeometry, int outputCount,
            Optional<CpuGeneratedKernel.PartialReductionArtifact> partialReductionArtifact,
            CpuConcurrencyBudget concurrencyBudget) {
        super(memoryPlan, selections, java.util.stream.Stream.concat(workspaceSelection.stream(),
                representationWorkspaceSelections.stream()).toList(),
                accesses(selections.size(), outputCount));
        this.batchRoute = null;
        if (selections.isEmpty()) throw new IllegalArgumentException("at least one buffer required");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
        this.partialReductionArtifact = Objects.requireNonNull(partialReductionArtifact,
                "partialReductionArtifact");
        this.bindings = List.copyOf(bindings);
        this.carrierPattern = List.copyOf(carrierPattern);
        this.generatedCarrierPattern = List.copyOf(generatedCarrierPattern);
        if (this.bindings.size() != selections.size()
                || this.carrierPattern.size() != selections.size()
                || this.generatedCarrierPattern.size() != selections.size()
                || !artifact.specialization().carrierPattern().equals(this.generatedCarrierPattern)) {
            throw new IllegalArgumentException("binding and specialization patterns must agree");
        }
        this.movementGeometry = Objects.requireNonNull(movementGeometry, "movementGeometry");
        this.indexingGeometry = Objects.requireNonNull(indexingGeometry, "indexingGeometry");
        this.scatterGeometry = Objects.requireNonNull(scatterGeometry, "scatterGeometry");
        this.foldGeometry = Objects.requireNonNull(foldGeometry, "foldGeometry");
        this.orderingGeometry = Objects.requireNonNull(orderingGeometry, "orderingGeometry");
        this.randomGeometry = Objects.requireNonNull(randomGeometry, "randomGeometry");
        this.scanGeometry = Objects.requireNonNull(scanGeometry, "scanGeometry");
        this.aggregateGeometry = Objects.requireNonNull(aggregateGeometry, "aggregateGeometry");
        this.argExtremaGeometry = Objects.requireNonNull(argExtremaGeometry, "argExtremaGeometry");
        this.maskedReductionGeometry = Objects.requireNonNull(maskedReductionGeometry,
                "maskedReductionGeometry");
        this.advancedReductionGeometry = Objects.requireNonNull(advancedReductionGeometry,
                "advancedReductionGeometry");
        this.softmaxGeometry = Objects.requireNonNull(softmaxGeometry, "softmaxGeometry");
        this.trailingNormalizationGeometry = Objects.requireNonNull(trailingNormalizationGeometry,
                "trailingNormalizationGeometry");
        this.batchNormInferenceGeometry = Objects.requireNonNull(batchNormInferenceGeometry,
                "batchNormInferenceGeometry");
        this.batchNormTrainingGeometry = Objects.requireNonNull(batchNormTrainingGeometry,
                "batchNormTrainingGeometry");
        this.conv2dGeometry = Objects.requireNonNull(conv2dGeometry, "conv2dGeometry");
        this.conv3dGeometry = Objects.requireNonNull(conv3dGeometry, "conv3dGeometry");
        this.matmulGeometry = Objects.requireNonNull(matmulGeometry, "matmulGeometry");
        this.pool2dGeometry = Objects.requireNonNull(pool2dGeometry, "pool2dGeometry");
        this.pool3dGeometry = Objects.requireNonNull(pool3dGeometry, "pool3dGeometry");
        this.attentionGeometry = Objects.requireNonNull(attentionGeometry, "attentionGeometry");
        this.lossGeometry = Objects.requireNonNull(lossGeometry, "lossGeometry");
        this.outputCount = outputCount;
        if (outputCount <= 0 || outputCount > selections.size()) {
            throw new IllegalArgumentException("output boundary count is inconsistent");
        }
        long count = artifact.specialization().rowMapReductionIr().isPresent() ? 1 : this.lossGeometry.isPresent() ? end : this.attentionGeometry.isPresent()
                ? this.attentionGeometry.orElseThrow().rowCount()
                : this.pool3dGeometry.isPresent()
                ? this.pool3dGeometry.orElseThrow().outputCount()
                : this.pool2dGeometry.isPresent()
                ? this.pool2dGeometry.orElseThrow().outputCount()
                : this.matmulGeometry.isPresent() ? this.matmulGeometry.orElseThrow().outputCount()
                : this.conv2dGeometry.isPresent() || this.conv3dGeometry.isPresent()
                ? this.bindings.getLast().elementCount()
                : this.batchNormTrainingGeometry.isPresent()
                ? this.batchNormTrainingGeometry.orElseThrow().channelCount()
                : this.batchNormInferenceGeometry.isPresent()
                ? this.batchNormInferenceGeometry.orElseThrow().rangeItemCount()
                : this.trailingNormalizationGeometry.isPresent()
                ? (this.trailingNormalizationGeometry.orElseThrow().normalizedCount() == 0 ? 0
                    : this.trailingNormalizationGeometry.orElseThrow().leadingCount())
                : this.softmaxGeometry.isPresent()
                ? this.softmaxGeometry.orElseThrow().sliceCount()
                : this.advancedReductionGeometry.isPresent()
                ? this.advancedReductionGeometry.orElseThrow().outputCount()
                : this.maskedReductionGeometry.isPresent()
                ? this.maskedReductionGeometry.orElseThrow().outputCount()
                : this.argExtremaGeometry.isPresent()
                ? this.argExtremaGeometry.orElseThrow().outputCount()
                : this.aggregateGeometry.isPresent()
                ? this.aggregateGeometry.orElseThrow().outputCount()
                : this.scanGeometry.isPresent()
                ? this.scanGeometry.orElseThrow().sliceCount()
                : this.randomGeometry.isPresent()
                ? this.randomGeometry.orElseThrow().elementCount()
                : this.orderingGeometry.isPresent()
                ? this.orderingGeometry.orElseThrow().sliceCount()
                : this.foldGeometry.isPresent()
                ? elementCount(this.foldGeometry.orElseThrow().outputExtents())
                : this.scatterGeometry.isPresent()
                ? elementCount(this.scatterGeometry.orElseThrow().outputExtents())
                : this.indexingGeometry.isPresent()
                ? elementCount(this.indexingGeometry.orElseThrow().outputExtents())
                : this.movementGeometry.isPresent()
                ? elementCount(this.movementGeometry.orElseThrow().outputExtents())
                : this.bindings.getLast().elementCount();
        if (start < 0 || end < start || end > count) throw new IllegalArgumentException("invalid range");
        if (artifact.specialization().rowMapReductionIr().isPresent()
                && (start != 0 || artifact.specialization().rowFusionEmission().isEmpty() && (selectedRangeCount != 1 || workerGroup != null)
                    || outputCount != artifact.specialization().rowMapReductionIr().orElseThrow().arithmetic().stores().size()))
            throw new IllegalArgumentException("row-region execution owns one unsplittable full-body item");
        if (artifact.specialization().rowFusionEmission().isPresent()
                && selectedRangeCount != artifact.specialization().rowFusionEmission().orElseThrow().maximumPartialCount())
            throw new IllegalArgumentException("row range policy exceeds authenticated partial layout");
        this.start = start;
        this.end = end;
        if (selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                || (selectedRangeCount >= 2) != (workerGroup != null)) {
            throw new IllegalArgumentException("parallel execution facts are inconsistent");
        }
        this.selectedRangeCount = selectedRangeCount;
        this.minimumElementsPerWorker = minimumElementsPerWorker;
        this.workerGroup = workerGroup;
        this.concurrencyBudget = concurrencyBudget;
        if (workerGroup != null && concurrencyBudget != null) {
            throw new IllegalArgumentException("parallel execution acquires only through worker group");
        }
        if (this.partialReductionArtifact.isPresent()) {
            var partial = this.partialReductionArtifact.orElseThrow();
            if (bindings.size() != 2 || outputCount != 1 || this.carrierPattern.size() != 2
                    || (partial.ir().dataType()
                            == io.github.pho001.synaptik.model.datatype.DataType.INT32
                            ? !this.carrierPattern.equals(List.of(CarrierAccess.INT_ARRAY,
                                    CarrierAccess.INT_ARRAY))
                            : !this.carrierPattern.equals(List.of(CarrierAccess.LONG_ARRAY,
                                    CarrierAccess.LONG_ARRAY)))
                    || selectedRangeCount != partial.ir().partialCount()
                    || workerGroup == null || workerGroup.workerCount() < partial.ir().partialCount()
                    || end - start != partial.ir().outputCount()) {
                throw new IllegalArgumentException("partial-reduction prepared facts disagree");
            }
        }
        this.materialization = Optional.empty();
        this.workspaceSelection = Objects.requireNonNull(workspaceSelection, "workspaceSelection");
        this.representationMaterializations = List.copyOf(representationMaterializations);
        this.representationWorkspaceSelections = List.copyOf(representationWorkspaceSelections);
        if (this.representationMaterializations.size() > 2
                || this.representationMaterializations.size()
                        != this.representationWorkspaceSelections.size()) {
            throw new IllegalArgumentException("representation workspace selections disagree");
        }
        this.affineCopy = affineAddressPairs != null;
        this.affineAddressPairs = affineAddressPairs == null ? new long[0] : affineAddressPairs.clone();
        if (affineCopy && (this.affineAddressPairs.length % 2 != 0
                || this.affineAddressPairs.length < Math.multiplyExact(end, 2))) {
            throw new IllegalArgumentException("affine address pairs must cover the selected copy range");
        }
        int geometryCount=(affineCopy?1:0)+(this.movementGeometry.isPresent()?1:0)
                +(this.indexingGeometry.isPresent()?1:0)+(this.scatterGeometry.isPresent()?1:0)
                +(this.foldGeometry.isPresent()?1:0)+(this.orderingGeometry.isPresent()?1:0)
                +(this.randomGeometry.isPresent()?1:0)+(this.scanGeometry.isPresent()?1:0)
                +(this.aggregateGeometry.isPresent()?1:0)+(this.argExtremaGeometry.isPresent()?1:0);
        geometryCount += this.maskedReductionGeometry.isPresent() ? 1 : 0;
        geometryCount += this.advancedReductionGeometry.isPresent() ? 1 : 0;
        geometryCount += this.softmaxGeometry.isPresent() ? 1 : 0;
        geometryCount += this.trailingNormalizationGeometry.isPresent() ? 1 : 0;
        geometryCount += this.batchNormInferenceGeometry.isPresent() ? 1 : 0;
        geometryCount += this.batchNormTrainingGeometry.isPresent() ? 1 : 0;
        geometryCount += this.conv2dGeometry.isPresent() ? 1 : 0;
        geometryCount += this.conv3dGeometry.isPresent() ? 1 : 0;
        geometryCount += this.matmulGeometry.isPresent() ? 1 : 0;
        geometryCount += this.pool2dGeometry.isPresent() ? 1 : 0;
        geometryCount += this.pool3dGeometry.isPresent() ? 1 : 0;
        geometryCount += this.attentionGeometry.isPresent() ? 1 : 0;
        geometryCount += this.lossGeometry.isPresent() ? 1 : 0;
        if (geometryCount>1) {
            throw new IllegalArgumentException("affine, movement, indexing, scatter, and fold geometry are exclusive");
        }
        boolean scatterScratch=this.scatterGeometry.filter(g->g.scratchSliceBytes()>0).isPresent();
        boolean aggregateScratch=this.aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent();
        boolean maskedScratch=this.maskedReductionGeometry.filter(g -> g.outputCount() > 0)
                .isPresent();
        boolean advancedScratch=this.advancedReductionGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent();
        boolean normalizationScratch=this.trailingNormalizationGeometry
                .filter(g -> g.scratchSliceBytes() > 0 && g.normalizedCount() > 0).isPresent();
        boolean trainingScratch=this.batchNormTrainingGeometry
                .filter(g -> g.scratchSliceBytes() > 0 && g.channelCount() > 0).isPresent();
        boolean attentionScratch=this.attentionGeometry
                .filter(g -> g.scratchSliceBytes() > 0 && g.rowCount() > 0).isPresent();
        if (workspaceSelection.isPresent() != (scatterScratch
                || aggregateScratch || maskedScratch || advancedScratch
                || normalizationScratch || trainingScratch || attentionScratch
                || this.orderingGeometry.isPresent() || this.partialReductionArtifact.isPresent()
                || artifact.specialization().rowFusionEmission().isPresent())) {
            throw new IllegalArgumentException("workspace selection purpose is inconsistent");
        }
        this.representationMaterializationPositions = representationPositions(
                this.representationMaterializations, this.bindings, this.carrierPattern,
                this.generatedCarrierPattern,
                artifact.specialization().materializedSourcePosition());
    }

    /**
     * Creates a direct or legacy single-materialization recipe with optional Pool2d geometry.
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered input then trailing output selections
     * @param artifact verified generated artifact retained by this recipe
     * @param bindings complete represented boundary geometry; copied defensively
     * @param carrierPattern Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern generated-entry carrier forms; copied defensively
     * @param start inclusive logical output-cell bound
     * @param end exclusive logical output-cell bound
     * @param selectedRangeCount positive selected maximum simultaneous range count
     * @param minimumElementsPerWorker positive minimum output cells per worker chunk
     * @param workerGroup borrowed worker group for parallel execution, otherwise {@code null}
     * @param materialization non-null optional pointwise input materialization
     * @param workspaceSelection non-null optional assigned workspace selection
     * @param affineAddressPairs affine copy pairs, or {@code null}
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional explicit-state geometry
     * @param scanGeometry non-null optional scan geometry
     * @param aggregateGeometry non-null optional aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @param conv2dGeometry non-null optional grouped NCHW Conv2d geometry
     * @param conv3dGeometry non-null optional grouped NCDHW Conv3d geometry
     * @param matmulGeometry non-null optional MATMUL geometry
     * @param pool2dGeometry non-null optional NCHW Pool2d geometry
     * @param pool3dGeometry non-null optional NCDHW Pool3d geometry
     * @param outputCount positive number of trailing output selections
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if recipe, geometry, range, or carrier facts disagree
     * @throws ArithmeticException if exact range, span, or geometry validation overflows
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<CpuMatmulLowering.Geometry> matmulGeometry,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry, int outputCount) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization.isPresent() ? Optional.empty() : workspaceSelection,
                materialization.stream().toList(), materialization.isPresent()
                        ? workspaceSelection.stream().toList() : List.of(),
                affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulGeometry,
                pool2dGeometry, pool3dGeometry, Optional.empty(), Optional.empty(), outputCount,
                Optional.empty(), null);
    }

    /**
     * Creates the complete recipe variant that carries cold attention or direct-loss geometry.
     *
     * @param memoryPlan exact immutable prepared memory plan
     * @param selections ordered input then trailing output buffer selections
     * @param artifact verified generated artifact retained by this recipe
     * @param bindings complete represented boundary geometry
     * @param carrierPattern Runtime carrier forms in boundary order
     * @param generatedCarrierPattern generated-entry carrier forms in boundary order
     * @param start inclusive logical work bound
     * @param end exclusive logical work bound
     * @param selectedRangeCount positive maximum simultaneous range count
     * @param minimumElementsPerWorker positive work threshold for a worker range
     * @param workerGroup borrowed worker group, or {@code null} for serial execution
     * @param materialization non-null optional retained representation-copy plan
     * @param workspaceSelection non-null optional assigned workspace selection
     * @param affineAddressPairs nullable affine-copy address pairs
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional explicit-state random geometry
     * @param scanGeometry non-null optional cumulative-scan geometry
     * @param aggregateGeometry non-null optional aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
     * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
     * @param batchNormTrainingGeometry non-null optional batch-normalization training geometry
     * @param conv2dGeometry non-null optional grouped NCHW Conv2d geometry
     * @param conv3dGeometry non-null optional grouped NCDHW Conv3d geometry
     * @param matmulGeometry non-null optional full-K MATMUL geometry
     * @param pool2dGeometry non-null optional NCHW Pool2d geometry
     * @param pool3dGeometry non-null optional NCDHW Pool3d geometry
     * @param attentionGeometry non-null optional attention geometry
     * @param lossGeometry non-null optional loss geometry, including the index pre-write domain
     * @param outputCount positive trailing output-boundary count
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if retained facts disagree
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
            CpuGeneratedKernel artifact, List<CpuAccessPlan.Binding> bindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long start, long end, int selectedRangeCount, long minimumElementsPerWorker,
            CpuWorkerGroup workerGroup, Optional<CpuMaterializationPlan> materialization,
            Optional<WorkspaceSelection> workspaceSelection, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<CpuMatmulLowering.Geometry> matmulGeometry,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
            Optional<CpuAttentionLowering.Geometry> attentionGeometry,
            Optional<CpuLossLowering.Geometry> lossGeometry, int outputCount) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization.isPresent() ? Optional.empty() : workspaceSelection,
                materialization.stream().toList(), materialization.isPresent()
                        ? workspaceSelection.stream().toList() : List.of(),
                affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulGeometry,
                pool2dGeometry, pool3dGeometry, attentionGeometry, lossGeometry, outputCount,
                Optional.empty(), null);
    }

    /**
     * Resolves complete-plan copy identities to this generated unit's local input positions.
     * The generated specialization disambiguates a reused single copy; multiple copies must
     * identify one unique retained consumer-coordinate vector. No graph identity is consulted.
     *
     * @param copies selected complete-plan copies consumed by this unit
     * @param bindings representation-adjusted unit-local bindings
     * @param carriers original unit-local carrier forms
     * @param generatedCarriers representation-adjusted unit-local carrier forms
     * @param specializedPosition generated single-copy position, or {@code -1}
     * @return immutable copy-aligned unit-local input positions
     * @throws IllegalArgumentException if retained facts do not identify one exact local mapping
     */
    private static List<Integer> representationPositions(List<CpuMaterializationPlan> copies,
            List<CpuAccessPlan.Binding> bindings, List<CarrierAccess> carriers,
            List<CarrierAccess> generatedCarriers, int specializedPosition) {
        if (copies.isEmpty()) {
            if (specializedPosition != -1) throw new IllegalArgumentException(
                    "generated specialization has no selected materialization");
            return List.of();
        }
        var candidates = new ArrayList<List<Integer>>();
        for (int unit = 0; unit < 8; unit++) {
            int unitPosition = unit;
            var positions = new ArrayList<Integer>(copies.size());
            boolean compatible = true;
            for (CpuMaterializationPlan copy : copies) {
                var consumers = copy.consumers().stream()
                        .filter(value -> value.unitPosition() == unitPosition).toList();
                if (consumers.size() != 1) {
                    compatible = false;
                    break;
                }
                int position = consumers.getFirst().boundaryPosition();
                if (position < 0 || position >= bindings.size()
                        || !bindings.get(position).equals(copy.consumerBinding())
                        || carriers.get(position) != copy.sourceCarrier()
                        || generatedCarriers.get(position) != CarrierAccess.MEMORY_SEGMENT) {
                    compatible = false;
                    break;
                }
                positions.add(position);
            }
            if (compatible && (copies.size() != 1
                    ? specializedPosition == -1
                    : positions.getFirst() == specializedPosition)
                    && !candidates.contains(positions)) {
                candidates.add(List.copyOf(positions));
            }
        }
        if (candidates.size() != 1
                || candidates.getFirst().stream().distinct().count() != copies.size()) {
            throw new IllegalArgumentException(
                    "materialization and generated specialization must agree");
        }
        return candidates.getFirst();
    }

    /**
     * Returns the strongly retained generated artifact.
     *
     * @return the non-null verified artifact
     */
    public CpuGeneratedKernel artifact() { return artifact; }

    /**
     * Authenticates a container's nonfixed workspace selection to this genuine batch route.
     * @param selection exact dense workspace position selected by the container
     * @return whether this dynamic unit will validate that workspace when the child binds
     */
    boolean hasDynamicWorkspaceSelection(WorkspaceSelection selection) {
        return batchRoute != null && workspaceSelection.isPresent()
                && workspaceSelection.orElseThrow().equals(selection);
    }

    /**
     * Returns the iteration boundary's geometry adjusted to this recipe's range.
     * For movement this is the output boundary; other routes retain the first boundary contract.
     *
     * @return a new immutable ranged binding
     */
    public CpuAccessPlan.Binding binding() {
        if (artifact.specialization().rowMapReductionIr().isPresent()) return bindings.getLast();
        if (lossGeometry.isPresent() || softmaxGeometry.isPresent() || trailingNormalizationGeometry.isPresent()
                || batchNormInferenceGeometry.isPresent() || batchNormTrainingGeometry.isPresent()
                || conv2dGeometry.isPresent() || conv3dGeometry.isPresent()
                || matmulGeometry.isPresent() || pool2dGeometry.isPresent()
                || pool3dGeometry.isPresent() || attentionGeometry.isPresent()) return bindings.getLast();
        return ranged(movementGeometry.isPresent() || indexingGeometry.isPresent()
                || scatterGeometry.isPresent() || foldGeometry.isPresent() || orderingGeometry.isPresent()
                || randomGeometry.isPresent()
                || scanGeometry.isPresent() || aggregateGeometry.isPresent()
                || argExtremaGeometry.isPresent() || maskedReductionGeometry.isPresent()
                || advancedReductionGeometry.isPresent() || trailingNormalizationGeometry.isPresent()
                ? bindings.getLast() : bindings.getFirst());
    }
    /**
     * Returns the exact geometry for every ordered boundary.
     * Movement inputs retain their complete input domains while its output carries this recipe's
     * output range; same-domain routes range every boundary.
     *
     * @return a new immutable list in boundary order with route-appropriate ranges
     */
    public List<CpuAccessPlan.Binding> accessBindings() {
        if (artifact.specialization().rowMapReductionIr().isPresent()) return bindings;
        if (lossGeometry.isPresent() || softmaxGeometry.isPresent() || trailingNormalizationGeometry.isPresent()
                || batchNormInferenceGeometry.isPresent() || batchNormTrainingGeometry.isPresent()
                || conv2dGeometry.isPresent() || conv3dGeometry.isPresent()
                || matmulGeometry.isPresent() || pool2dGeometry.isPresent()
                || pool3dGeometry.isPresent() || attentionGeometry.isPresent()) return bindings;
        if (movementGeometry.isPresent() || indexingGeometry.isPresent()
                || scatterGeometry.isPresent() || foldGeometry.isPresent() || orderingGeometry.isPresent()
                || randomGeometry.isPresent() || scanGeometry.isPresent()
                || aggregateGeometry.isPresent() || argExtremaGeometry.isPresent()
                || maskedReductionGeometry.isPresent() || advancedReductionGeometry.isPresent()) {
            var result = new ArrayList<CpuAccessPlan.Binding>(bindings);
            if (orderingGeometry.isEmpty() && randomGeometry.isEmpty())
                result.set(result.size() - 1, ranged(result.getLast()));
            return List.copyOf(result);
        }
        return bindings.stream().map(this::ranged).toList();
    }

    /**
     * Returns an immutable recipe sharing the artifact, full geometry, and slots for another
     * half-open logical range.
     *
     * @param rangeStart non-negative inclusive logical element bound
     * @param rangeEnd exclusive logical element bound no greater than the element count
     * @return a new immutable recipe; never {@code null}
     * @throws IllegalArgumentException if the range is negative, reversed, or out of bounds
     */
    public CpuPreparedExecutable forRange(long rangeStart, long rangeEnd) {
        var selections = new ArrayList<BufferSelection>(bindings.size());
        for (int i = 0; i < bindings.size(); i++) selections.add(bufferSelection(i));
        return new CpuPreparedExecutable(memoryPlan(), selections, artifact, bindings,
                carrierPattern, generatedCarrierPattern, rangeStart, rangeEnd, selectedRangeCount,
                minimumElementsPerWorker, workerGroup, workspaceSelection,
                representationMaterializations, representationWorkspaceSelections,
                affineCopy ? affineAddressPairs : null, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry,
                advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry,
                batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry,
                conv3dGeometry, matmulGeometry, pool2dGeometry, pool3dGeometry,
                attentionGeometry, lossGeometry, outputCount, partialReductionArtifact,
                concurrencyBudget);
    }

    private CpuAccessPlan.Binding ranged(CpuAccessPlan.Binding source) {
        return ranged(source, start, end);
    }

    private static CpuAccessPlan.Binding ranged(CpuAccessPlan.Binding source,
            long rangeStart, long rangeEnd) {
        return CpuAccessPlan.Binding.create(source.plan(), source.extents().stream()
                        .mapToLong(Long::longValue).toArray(), source.baseElementOffset(),
                source.effectiveStrides().stream().mapToLong(Long::longValue).toArray(),
                source.elementCount(), rangeStart, rangeEnd, source.referencedElementSpan());
    }

    /**
     * Retains the external source span when generation consumes a canonical materialization.
     * @param index ordered external boundary position
     * @return original fixed referenced bytes, never the replacement workspace's span
     * @throws ArithmeticException if checked represented byte sizing overflows
     */
    private long retainedBoundaryBytes(int index) {
        int represented = -1;
        for (int p = 0; p < representationMaterializationPositions.size(); p++) {
            if (representationMaterializationPositions.get(p) == index) {
                represented = p;
                break;
            }
        }
        var binding = represented < 0 ? bindings.get(index)
                : representationMaterializations.get(represented).sourceBinding();
        return Math.multiplyExact(binding.referencedElementSpan(),
                (long) artifact.specialization().boundaryDataTypes().get(index).byteWidth());
    }

    /**
     * Authenticates the current typed usable span against this invocation's numeric recipe.
     * A fixed unit in a complete batch plan retains its original bindings and generated entry;
     * every selected declaration must itself be fixed and match those retained referenced spans.
     * @param state exact run owner and resolved geometry
     * @param index ordered selected boundary position
     * @param value actual CPU carrier or an incompatible representation to reject
     * @return whether dtype, usable bytes, access, carrier and alignment match
     */
    @Override protected boolean acceptsBufferRepresentation(RunState state, int index, BufferRepresentation value) {
        if (batchRoute == null) {
            if (!state.memoryPlan().requiresBatchSize()) return acceptsBufferRepresentation(index, value);
            return value instanceof CpuBufferRepresentation cpu
                    && cpu.acceptsFixedGeometry(state, memoryPlan(), bufferSelection(index).bufferIndex(),
                            retainedBoundaryBytes(index))
                    && acceptsBufferRepresentation(index, value);
        }
        if (!(value instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != artifact.specialization().boundaryDataTypes().get(index)
                || cpu.byteSize() != state.geometry().bufferByteSize(bufferSelection(index).bufferIndex())) return false;
        try {
            var argument = preparedArgument(cpu, index);
            long alignment = memoryPlan().buffers().get(bufferSelection(index).bufferIndex()).byteAlignment();
            return carrierAccess(argument) == carrierPattern.get(index)
                    && argument.byteOffset() % cpu.dataType().byteWidth() == 0 && argument.byteSize() % cpu.dataType().byteWidth() == 0
                    && (!(argument instanceof CpuBufferArgument.Segment s) || s.segment().address() % alignment == 0)
                    && (index < bufferSelectionCount() - outputCount || !argument.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    /**
     * Keeps fixed scratch under its original route-specific size/alignment contract; only a real
     * batch route consumes actual B. Fixed materialization workspaces retain their own selections.
     * @param state exact open run and authenticated geometry
     * @param index ordered workspace selection position
     * @param value selected scratch, or an incompatible representation to reject
     * @return whether declared geometry and the retained route's physical scratch checks agree
     */
    @Override protected boolean acceptsWorkspaceRepresentation(RunState state, int index, WorkspaceRepresentation value) {
        if (batchRoute == null) {
            if (!state.memoryPlan().requiresBatchSize()) return acceptsWorkspaceRepresentation(index, value);
            int intrinsicCount = workspaceSelection.isPresent() ? 1 : 0;
            int position = index < intrinsicCount ? workspaceSelection.orElseThrow().workspaceIndex()
                    : representationWorkspaceSelections.get(index - intrinsicCount).workspaceIndex();
            return value instanceof CpuContiguousWorkspace scratch
                    && CpuBufferRepresentation.acceptsFixedWorkspaceGeometry(
                            state, memoryPlan(), position, scratch.byteSize(), scratch)
                    && acceptsWorkspaceRepresentation(index, value);
        }
        if (!(value instanceof CpuContiguousWorkspace scratch) || !scratch.isAccessible()) return false;
        int position = workspaceSelection.orElseThrow().workspaceIndex();
        long alignment = memoryPlan().workspaces().get(position).byteAlignment();
        return scratch.byteSize() == state.geometry().workspaceByteSize(position)
                && scratch.byteAlignment() == alignment && scratch.writableSegment().address() % alignment == 0;
    }

    @Override protected boolean acceptsBufferRepresentation(int index, BufferRepresentation value) {
        if (!(value instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != artifact.specialization().boundaryDataTypes().get(index)) return false;
        var entry = memoryPlan().buffers().get(bufferSelection(index).bufferIndex());
        if (cpu.byteSize() != entry.byteSize()) return false;
        try {
            CpuBufferArgument argument = preparedArgument(cpu, index);
            CarrierAccess actual = carrierAccess(argument);
            boolean aligned = !(argument instanceof CpuBufferArgument.Segment segment)
                    || segment.segment().address() % entry.byteAlignment() == 0;
            int firstOutput = bindings.size() - outputCount;
            return actual == carrierPattern.get(index) && aligned
                    && (index < firstOutput || !argument.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        if (!(representation instanceof CpuContiguousWorkspace workspace)
                || !workspace.isAccessible()) return false;
        if (artifact.specialization().rowFusionEmission().isPresent()) {
            var requirement = memoryPlan().workspaces().get(workspaceSelection.orElseThrow().workspaceIndex());
            return index == 0 && workspace.byteSize() == requirement.byteSize() && workspace.byteAlignment() >= 8
                    && workspace.writableSegment().address()%8 == 0;
        }
        if (partialReductionArtifact.isPresent()) {
            var partial = partialReductionArtifact.orElseThrow();
            return index == 0 && workspace.byteSize() == partial.ir().workspaceBytes()
                    && workspace.byteAlignment() == CpuPartialReductionIr.STATE_SLICE_BYTES
                    && workspace.writableSegment().address() % CpuPartialReductionIr.STATE_SLICE_BYTES
                            == 0;
        }
        int intrinsicCount = workspaceSelection.isPresent() ? 1 : 0;
        if (index >= intrinsicCount) {
            int copyIndex = index - intrinsicCount;
            if (copyIndex < 0 || copyIndex >= representationMaterializations.size()) return false;
            CpuMaterializationPlan copy = representationMaterializations.get(copyIndex);
            return workspace.byteSize() == copy.byteCount()
                    && workspace.byteAlignment() == copy.byteAlignment()
                    && workspace.writableSegment().address() % copy.byteAlignment() == 0;
        }
        if (index != 0 || scatterGeometry.isEmpty()
                    && orderingGeometry.isEmpty() && aggregateGeometry.isEmpty()
                    && maskedReductionGeometry.isEmpty() && advancedReductionGeometry.isEmpty()
                    && trailingNormalizationGeometry.isEmpty() && batchNormTrainingGeometry.isEmpty()
                    && attentionGeometry.isEmpty()
                ) return false;
        long bytes=scatterGeometry.filter(g -> g.scratchSliceBytes() > 0)
                        .map(g -> g.workspaceBytes(selectedRangeCount))
                        .orElseGet(() -> aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0)
                        .map(g -> g.workspaceBytes(selectedRangeCount))
                        .orElseGet(() -> maskedReductionGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> advancedReductionGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> trailingNormalizationGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> batchNormTrainingGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> attentionGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> orderingGeometry.orElseThrow()
                                .workspaceBytes(selectedRangeCount))))))));
        long alignment=8L;
        return workspace.byteSize() == bytes && workspace.byteAlignment() == alignment
                && workspace.writableSegment().address() % alignment == 0;
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (batchRoute != null) return bindOrdinaryBatch(state, buffers, workspaces);
        if (partialReductionArtifact.isPresent()) {
            return bindPartialReduction(state, buffers, workspaces,
                    partialReductionArtifact.orElseThrow());
        }
        boolean hasWorkspace=artifact.specialization().rowFusionEmission().isPresent() || scatterGeometry.filter(g->g.scratchSliceBytes()>0).isPresent()
                || aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.outputCount() > 0).isPresent()
                || maskedReductionGeometry.filter(g -> g.outputCount() > 0).isPresent()
                || advancedReductionGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.outputCount() > 0).isPresent()
                || trailingNormalizationGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.normalizedCount() > 0).isPresent()
                || batchNormTrainingGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.channelCount() > 0).isPresent()
                || attentionGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.rowCount() > 0).isPresent()
                || orderingGeometry.isPresent();
        int intrinsicCount = hasWorkspace ? 1 : 0;
        if (workspaces.length != intrinsicCount + representationMaterializations.size()) {
            throw new IllegalArgumentException("workspace count disagrees with prepared use");
        }
        var arguments = new ArrayList<CpuBufferArgument>(bindings.size());
        for (int index = 0; index < buffers.length; index++) {
            arguments.add(preparedArgument((CpuBufferRepresentation) buffers[index], index));
        }
        int firstOutputIndex = bindings.size() - outputCount;
        if (randomGeometry.filter(g -> g.family()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT)
                .isPresent()) {
            for (int input = 0; input < 2; input++) for (int output = 2; output < 5; output++) {
                if (overlaps(arguments.get(input), bindings.get(input), arguments.get(output),
                        bindings.get(output))) throw new IllegalArgumentException(
                                "output accessed span must not overlap an input");
            }
            for (int left = 2; left < 5; left++) for (int right = left + 1; right < 5; right++) {
                if (overlaps(arguments.get(left), bindings.get(left), arguments.get(right),
                        bindings.get(right))) throw new IllegalArgumentException(
                                "random outputs must not overlap");
            }
        } else {
        for (int input = 0; input < firstOutputIndex; input++) {
            int inputPosition = input;
            int representedIndex = representationMaterializationPositions.indexOf(inputPosition);
            CpuMaterializationPlan represented = representedIndex < 0 ? null
                    : representationMaterializations.get(representedIndex);
            CpuAccessPlan.Binding inputBinding = represented == null ? bindings.get(input)
                    : represented.sourceBinding();
            boolean overlap = affineCopy
                    ? affineOverlaps(arguments.get(input), arguments.get(firstOutputIndex), input,
                            firstOutputIndex)
                    : movementGeometry.isPresent() || indexingGeometry.isPresent()
                            || scatterGeometry.isPresent() || foldGeometry.isPresent()
                            || orderingGeometry.isPresent() || scanGeometry.isPresent()
                            || aggregateGeometry.isPresent() || argExtremaGeometry.isPresent()
                            || maskedReductionGeometry.isPresent()
                            || advancedReductionGeometry.isPresent() || softmaxGeometry.isPresent()
                            || trailingNormalizationGeometry.isPresent()
                            || batchNormInferenceGeometry.isPresent() || batchNormTrainingGeometry.isPresent()
                            || conv2dGeometry.isPresent() || conv3dGeometry.isPresent()
                            || matmulGeometry.isPresent() || pool2dGeometry.isPresent()
                            || pool3dGeometry.isPresent() || attentionGeometry.isPresent()
                            || lossGeometry.isPresent() || artifact.specialization().rowMapReductionIr().isPresent()
                        ? overlaps(arguments.get(input), inputBinding,
                            arguments.get(firstOutputIndex), bindings.get(firstOutputIndex))
                        : overlaps(arguments.get(input), ranged(inputBinding),
                            arguments.get(firstOutputIndex), ranged(bindings.get(firstOutputIndex)));
            if (overlap) {
                throw new IllegalArgumentException("output accessed span must not overlap an input");
            }
            for (int output = firstOutputIndex + 1; output < bindings.size(); output++) {
                if (overlaps(arguments.get(input), inputBinding, arguments.get(output),
                        bindings.get(output))) {
                    throw new IllegalArgumentException(
                            "output accessed span must not overlap an input");
                }
            }
        }
        if (batchNormTrainingGeometry.isPresent()) {
            for (int input = 0; input < firstOutputIndex; input++)
                for (int output = firstOutputIndex; output < bindings.size(); output++)
                    if (overlaps(arguments.get(input), bindings.get(input), arguments.get(output),
                            bindings.get(output)))
                        throw new IllegalArgumentException("batch-normalization output must not overlap an input");
            for (int leftOutput = firstOutputIndex; leftOutput < bindings.size(); leftOutput++)
                for (int rightOutput = leftOutput + 1; rightOutput < bindings.size(); rightOutput++)
                    if (overlaps(arguments.get(leftOutput), bindings.get(leftOutput),
                            arguments.get(rightOutput), bindings.get(rightOutput)))
                        throw new IllegalArgumentException("batch-normalization outputs must not overlap");
        }
        for (int left = firstOutputIndex; left < bindings.size(); left++)
            for (int right = left + 1; right < bindings.size(); right++)
                if (overlaps(arguments.get(left), bindings.get(left), arguments.get(right),
                        bindings.get(right)))
                    throw new IllegalArgumentException("outputs must not overlap");
        }
        IndexValidation validation = indexingGeometry.map(g ->
                new IndexValidation(arguments, g)).orElse(null);
        ScatterValidation scatterValidation = scatterGeometry.map(g ->
                new ScatterValidation(arguments, g)).orElse(null);
        for (int copyIndex = 0; copyIndex < representationMaterializations.size(); copyIndex++) {
            var copy = representationMaterializations.get(copyIndex);
            MemorySegment workspace = ((CpuContiguousWorkspace)
                    workspaces[intrinsicCount + copyIndex]).writableSegment();
            arguments.set(representationMaterializationPositions.get(copyIndex),
                    new CpuBufferArgument.Segment(
                    copy.dataType(),
                    workspace.asReadOnly(), workspace.byteSize(), true));
        }
        if (workerGroup != null) for (CpuBufferArgument argument : arguments) {
            if (argument instanceof CpuBufferArgument.Segment segment
                    && !workerGroup.workersCanAccess(segment.segment())) {
                throw new IllegalArgumentException("segment is not accessible to every CPU worker");
            }
        }
        if (workerGroup != null && hasWorkspace && !workerGroup.workersCanAccess(
                ((CpuContiguousWorkspace) workspaces[0]).writableSegment()))
            throw new IllegalArgumentException("scratch is not accessible to every CPU worker");
        if (workerGroup != null) for (int index = intrinsicCount; index < workspaces.length; index++)
            if (!workerGroup.workersCanAccess(((CpuContiguousWorkspace) workspaces[index])
                    .writableSegment())) throw new IllegalArgumentException(
                            "materialization workspace is not accessible to every CPU worker");
        var valueScans = new ArrayList<Runnable>();
        bindCanonicalBooleanInputs(arguments, valueScans);
        attentionGeometry.filter(g -> g.mask().isPresent()).ifPresent(g -> {
            int mask = g.roleBoundaryPositions().get(3);
            valueScans.add(CpuAttentionMaskValidator.bind(arguments.get(mask), bindings.get(mask)));
        });
        if (lossGeometry.filter(geometry -> geometry.axis() >= 0
                && geometry.targetRank() == geometry.extents().length - 1).isPresent()) {
            // Index loss has two distinct read boundaries: logits then integral targets.
            // Bind now, but scan only after preceding units have produced those targets.
            valueScans.add(CpuLossInputValidator.bind(arguments.get(1), lossGeometry.orElseThrow()));
        }
        Runnable[] valueValidations = valueScans.toArray(Runnable[]::new);
        if (aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            MemorySegment aggregateScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && aggregateScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException("aggregate scratch must not overlap a buffer");
        }
        if (maskedReductionGeometry.filter(g -> g.outputCount() > 0).isPresent()) {
            MemorySegment maskedScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && maskedScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException(
                            "masked-reduction scratch must not overlap a buffer");
        }
        if (advancedReductionGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            MemorySegment advancedScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && advancedScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException(
                            "advanced-reduction scratch must not overlap a buffer");
        }
        if (trailingNormalizationGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent()) {
            MemorySegment normalizationScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && normalizationScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException(
                            "trailing-normalization scratch must not overlap a buffer");
        }
        if (batchNormTrainingGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent()) {
            MemorySegment trainingScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && trainingScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException("batch-normalization scratch must not overlap a buffer");
        }
        if (attentionGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.rowCount() > 0).isPresent()) {
            MemorySegment attentionScratch = scratch(workspaces);
            for (CpuBufferArgument argument : arguments)
                if (argument instanceof CpuBufferArgument.Segment segment
                        && attentionScratch.asOverlappingSlice(segment.segment()).isPresent())
                    throw new IllegalArgumentException("attention scratch must not overlap a buffer");
        }
        if (artifact.specialization().rowFusionEmission().isPresent()) {
            // The public full-body work domain is [0,1), independent of R. Its accepted
            // [0,0) invocation must not enter staging, worker dispatch or terminal publication.
            if (start == end) return new Invocation(state,validation,scatterValidation,valueValidations,null,null,null);
            return bindSelectedRow(state,buffers,arguments,bindings,scratch(workspaces),0);
        }
        long length = end - start;
        KernelCall prologue = randomGeometry.isPresent() ? callFor(artifact.entryPoint(), arguments,
                null, geometry(arguments, 0, 0, 0), 0, 0) : null;
        int chunkCount = length == 0 ? 0 : Math.min(selectedRangeCount,
                Math.toIntExact(1 + (length - 1) / minimumElementsPerWorker));
        if (chunkCount <= 1) {
            long[] geometry = geometry(arguments, start, end, 0);
            KernelCall call = length == 0 ? null : callFor(lossEntryPoint(geometry, start, end),
                    arguments, scratchArgument(workspaces, 0), geometry, start, end);
            return new Invocation(state, validation, scatterValidation, valueValidations,
                    prologue, call, null);
        }
        CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[chunkCount];
        long quotient = length / chunkCount;
        long remainder = length % chunkCount;
        long chunkStart = start;
        for (int index = 0; index < chunkCount; index++) {
            long chunkEnd = chunkStart + quotient + (index < remainder ? 1 : 0);
            long[] geometry = geometry(arguments, chunkStart, chunkEnd, index);
            KernelCall call = callFor(lossEntryPoint(geometry, chunkStart, chunkEnd), arguments,
                    scratchArgument(workspaces, index), geometry, chunkStart, chunkEnd);
            calls[index] = call::invoke;
            chunkStart = chunkEnd;
        }
        return new Invocation(state, validation, scatterValidation, valueValidations,
                prologue, null, calls);
    }

    /**
     * Resolves prepared access recipes once into fresh mutable invocation arrays. Each boundary
     * uses its own represented width; no F32 narrowing or four-byte control base is introduced.
     * A row region resolves every boundary independently and binds work count one, including
     * B=0. Every output must be disjoint from every other boundary before any call is bound.
     * @param state exact run owner with explicit admitted B
     * @param buffers compatible ordered typed representations, borrowed
     * @param workspaces exact declared scratch, including structural zero-byte scratch
     * @return fresh bound invocation with direct typed carriers, never null
     * @throws IllegalArgumentException if typed spans or writable overlap are invalid
     * @throws ArithmeticException if numeric geometry is unrepresentable
     */
    private BoundInvocation bindOrdinaryBatch(RunState state, BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        long batch = state.geometry().batchSize().orElseThrow();
        var resolved = ordinaryBatchBindings(batchRoute).stream()
                .map(recipe -> recipe.bind(batch)).toList();
        var arguments = new ArrayList<CpuBufferArgument>(buffers.length);
        for (int i = 0; i < buffers.length; i++) {
            var argument = preparedArgument((CpuBufferRepresentation) buffers[i], i);
            if (Math.multiplyExact(resolved.get(i).referencedElementSpan(), (long)artifact.specialization().boundaryDataTypes().get(i).byteWidth()) > argument.byteSize()) {
                throw new IllegalArgumentException("batch access exceeds selected usable span");
            }
            arguments.add(argument);
        }
        int firstWrite = batchRoute.specialization().rowMapReductionIr().map(r ->
                (int)r.arithmetic().values().stream().filter(v -> v.kind()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.INPUT).count())
                .orElse(arguments.size()-1);
        for (int output = firstWrite; output < arguments.size(); output++) for (int other = 0; other < output; other++)
            if (overlaps(arguments.get(other),resolved.get(other),arguments.get(output),resolved.get(output)))
                throw new IllegalArgumentException("batch writable outputs must not overlap any other boundary");
        MemorySegment scratch = workspaces.length == 0 ? null : ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
        if (scratch != null) for (var argument : arguments) if (scratch.asOverlappingSlice(selectedSegment(argument)).isPresent()) {
            throw new IllegalArgumentException("batch scratch must not overlap a buffer");
        }
        long length; long[] geometry;
        long[] bases = java.util.stream.IntStream.range(0,arguments.size()).mapToLong(i ->
                arguments.get(i).byteOffset() / artifact.specialization().boundaryDataTypes().get(i).byteWidth()).toArray();
        if (batchRoute.ordinaryBatchGeometry().isPresent()) {
            var recipe = batchRoute.ordinaryBatchGeometry().orElseThrow();
            var lowered = recipe.bind(batch);
            long expectedScratch = recipe.scratchBytes().byteSize(batch);
            if ((scratch == null ? 0 : scratch.byteSize()) != expectedScratch
                    || !lowered.boundaryDataTypes().equals(artifact.specialization().boundaryDataTypes())
                    || lowered.accessBindings().size() != arguments.size()) {
                throw new IllegalArgumentException("ordinary batch numeric geometry changed declarations");
            }
            geometry = packGeometry(lowered, artifact.specialization(), bases, batch);
            length = lowered.elementCount();
            var validations = new ArrayList<Runnable>();
            for (int index = 0; index < firstWrite; index++) {
                if (artifact.specialization().boundaryDataTypes().get(index)
                        == io.github.pho001.synaptik.model.datatype.DataType.BOOL) {
                    validations.add(CpuBooleanInputValidator.bind(arguments.get(index),
                            lowered.accessBindings().get(index),
                            "BOOL condition must use canonical bytes"));
                }
            }
            if (!recipe.resets().isEmpty()) {
                int[] cells = new int[recipe.resets().size()];
                long[] seeds = new long[cells.length];
                for (int index = 0; index < cells.length; index++) {
                    cells[index] = recipe.resets().get(index).cell();
                    seeds[index] = geometry[cells[index]];
                }
                long[] packed = geometry;
                validations.add(() -> {
                    for (int index = 0; index < cells.length; index++)
                        packed[cells[index]] = seeds[index];
                });
            }
            KernelCall call = length == 0 ? null
                    : callFor(artifact.entryPoint(), arguments, scratch, geometry, 0, length);
            return new Invocation(state, null, null, validations.toArray(Runnable[]::new),
                    null, call, null);
        }
        if (batchRoute.specialization().rowMapReductionIr().isPresent()) {
            if (artifact.specialization().rowFusionEmission().isPresent())
                return bindSelectedRow(state,buffers,arguments,resolved,scratch,batch);
            geometry = new long[1+bases.length]; geometry[0] = batch;
            System.arraycopy(bases,0,geometry,1,bases.length); length = 1;
        } else if (batchRoute.batchMatmul().isPresent()) {
            var matrix = batchRoute.batchMatmul().orElseThrow().bind(batch);
            geometry = matrix.pack(bases[0], bases[1], bases[2]); length = matrix.outputCount();
        } else if (batchRoute.batchAggregate().isPresent()) {
            var aggregate = batchRoute.batchAggregate().orElseThrow().bind(batch);
            geometry = aggregate.pack(bases); length = aggregate.outputCount();
        } else {
            length = resolved.getLast().elementCount();
            int rank = resolved.getFirst().plan().iterationRank(), count = resolved.size();
            geometry = new long[2 * rank + count + count * rank + 2 * count];
            for (int axis = 0; axis < rank; axis++) geometry[axis] = resolved.getFirst().extents().get(axis);
            for (int value = 0; value < count; value++) {
                var binding = resolved.get(value);
                geometry[2 * rank + value] = Math.addExact(bases[value], binding.startAddress());
                for (int axis = 0; axis < rank; axis++) geometry[2 * rank + count + value * rank + axis] = binding.effectiveStrides().get(axis);
                long inner = 1;
                for (int axis = rank - binding.plan().contiguousSuffix(); axis < rank; axis++) inner = Math.multiplyExact(inner, binding.extents().get(axis));
                geometry[2 * rank + count + count * rank + count + value] = inner;
            }
        }
        KernelCall call = length == 0 ? null : callFor(artifact.entryPoint(), arguments, scratch, geometry, 0, length);
        return new Invocation(state, null, null, new Runnable[0], null, call, null);
    }

    /**
     * Cold-binds one complete selected ordinary region. Numeric ranges are fixed from R/m/W,
     * transposes and scratch are run-owned, and every helper/call is resolved before execution.
     * Producer-sensitive accessibility is rechecked before staging or output writes. A coordinated
     * group lends one reservation through staging, worker join and ordered merge; compatibility
     * groups retain their existing uncoordinated behavior. No invocation chooses another artifact.
     * @param state sole RunState owner
     * @param buffers typed owners retained for access preflight
     * @param arguments direct admitted carriers in boundary order
     * @param resolved this invocation's concrete boundary geometry
     * @param scratch exact private declared workspace
     * @param batch admitted B for symbolic regions, otherwise ignored
     * @return fresh direct synchronous invocation
     * @throws IllegalArgumentException if scratch aliases storage or workers cannot access it
     * @throws ArithmeticException if numeric range/staging arithmetic is unrepresentable
     */
    private BoundInvocation bindSelectedRow(RunState state,BufferRepresentation[] buffers,
            List<CpuBufferArgument> arguments,List<CpuAccessPlan.Binding> resolved,MemorySegment scratch,long batch) {
        var spec = artifact.specialization(); var region = spec.rowMapReductionIr().orElseThrow();
        var emission = spec.rowFusionEmission().orElseThrow(); int width = spec.boundaryDataTypes().getFirst().byteWidth();
        int anchor = region.rowSites().getFirst(), rowAxis = region.rowAxes().get(anchor);
        long rows = spec.ordinaryBatch() ? batch : region.accesses().get(anchor).extentConstants().get(rowAxis);
        int count = rows == 0 ? 0 : Math.toIntExact(Math.min(selectedRangeCount,1+(rows-1)/minimumElementsPerWorker));
        long stageCells = 0, terminalCells = 0;
        var transposes = new ArrayList<>(emission.invariantTransposeValues()); transposes.addAll(emission.temporalTransposeValues());
        for (int v : transposes) {
            var a = region.accesses().get(v); long cells = 1;
            for (long extent : a.extentConstants()) cells = Math.multiplyExact(cells,extent);
            stageCells = Math.addExact(stageCells,cells);
        }
        for (int v : region.terminalOutputs()) {
            long cells = 1; for (long extent : region.accesses().get(v).extentConstants()) cells = Math.multiplyExact(cells,extent);
            terminalCells = Math.addExact(terminalCells,cells);
        }
        long stageBytes = (stageCells*width+7)/8*8, slice = 8+(terminalCells*width+7)/8*8;
        if (scratch == null || scratch.byteSize() != Math.addExact(stageBytes,Math.multiplyExact(selectedRangeCount,slice))
                || scratch.address()%8 != 0) throw new IllegalArgumentException("row workspace layout disagrees");
        var owners = java.util.Arrays.copyOf(buffers,buffers.length,CpuBufferRepresentation[].class);
        var segments = arguments.stream().map(CpuPreparedExecutable::selectedSegment).toList();
        for (var segment : segments) if (segment.byteSize() != 0 && scratch.asOverlappingSlice(segment).isPresent())
            throw new IllegalArgumentException("row workspace overlaps selected capacity");
        Runnable preflight = () -> {
            for (var owner : owners) if (!owner.isAccessible()) throw new IllegalArgumentException("row carrier is no longer accessible");
            if (!scratch.scope().isAlive() || !scratch.isAccessibleBy(Thread.currentThread()))
                throw new IllegalArgumentException("row workspace is no longer accessible");
            if (count >= 2) {
                if (workerGroup == null || !workerGroup.workersCanAccess(scratch))
                    throw new IllegalArgumentException("row workspace is not worker-accessible");
                for (var segment : segments) if (!workerGroup.workersCanAccess(segment))
                    throw new IllegalArgumentException("row carrier is not worker-accessible");
            }
        };
        preflight.run();
        int[] boundary = new int[region.arithmetic().values().size()]; java.util.Arrays.fill(boundary,-1);
        int next = 0;
        for (var v : region.arithmetic().values()) if (v.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.INPUT) boundary[v.ordinal()] = next++;
        for (var v : region.arithmetic().values()) if (v.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.OUTPUT) boundary[v.ordinal()] = next++;
        var stagedBindings = transposes.stream().map(v -> region.accesses().get(v).bind(batch)).toList();
        Runnable stage = () -> {
            if (rows == 0) return;
            long destination = 0;
            for (int t = 0; t < transposes.size(); t++) {
                int v = transposes.get(t); var a = stagedBindings.get(t);
                var source = segments.get(boundary[region.viewOrigins().get(v)]);
                for (long k = 0; k < a.extents().get(0); k++) for (long n = 0; n < a.extents().get(1); n++) {
                    long offset = Math.addExact(a.baseElementOffset(),Math.addExact(k*a.effectiveStrides().get(0),n*a.effectiveStrides().get(1)));
                    MemorySegment.copy(source,offset*width,scratch,destination,width); destination += width;
                }
            }
        };
        long[] packed = new long[arguments.size()+3]; packed[0] = batch;
        for (int b = 0; b < arguments.size(); b++) packed[b+1] = arguments.get(b).byteOffset()/width;
        packed[arguments.size()+1] = count;
        KernelCall inline = count < 2 ? callFor(artifact.entryPoint(),arguments,scratch,packed,0,1) : null;
        KernelCall merge = count < 2 ? null : callFor(artifact.ordinaryRowMergeEntryPoint(),arguments,scratch,packed,0,1);
        CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[count < 2 ? 0 : count];
        long begin = 0;
        for (int i = 0; i < calls.length; i++) {
            long end = begin+rows/count+(i < rows%count ? 1 : 0);
            long[] frame = packed.clone(); frame[arguments.size()+2] = stageBytes+i*slice;
            KernelCall range = callFor(artifact.ordinaryRowRangeEntryPoint(),arguments,scratch,frame,begin,end);
            calls[i] = range::invoke; begin = end;
        }
        KernelCall whole = () -> {
            var groupBudget = workerGroup == null ? null : workerGroup.concurrencyBudget();
            var reservation = groupBudget == null ? null : groupBudget.acquire(Math.max(1,count));
            try {
                preflight.run(); stage.run();
                if (inline != null) inline.invoke();
                else {
                    if (reservation == null) workerGroup.execute(calls); else workerGroup.execute(calls,reservation);
                    merge.invoke();
                }
            } finally { if (reservation != null) reservation.close(); }
        };
        return new Invocation(state,null,null,new Runnable[0],null,whole,null);
    }

    /** Binds the already-selected direct-array partial route without reselecting a kernel. */
    private BoundInvocation bindPartialReduction(RunState state, BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces,
            CpuGeneratedKernel.PartialReductionArtifact partial) {
        if (buffers.length != 2 || workspaces.length != 1 || workerGroup == null
                || !workerGroup.isOpen() || workerGroup.workerCount() < partial.ir().partialCount()) {
            throw new IllegalArgumentException("partial-reduction binding facts disagree");
        }
        CpuBufferArgument input = ((CpuBufferRepresentation) buffers[0]).argument();
        CpuBufferArgument output = ((CpuBufferRepresentation) buffers[1]).argument();
        MemorySegment workspace = ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
        if (!workspace.scope().isAlive() || !workspace.isAccessibleBy(Thread.currentThread())
                || workspace.byteSize() != partial.ir().workspaceBytes()
                || workspace.address() % CpuPartialReductionIr.STATE_SLICE_BYTES != 0
                || !workerGroup.workersCanAccess(workspace)) {
            throw new IllegalArgumentException("partial-reduction workspace is incompatible");
        }
        if (partial.ir().dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32
                && input instanceof CpuBufferArgument.Ints values
                && output instanceof CpuBufferArgument.Ints result && !result.readOnly()) {
            int inputBase = arrayBase(values.byteOffset(), Integer.BYTES, values.byteSize(),
                    values.carrier().length, partial.ir().outputCount(), partial.ir().domainCount());
            int outputBase = arrayBase(result.byteOffset(), Integer.BYTES, result.byteSize(),
                    result.carrier().length, partial.ir().outputCount(), 1);
            rejectPartialWorkspaceOverlap(workspace, MemorySegment.ofArray(values.carrier()),
                    MemorySegment.ofArray(result.carrier()));
            KernelCall call = () -> CpuPartialReductionExecution.executeInt(partial,
                    values.carrier(), inputBase, result.carrier(), outputBase, workspace, workerGroup);
            return new Invocation(state, null, null, new Runnable[0], null, call, null);
        }
        if (partial.ir().dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT64
                && input instanceof CpuBufferArgument.Longs values
                && output instanceof CpuBufferArgument.Longs result && !result.readOnly()) {
            int inputBase = arrayBase(values.byteOffset(), Long.BYTES, values.byteSize(),
                    values.carrier().length, partial.ir().outputCount(), partial.ir().domainCount());
            int outputBase = arrayBase(result.byteOffset(), Long.BYTES, result.byteSize(),
                    result.carrier().length, partial.ir().outputCount(), 1);
            rejectPartialWorkspaceOverlap(workspace, MemorySegment.ofArray(values.carrier()),
                    MemorySegment.ofArray(result.carrier()));
            KernelCall call = () -> CpuPartialReductionExecution.executeLong(partial,
                    values.carrier(), inputBase, result.carrier(), outputBase, workspace, workerGroup);
            return new Invocation(state, null, null, new Runnable[0], null, call, null);
        }
        throw new IllegalArgumentException("partial-reduction requires direct matching primitive arrays");
    }

    private static int arrayBase(long byteOffset, int width, long byteSize, int arrayLength,
            long outputCount, long elementsPerOutput) {
        if (byteOffset % width != 0 || byteSize % width != 0) {
            throw new IllegalArgumentException("partial-reduction array offset is not aligned");
        }
        long base = byteOffset / width;
        long required = Math.multiplyExact(outputCount, elementsPerOutput);
        if (base > Integer.MAX_VALUE || required > Integer.MAX_VALUE
                || required > byteSize / width
                || base + required > arrayLength || base + required > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("partial-reduction array span exceeds direct ABI");
        }
        return (int) base;
    }

    private static void rejectPartialWorkspaceOverlap(MemorySegment workspace,
            MemorySegment input, MemorySegment output) {
        if (workspace.asOverlappingSlice(input).isPresent()
                || workspace.asOverlappingSlice(output).isPresent()) {
            throw new IllegalArgumentException("partial-reduction workspace must not overlap a buffer");
        }
    }

    /**
     * Resolves a loss helper once after binding has packed and validated its cold geometry.
     *
     * <p>Only loss artifacts contain these private members.  All other families retain their one
     * public entry unchanged.  The proof deliberately mirrors the generated entry guard, so a
     * failed or incomplete cold proof binds the public entry and its generic affine fallback;
     * neither lookup nor selection is reachable from a generated loop.</p>
     *
     * @param geometry non-null packed invocation geometry
     * @param rangeStart inclusive range supplied to the generated entry
     * @param rangeEnd exclusive range supplied to the generated entry
     * @return the public entry for non-loss or unproved geometry, otherwise the exact private
     *     helper selected from the retained hidden-class lookup
     * @throws IllegalArgumentException if a generated loss helper cannot be resolved
     */
    private MethodHandle lossEntryPoint(long[] geometry, long rangeStart, long rangeEnd) {
        if (lossGeometry.isEmpty()) return artifact.entryPoint();
        return artifact.lossEntryPointFor(contiguousLossGeometry(geometry, rangeStart, rangeEnd));
    }

    /**
     * Reports whether one bound loss invocation may select the dedicated contiguous int-address
     * body directly.
     *
     * <p>The check is cold binding-time proof only. It requires all packed geometry and the
     * supplied half-open range to fit the direct body's {@code int} address domain, and rejects
     * any negative, strided, or broadcast address fact. A {@code false} result retains the public
     * entry, which dispatches to the generic affine body; it does not reject the invocation.</p>
     *
     * @param geometry non-null packed loss geometry in the generated entry's documented layout
     * @param start inclusive logical range bound supplied to the generated entry
     * @param end exclusive logical range bound supplied to the generated entry; must be no less
     *     than {@code start} to qualify
     * @return {@code true} only when the geometry and range prove the direct contiguous
     *     int-address form; otherwise {@code false}
     */
    static boolean contiguousLossGeometry(long[] geometry, long start, long end) {
        if (geometry.length < 10 || start < 0 || end < start || start > Integer.MAX_VALUE
                || end > Integer.MAX_VALUE) return false;
        for (int slot = 2; slot < geometry.length; slot++) {
            if (slot == 7 || slot == 8) continue;
            if (geometry[slot] < 0 || geometry[slot] > Integer.MAX_VALUE) return false;
        }
        int rank = (int) geometry[0];
        if (rank < 0 || geometry.length != 10 + rank + rank + geometry[2] + geometry[3]) {
            return false;
        }
        return geometry[1] < 0 ? contiguousMseGeometry(geometry, rank)
                : contiguousCategoricalGeometry(geometry, rank, start, end);
    }

    private static boolean contiguousMseGeometry(long[] geometry, int rank) {
        int targetRank = (int) geometry[2], outputRank = (int) geometry[3];
        if (targetRank != rank || (outputRank != rank && outputRank != 0)) return false;
        long expected = 1L;
        for (int axis = rank - 1; axis >= 0; axis--) {
            if (predictionStride(geometry, rank, axis) != expected
                    || targetStride(geometry, rank, targetRank, axis) != expected
                    || outputRank != 0 && outputStride(geometry, rank, targetRank, axis) != expected) {
                return false;
            }
            expected *= geometry[10 + axis];
        }
        return basePlusCountFitsInt(geometry[4], expected)
                && basePlusCountFitsInt(geometry[5], expected)
                && (outputRank == 0 || basePlusCountFitsInt(geometry[6], expected));
    }

    private static boolean contiguousCategoricalGeometry(long[] geometry, int rank, long start,
            long end) {
        int axis = (int) geometry[1], targetRank = (int) geometry[2], outputRank = (int) geometry[3];
        if (axis < 0 || axis >= rank || (targetRank != rank && targetRank != rank - 1)) return false;
        boolean noneRange = start == 0 && end == geometry[9];
        if (noneRange && outputRank != rank - 1) return false;
        long expectedLogits = 1L, expectedSample = 1L;
        for (int coordinate = rank - 1; coordinate >= 0; coordinate--) {
            if (predictionStride(geometry, rank, coordinate) != expectedLogits) return false;
            if (coordinate != axis) {
                int targetCoordinate = targetRank == rank ? coordinate
                        : coordinate < axis ? coordinate : coordinate - 1;
                long targetExpected = targetRank == rank ? expectedLogits : expectedSample;
                if (targetStride(geometry, rank, targetRank, targetCoordinate) != targetExpected) return false;
                if (noneRange) {
                    int outputCoordinate = coordinate < axis ? coordinate : coordinate - 1;
                    if (outputStride(geometry, rank, targetRank, outputCoordinate) != expectedSample) return false;
                }
                expectedSample *= geometry[10 + coordinate];
            }
            expectedLogits *= geometry[10 + coordinate];
        }
        return basePlusCountFitsInt(geometry[4], expectedLogits)
                && basePlusCountFitsInt(geometry[5], targetRank == rank
                        ? expectedLogits : expectedSample)
                && (outputRank == 0 || basePlusCountFitsInt(geometry[6], expectedSample));
    }

    private static long predictionStride(long[] geometry, int rank, int axis) {
        return geometry[10 + rank + axis];
    }

    private static long targetStride(long[] geometry, int rank, int targetRank, int axis) {
        return geometry[10 + 2 * rank + axis];
    }

    private static long outputStride(long[] geometry, int rank, int targetRank, int axis) {
        return geometry[10 + 2 * rank + targetRank + axis];
    }

    private static boolean basePlusCountFitsInt(long base, long count) {
        return base + count <= Integer.MAX_VALUE;
    }

    /**
     * Advances a row-major index cursor using the active geometry rank, not scratch capacity.
     * The caller owns all arrays; only the active prefix of {@code coordinates} is mutated,
     * with no allocation. Extra scratch cells are untouched. Advancing the final coordinate
     * wraps the active cursor and address back to their initial values; rank zero is a no-op.
     *
     * @param address current element address, including binding and layout offsets
     * @param extents non-null borrowed active extents; each is positive for a nonempty scan
     * @param strides non-null borrowed element strides with the same rank as {@code extents}
     * @param coordinates non-null caller-owned mutable cursor, distinct from the geometry
     *                    arrays, with capacity at least the active rank and a valid active prefix
     * @return the next element address, or the initial address after complete wraparound
     * @throws ArithmeticException if an address transition overflows {@code long}; cursor
     *                             mutation is not rolled back on failure
     */
    private static long advanceAddress(long address, long[] extents, long[] strides,
            long[] coordinates) {
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            coordinates[axis]++;
            address = Math.addExact(address, strides[axis]);
            if (coordinates[axis] < extents[axis]) break;
            coordinates[axis] = 0;
            address = Math.subtractExact(address,
                    Math.multiplyExact(extents[axis], strides[axis]));
        }
        return address;
    }

    /**
     * Returns only the family's mutable packed cursor seeds for fixed temporal helper calls.
     * Pointwise and affine-copy packs are immutable after binding and therefore have no resets.
     * Family geometry owners retain every packed-cell index; callers emit these assignments
     * directly rather than interpreting the list during execution.
     * @param lowered independently qualified ordinary lowering with resolved numeric geometry
     * @return immutable original cursor assignments, including an empty list for immutable packs
     * @throws ArithmeticException if a family cursor seed calculation overflows
     */
    public static List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset>
            temporalResets(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering.LoweredPartition lowered) {
        if (lowered.movementGeometry().isPresent()) return lowered.movementGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.indexingGeometry().isPresent()) return lowered.indexingGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.scatterGeometry().isPresent()) return lowered.scatterGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.foldGeometry().isPresent()) return lowered.foldGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.orderingGeometry().isPresent()) return lowered.orderingGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.randomGeometry().isPresent()) return lowered.randomGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.scanGeometry().isPresent()) return lowered.scanGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.aggregateGeometry().isPresent()) return (lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr
                    || lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr
                    ? lowered.aggregateGeometry().orElseThrow().temporalRebindFusion(0, lowered.boundaryValues().size())
                    : lowered.aggregateGeometry().orElseThrow().temporalRebind(0)).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.argExtremaGeometry().isPresent()) return lowered.argExtremaGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.maskedReductionGeometry().isPresent()) return lowered.maskedReductionGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.advancedReductionGeometry().isPresent()) return lowered.advancedReductionGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.softmaxGeometry().isPresent()) return lowered.softmaxGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.trailingNormalizationGeometry().isPresent()) return lowered.trailingNormalizationGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.batchNormInferenceGeometry().isPresent()) return lowered.batchNormInferenceGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.batchNormTrainingGeometry().isPresent()) return lowered.batchNormTrainingGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.conv2dGeometry().isPresent()) return lowered.conv2dGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.conv3dGeometry().isPresent()) return lowered.conv3dGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.matmulGeometry().isPresent()) return lowered.matmulGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.pool2dGeometry().isPresent()) return lowered.pool2dGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.pool3dGeometry().isPresent()) return lowered.pool3dGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.attentionGeometry().isPresent()) return lowered.attentionGeometry().orElseThrow().temporalRebind(0).resets().stream()
                .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        if (lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr loss)
            return loss.geometry().temporalRebind(0).resets().stream()
                    .map(r -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(r.cell(), r.value())).toList();
        return List.of();
    }

    private long[] geometry(List<CpuBufferArgument> arguments, long rangeStart, long rangeEnd,
            int rangeIndex) {
        long[] bases = new long[arguments.size()];
        for (int b = 0; b < bases.length; b++) bases[b] = arguments.get(b).byteOffset()
                / artifact.specialization().boundaryDataTypes().get(b).byteWidth();
        return packGeometry(artifact.specialization(), bindings, affineCopy, affineAddressPairs,
                movementGeometry, indexingGeometry, scatterGeometry, foldGeometry,
                orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry,
                trailingNormalizationGeometry, batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry,
                conv3dGeometry, matmulGeometry, pool2dGeometry, pool3dGeometry,
                attentionGeometry, lossGeometry, outputCount, bases,
                rangeStart, rangeEnd, rangeIndex);
    }

    /**
     * Packs an ordinary helper through the same family geometry owner as standalone binding.
     * Offsets are already normalized independently by each boundary's represented element width.
     * The returned array belongs exclusively to the bound invocation.
     * @param lowered exact controlled ordinary lowering, resolved to invocation geometry;
     *     distinct affine writes retain source/destination/original-logical-ordinal triples
     * @param specialization independently qualified matching ordinary specialization
     * @param elementBases workspace/carrier-relative element bases in ordinary ABI order
     * @param batch actual named numeric batch coordinate, or zero for fixed recipes
     * @return fresh invocation-owned numeric pack
     * @throws IllegalArgumentException if boundary counts or a family pack disagree
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    public static long[] packGeometry(
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering.LoweredPartition lowered,
            io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization specialization,
            long[] elementBases, long batch) {
        if (elementBases.length != lowered.boundaryDataTypes().size())
            throw new IllegalArgumentException("ordinary geometry boundary count mismatch");
        if (lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr affine
                && affine.writeDomain() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES) {
            long[] triples = lowered.affineAddressPairs();
            if (triples.length != Math.multiplyExact(lowered.elementCount(), 3))
                throw new IllegalArgumentException("controlled affine writes require original logical ordinals");
            for (int cell = 0; cell < triples.length; cell += 3) {
                triples[cell] = Math.addExact(triples[cell], elementBases[0]);
                triples[cell + 1] = Math.addExact(triples[cell + 1], elementBases[1]);
            }
            return triples;
        }
        var loss = lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr l
                ? Optional.of(l.geometry()) : Optional.<CpuLossLowering.Geometry>empty();
        long[] packed = packGeometry(specialization, lowered.accessBindings(), lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr,
                lowered.affineAddressPairs(),
                lowered.movementGeometry(), lowered.indexingGeometry(), lowered.scatterGeometry(),
                lowered.foldGeometry(), lowered.orderingGeometry(), lowered.randomGeometry(),
                lowered.scanGeometry(), lowered.aggregateGeometry(), lowered.argExtremaGeometry(),
                lowered.maskedReductionGeometry(), lowered.advancedReductionGeometry(), lowered.softmaxGeometry(),
                lowered.trailingNormalizationGeometry(), lowered.batchNormInferenceGeometry(), lowered.batchNormTrainingGeometry(),
                lowered.conv2dGeometry(), lowered.conv3dGeometry(), lowered.matmulGeometry(),
                lowered.pool2dGeometry(), lowered.pool3dGeometry(), lowered.attentionGeometry(),
                loss, (int) lowered.kernelIr().values().stream().filter(v -> v.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.OUTPUT).count(), elementBases,
                0, lowered.elementCount(), 0);
        if (specialization.rowMapReductionIr().isPresent()) packed[0] = batch;
        return packed;
    }

    /**
     * Selects the existing ordinary family pack owner for both standalone and Scan binding.
     * This cold method never reads tensor values; returned mutable state is exclusively run-owned.
     * @param specialization exact generated family and boundary type facts
     * @param bindings normalized full-range boundary access plans
     * @param affineCopy whether the affine address-pair ABI is selected
     * @param affineAddressPairs original immutable alternating source/result addresses
     * @param movementGeometry optional independently qualified movement geometry; empty for other families
     * @param indexingGeometry optional independently qualified indexing geometry; empty for other families
     * @param scatterGeometry optional independently qualified scatter geometry; empty for other families
     * @param foldGeometry optional independently qualified fold geometry; empty for other families
     * @param orderingGeometry optional independently qualified ordering geometry; empty for other families
     * @param randomGeometry optional independently qualified random geometry; empty for other families
     * @param scanGeometry optional independently qualified scan geometry; empty for other families
     * @param aggregateGeometry optional independently qualified aggregate geometry; empty for other families
     * @param argExtremaGeometry optional independently qualified argExtrema geometry; empty for other families
     * @param maskedReductionGeometry optional independently qualified maskedReduction geometry; empty for other families
     * @param advancedReductionGeometry optional independently qualified advancedReduction geometry; empty for other families
     * @param softmaxGeometry optional independently qualified softmax geometry; empty for other families
     * @param trailingNormalizationGeometry optional independently qualified trailingNormalization geometry; empty for other families
     * @param batchNormInferenceGeometry optional independently qualified batchNormInference geometry; empty for other families
     * @param batchNormTrainingGeometry optional independently qualified batchNormTraining geometry; empty for other families
     * @param conv2dGeometry optional independently qualified conv2d geometry; empty for other families
     * @param conv3dGeometry optional independently qualified conv3d geometry; empty for other families
     * @param matmulGeometry optional independently qualified matmul geometry; empty for other families
     * @param pool2dGeometry optional independently qualified pool2d geometry; empty for other families
     * @param pool3dGeometry optional independently qualified pool3d geometry; empty for other families
     * @param attentionGeometry optional independently qualified attention geometry; empty for other families
     * @param lossGeometry optional independently qualified loss geometry; empty for other families
     * @param outputCount number of materialized output boundaries
     * @param carrierBases typed element bases in ordinary boundary order
     * @param rangeStart inclusive logical range start
     * @param rangeEnd exclusive logical range end
     * @param rangeIndex zero-based private scratch range index
     * @return fresh family-owned numeric pack with the requested range's initial cursor state
     * @throws IllegalArgumentException if the selected family rejects boundary or range geometry
     * @throws ArithmeticException if checked address, range or geometry calculations overflow
     */
    private static long[] packGeometry(
            io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization specialization,
            List<CpuAccessPlan.Binding> bindings,
            boolean affineCopy,
            long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<CpuMatmulLowering.Geometry> matmulGeometry,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
            Optional<CpuAttentionLowering.Geometry> attentionGeometry,
            Optional<CpuLossLowering.Geometry> lossGeometry,
            int outputCount,
            long[] carrierBases,
            long rangeStart,
            long rangeEnd,
            int rangeIndex) {
        if (specialization.rowMapReductionIr().isPresent()) {
            long[] packed = new long[1+carrierBases.length];
            for (int b = 0; b < carrierBases.length; b++) packed[1+b] = carrierBases[b];
            return packed;
        }
        if (attentionGeometry.isPresent()) {
            long[] bases = carrierBases;
            return attentionGeometry.orElseThrow().pack(bases);
        }
        if (pool3dGeometry.isPresent()) {
            return pool3dGeometry.orElseThrow().pack(carrierBases[0], carrierBases[carrierBases.length - 1]);
        }
        if (pool2dGeometry.isPresent()) {
            return pool2dGeometry.orElseThrow().pack(carrierBases[0], carrierBases[carrierBases.length - 1]);
        }
        if (matmulGeometry.isPresent()) {
            long[] bases = carrierBases;
            if(bases.length==4) {
                CpuAccessPlan.Binding bias=bindings.get(2);
                if(bias.extents().size()!=1||bias.effectiveStrides().size()!=1)
                    throw new IllegalArgumentException("MATMUL bias binding is not rank one");
                long biasBase=Math.addExact(bases[2],bias.baseElementOffset());
                return matmulGeometry.orElseThrow().pack(bases[0],bases[1],bases[3],biasBase,
                        bias.effectiveStrides().getFirst());
            }
            return matmulGeometry.orElseThrow().pack(bases[0], bases[1], bases[bases.length - 1]);
        }
        if (conv3dGeometry.isPresent()) {
            long[] bases = carrierBases;
            return conv3dGeometry.orElseThrow().pack(bases);
        }
        if (conv2dGeometry.isPresent()) {
            long[] bases = carrierBases;
            return conv2dGeometry.orElseThrow().pack(bases);
        }
        if (batchNormTrainingGeometry.isPresent()) {
            long[] bases = carrierBases;
            return batchNormTrainingGeometry.orElseThrow().pack(bases, 0);
        }
        if (batchNormInferenceGeometry.isPresent()) {
            long[] bases = carrierBases;
            return batchNormInferenceGeometry.orElseThrow().pack(bases);
        }
        if (trailingNormalizationGeometry.isPresent()) {
            long[] bases = carrierBases;
            return trailingNormalizationGeometry.orElseThrow().pack(bases);
        }
        if (softmaxGeometry.isPresent()) {
            long[] bases = carrierBases;
            return softmaxGeometry.orElseThrow().pack(bases);
        }
        if (lossGeometry.isPresent()) {
            int output = bindings.size() - outputCount;
            if (outputCount != 1 || bindings.size() < 2 || bindings.size() > 3) {
                throw new IllegalArgumentException("loss boundaries must have two inputs and one output");
            }
            long[] bases = {carrierBases[0], carrierBases.length == 2 ? carrierBases[0] : carrierBases[1],
                    carrierBases[output]};
            return lossGeometry.orElseThrow().pack(bases);
        }
        if (advancedReductionGeometry.isPresent()) {
            long[] bases = carrierBases;
            return advancedReductionGeometry.orElseThrow().pack(bases);
        }
        if (maskedReductionGeometry.isPresent()) {
            long[] bases = carrierBases;
            // The worker call already receives its private slice of the run-owned workspace.
            return maskedReductionGeometry.orElseThrow().pack(bases, 0);
        }
        if (argExtremaGeometry.isPresent()) {
            long[] bases = carrierBases;
            return argExtremaGeometry.orElseThrow().pack(bases);
        }
        if (aggregateGeometry.isPresent()) {
            boolean fused = specialization.aggregateFusionIr().isPresent()
                    || specialization.softmaxBackwardIr().isPresent();
            long[] bases = carrierBases;
            if (specialization.softmaxBackwardIr().isPresent()) {
                var physical = specialization.softmaxBackwardIr().orElseThrow();
                for (int i = 0; i < bases.length; i++) {
                    var binding = physical.boundaries().get(i);
                    long span = Math.max(binding.referencedElementSpan(),binding.baseElementOffset());
                    Math.toIntExact(Math.multiplyExact(Math.addExact(bases[i],span),
                            physical.aggregate().dataType().byteWidth()));
                }
            }
            return fused ? aggregateGeometry.orElseThrow().packFusion(bases)
                    : aggregateGeometry.orElseThrow().pack(bases, 0);
        }
        if (scanGeometry.isPresent()) {
            long[] bases = carrierBases;
            return scanGeometry.orElseThrow().pack(bases);
        }
        if (randomGeometry.isPresent()) {
            long[] bases = carrierBases;
            return randomGeometry.orElseThrow().pack(bases);
        }
        if (orderingGeometry.isPresent()) {
            long[] bases = carrierBases;
            return orderingGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd, rangeIndex);
        }
        if (foldGeometry.isPresent()) {
            long[] bases = carrierBases;
            return foldGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (scatterGeometry.isPresent()) {
            long[] bases = carrierBases;
            return scatterGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd, rangeIndex);
        }
        if (indexingGeometry.isPresent()) {
            long[] bases = carrierBases;
            return indexingGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (movementGeometry.isPresent()) {
            long[] bases = carrierBases;
            return movementGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (affineCopy) {
            long[] geometry = affineAddressPairs.clone();
            long sourceBase = carrierBases[0], resultBase = carrierBases[1];
            for (int index = 0; index < geometry.length; index += 2) {
                geometry[index] = Math.addExact(geometry[index], sourceBase);
                geometry[index + 1] = Math.addExact(geometry[index + 1], resultBase);
            }
            return geometry;
        }
        int rank = bindings.getFirst().plan().iterationRank();
        int boundaryCount = bindings.size();
        long[] geometry = new long[2 * rank + boundaryCount + boundaryCount * rank
                + 2 * boundaryCount];
        CpuAccessPlan.Binding first = ranged(bindings.getFirst(), rangeStart, rangeEnd);
        for (int axis = 0; axis < rank; axis++) {
            geometry[axis] = first.extents().get(axis);
            geometry[rank + axis] = first.startCoordinates().get(axis);
        }
        for (int value = 0; value < boundaryCount; value++) {
            CpuAccessPlan.Binding binding = ranged(bindings.get(value), rangeStart, rangeEnd);
            long carrierBase = carrierBases[value];
            geometry[2 * rank + value] = Math.addExact(carrierBase, binding.startAddress());
            for (int axis = 0; axis < rank; axis++) geometry[2 * rank + boundaryCount + value * rank + axis]
                    = binding.effectiveStrides().get(axis);
            long innerPosition = 0;
            long innerSize = 1;
            for (int axis = rank - binding.plan().contiguousSuffix(); axis < rank; axis++) {
                innerPosition = Math.addExact(Math.multiplyExact(innerPosition,
                        binding.extents().get(axis)), binding.startCoordinates().get(axis));
                innerSize = Math.multiplyExact(innerSize, binding.extents().get(axis));
            }
            geometry[2 * rank + boundaryCount + boundaryCount * rank + value] = innerPosition;
            geometry[2 * rank + boundaryCount + boundaryCount * rank + boundaryCount + value] = innerSize;
        }
        return geometry;
    }

    /**
     * Tests represented byte spans on the exact selected storage, regardless of the two prepared
     * access forms. The segment intersection establishes shared backing storage before addresses
     * (heap carrier-relative offsets or native addresses) are compared.
     *
     * @param left non-null first selected argument
     * @param leftBinding non-null first access plan with a checked selected-region span
     * @param right non-null second selected argument
     * @param rightBinding non-null second access plan with a checked selected-region span
     * @return whether both nonempty accessed byte spans intersect in the same storage
     */
    private static boolean overlaps(CpuBufferArgument left, CpuAccessPlan.Binding leftBinding,
            CpuBufferArgument right, CpuAccessPlan.Binding rightBinding) {
        if (leftBinding.elementCount() == 0 || rightBinding.elementCount() == 0) return false;
        MemorySegment leftSegment = selectedSegment(left);
        MemorySegment rightSegment = selectedSegment(right);
        if (leftSegment.asOverlappingSlice(rightSegment).isEmpty()) return false;
        int leftWidth = elementWidth(left);
        int rightWidth = elementWidth(right);
        long leftStart = Math.addExact(leftSegment.address(),
                Math.multiplyExact(leftBinding.accessedElementStart(), leftWidth));
        long leftEnd = Math.addExact(leftSegment.address(),
                Math.multiplyExact(leftBinding.accessedElementEnd(), leftWidth));
        long rightStart = Math.addExact(rightSegment.address(),
                Math.multiplyExact(rightBinding.accessedElementStart(), rightWidth));
        long rightEnd = Math.addExact(rightSegment.address(),
                Math.multiplyExact(rightBinding.accessedElementEnd(), rightWidth));
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    /**
     * Tests the prepared affine copy's exact accessed element addresses, so unused bytes in
     * overlapping selected slices do not cause a false rejection. Both boundaries have the same
     * element type and width in an affine copy.
     *
     * @param left non-null first selected argument
     * @param right non-null second selected argument
     * @param leftIndex first boundary index in the prepared affine address pairs
     * @param rightIndex second boundary index in the prepared affine address pairs
     * @return whether the selected range reads and writes any shared element address
     */
    private boolean affineOverlaps(CpuBufferArgument left, CpuBufferArgument right,
            int leftIndex, int rightIndex) {
        MemorySegment leftSegment = selectedSegment(left);
        MemorySegment rightSegment = selectedSegment(right);
        if (leftSegment.asOverlappingSlice(rightSegment).isEmpty()) return false;
        int width = artifact.specialization().boundaryDataTypes().get(leftIndex).byteWidth();
        long leftBase = leftSegment.address(), rightBase = rightSegment.address();
        var addresses = new java.util.HashSet<Long>();
        for (long logical = start; logical < end; logical++) {
            int pair = Math.toIntExact(Math.multiplyExact(logical, 2));
            long element = leftIndex == 0 ? affineAddressPairs[pair] : affineAddressPairs[pair + 1];
            addresses.add(Math.addExact(leftBase, Math.multiplyExact(element, width)));
        }
        for (long logical = start; logical < end; logical++) {
            int pair = Math.toIntExact(Math.multiplyExact(logical, 2));
            long element = rightIndex == 0 ? affineAddressPairs[pair] : affineAddressPairs[pair + 1];
            if (addresses.contains(Math.addExact(rightBase, Math.multiplyExact(element, width))))
                return true;
        }
        return false;
    }

    /**
     * Recreates only the bounded segment view of an array argument for cold overlap analysis;
     * the generated call retains its original direct array or segment carrier. Segment arguments
     * already retain their exact selected view, including read-only and native slices.
     *
     * @param argument non-null classified argument with a valid selected byte region
     * @return an exact, non-owning view of that selected region and its original backing storage
     */
    private static MemorySegment selectedSegment(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Floats value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Shorts value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Ints value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Longs value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Bytes value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }

    @FunctionalInterface private interface KernelCall { void invoke() throws Throwable; }

    private static KernelCall callFor(MethodHandle h, List<CpuBufferArgument> a,
            MemorySegment scratch, long[] g, long start, long end) {
        MethodHandle bound;
        if (h.type().parameterType(0) == MemorySegment[].class) {
            // Cold, run-owned non-owning views retain the same bases and lifecycle as the
            // classified arguments; invocation never rebuilds or copies this operand table.
            MemorySegment[] operands = new MemorySegment[a.size()];
            for (int i = 0; i < operands.length; i++) operands[i] = fullCarrierSegment(a.get(i));
            bound = MethodHandles.insertArguments(h, 0, (Object) operands);
        } else {
            Object[] carriers = a.stream().map(CpuPreparedExecutable::carrier).toArray();
            bound = MethodHandles.insertArguments(h, 0, carriers);
        }
        if (scratch != null) bound = MethodHandles.insertArguments(bound, 0, scratch);
        bound = MethodHandles.insertArguments(bound, 0, (Object) g);
        MethodHandle target = bound;
        return () -> invokeVoid(target, start, end);
    }

    private static MemorySegment fullCarrierSegment(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Floats value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Shorts value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Ints value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Longs value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Bytes value -> MemorySegment.ofArray(value.carrier());
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }

    private static MemorySegment scratch(WorkspaceRepresentation[] workspaces) {
        return workspaces.length == 0 ? null
                : ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
    }

    private MemorySegment scratch(WorkspaceRepresentation[] workspaces, int rangeIndex) {
        MemorySegment whole = scratch(workspaces);
        if (whole != null && aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            long bytes = aggregateGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        if (whole != null && maskedReductionGeometry.filter(g -> g.outputCount() > 0).isPresent()) {
            long bytes = maskedReductionGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        if (whole != null && advancedReductionGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            long bytes = advancedReductionGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        if (whole != null && trailingNormalizationGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent()) {
            long bytes = trailingNormalizationGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        if (whole != null && batchNormTrainingGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent()) {
            long bytes = batchNormTrainingGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        if (whole != null && attentionGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.rowCount() > 0).isPresent()) {
            long bytes = attentionGeometry.orElseThrow().scratchSliceBytes();
            return whole.asSlice(Math.multiplyExact((long) rangeIndex, bytes), bytes);
        }
        return whole;
    }

    private MemorySegment scratchArgument(WorkspaceRepresentation[] workspaces, int rangeIndex) {
        if (!artifact.specialization().scratchParameter()) return null;
        MemorySegment value = scratch(workspaces, rangeIndex);
        return value == null && attentionGeometry.isPresent() ? MemorySegment.NULL : value;
    }

    private static void invokeVoid(MethodHandle target, long start, long end) throws Throwable {
        target.invokeExact(start, end);
    }

    private static Object carrier(CpuBufferArgument argument) {
        if (argument instanceof CpuBufferArgument.Doubles value) return value.carrier();
        if (argument instanceof CpuBufferArgument.Floats value) return value.carrier();
        if (argument instanceof CpuBufferArgument.Shorts value) return value.carrier();
        if (argument instanceof CpuBufferArgument.Ints value) return value.carrier();
        if (argument instanceof CpuBufferArgument.Longs value) return value.carrier();
        if (argument instanceof CpuBufferArgument.Bytes value) return value.carrier();
        return ((CpuBufferArgument.Segment) argument).segment();
    }

    private static int establishedOutputCount(
            Optional<CpuBatchNormTrainingLowering.Geometry> training,
            Optional<CpuRandomLowering.Geometry> random,
            Optional<CpuOrderingLowering.Geometry> ordering) {
        if (training.isPresent()) return 5;
        if (random.filter(g -> g.family()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT)
                .isPresent()) return 3;
        return ordering.filter(g -> g.family()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr.Family.TOP_K)
                .isPresent() ? 2 : 1;
    }

    private static List<BufferAccess> accesses(int count) { return accesses(count, 1); }

    private static List<BufferAccess> accesses(int count, int outputCount) {
        if (count < 1 || outputCount < 1 || outputCount > count)
            throw new IllegalArgumentException("buffer access counts are invalid");
        var result = new ArrayList<BufferAccess>(count);
        for (int i = 0; i < count - outputCount; i++) result.add(BufferAccess.READ_ONLY);
        for (int i = 0; i < outputCount; i++) result.add(BufferAccess.WRITE_ONLY);
        return List.copyOf(result);
    }

    private static CarrierAccess carrierAccess(CpuBufferArgument argument) {
        if (argument instanceof CpuBufferArgument.Doubles) return CarrierAccess.DOUBLE_ARRAY;
        if (argument instanceof CpuBufferArgument.Floats) return CarrierAccess.FLOAT_ARRAY;
        if (argument instanceof CpuBufferArgument.Shorts) return CarrierAccess.SHORT_ARRAY;
        if (argument instanceof CpuBufferArgument.Ints) return CarrierAccess.INT_ARRAY;
        if (argument instanceof CpuBufferArgument.Longs) return CarrierAccess.LONG_ARRAY;
        if (argument instanceof CpuBufferArgument.Bytes) return CarrierAccess.BYTE_ARRAY;
        return CarrierAccess.MEMORY_SEGMENT;
    }

    /**
     * Selects the argument form already fixed by this executable's prepared boundary. A matching
     * heap array may satisfy a segment route through its exact selected segment, but a segment
     * cannot satisfy an explicitly specialized array route.
     *
     * @param representation non-null accessible CPU representation retaining the selected region
     * @param index zero-based boundary index in the prepared carrier pattern
     * @return non-null direct argument in the prepared form; never a copied region
     * @throws IllegalArgumentException if an observable heap carrier conflicts with its data type
     * @throws IllegalStateException if the representation is closed or inaccessible
     */
    private CpuBufferArgument preparedArgument(CpuBufferRepresentation representation, int index) {
        return carrierPattern.get(index) == CarrierAccess.MEMORY_SEGMENT
                ? representation.segmentArgument() : representation.argument();
    }

    /**
     * Returns physical carrier width without inferring a logical dtype from shared raw storage.
     * @param argument non-null already classified direct carrier or typed segment
     * @return positive represented element width in bytes
     */
    private static int elementWidth(CpuBufferArgument argument) {
        if (argument instanceof CpuBufferArgument.Doubles) return Double.BYTES;
        if (argument instanceof CpuBufferArgument.Floats) return Float.BYTES;
        if (argument instanceof CpuBufferArgument.Shorts) return Short.BYTES;
        if (argument instanceof CpuBufferArgument.Ints) return Integer.BYTES;
        if (argument instanceof CpuBufferArgument.Longs) return Long.BYTES;
        if (argument instanceof CpuBufferArgument.Bytes) return Byte.BYTES;
        return ((CpuBufferArgument.Segment) argument).dataType().byteWidth();
    }

    /**
     * Captures BOOL scans against final consumer carriers without observing unfinished values.
     *
     * @param arguments non-null resolved generated-boundary arguments
     * @param scans non-null cold mutable list receiving one scan per BOOL read boundary
     */
    private void bindCanonicalBooleanInputs(List<CpuBufferArgument> arguments, List<Runnable> scans) {
        int output = orderingGeometry.filter(g -> g.family()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr.Family.TOP_K)
                .isPresent() ? arguments.size() - 2 : randomGeometry.map(g -> g.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT
                    ? arguments.size() - 3 : 0).orElse(arguments.size() - 1);
        for (int index = 0; index < output; index++) {
            if (attentionGeometry.isPresent()) continue;
            if (artifact.specialization().boundaryDataTypes().get(index)
                    != io.github.pho001.synaptik.model.datatype.DataType.BOOL) continue;
            if (affineCopy) {
                scans.add(CpuBooleanInputValidator.bindAffine(arguments.get(index),
                        affineAddressPairs, start, end));
                continue;
            }
            CpuAccessPlan.Binding binding = movementGeometry.isPresent() || indexingGeometry.isPresent()
                    || scatterGeometry.isPresent() || aggregateGeometry.isPresent()
                    || maskedReductionGeometry.isPresent() || attentionGeometry.isPresent()
                    ? bindings.get(index) : ranged(bindings.get(index));
            scans.add(CpuBooleanInputValidator.bind(arguments.get(index), binding,
                    "BOOL condition must use canonical bytes"));
        }
    }

    private static long elementCount(long[] extents) {
        if (java.util.Arrays.stream(extents).anyMatch(extent -> extent == 0)) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    /** Bound allocation-free deterministic index validator executed before every write call. */
    private static final class IndexValidation {
        private final CpuBufferArgument indexArgument;
        private final io.github.pho001.synaptik.model.datatype.DataType indexType;
        private final io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family family;
        private final long[] extents;
        private final long[] strides;
        private final long base;
        private final long bound;
        private final int axis;
        private final int batch;
        private final int tuple;
        private final long[] coordinates;
        private final long indexCount;
        private final MemorySegment storage;

        IndexValidation(List<CpuBufferArgument> arguments, CpuIndexingLowering.Geometry geometry) {
            family = geometry.family();
            int indexBoundary = family
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.ONE_HOT
                    ? geometry.occurrenceToBoundary().getFirst()
                    : geometry.occurrenceToBoundary().get(1);
            indexArgument = arguments.get(indexBoundary);
            indexType = geometry.boundaryTypes().get(indexBoundary);
            storage = indexStorage(indexArgument);
            var indexLayout = geometry.boundaries().get(indexBoundary);
            extents = indexLayout.extents(); strides = indexLayout.strides();
            base = Math.addExact(indexArgument.byteOffset() / indexType.byteWidth(), indexLayout.offset());
            coordinates = new long[extents.length];
            indexCount = elementCount(extents);
            if (geometry.variant() instanceof CpuIndexingLowering.Geometry.Axis value) {
                axis = value.axis(); batch = 0; tuple = 0;
                int dataBoundary = geometry.occurrenceToBoundary().getFirst();
                bound = geometry.boundaries().get(dataBoundary).extents()[axis];
            } else if (geometry.variant() instanceof CpuIndexingLowering.Geometry.Nd value) {
                axis = -1; batch = value.batchDimensions(); tuple = value.tupleDepth(); bound = -1;
            } else {
                axis = -1; batch = 0; tuple = 0;
                bound = ((CpuIndexingLowering.Geometry.Hot) geometry.variant()).depth();
            }
            dataExtents = family
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.GATHER_ND
                    ? geometry.boundaries().get(geometry.occurrenceToBoundary().getFirst()).extents()
                    : new long[0];
        }

        private final long[] dataExtents;

        void validate() {
            validateTemporalIndices(storage, indexType.byteWidth(), base, extents, strides,
                    coordinates, 0L, indexCount, family, axis, batch, tuple, bound, dataExtents,
                    0L, true, null, 0L, 0L, 0L);
        }

        private static IndexOutOfBoundsException failure(
                io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family family,
                long ordinal, long value, int selectedAxis, long extent) {
            String message = switch (family) {
                case GATHER -> "GATHER index at logical position " + ordinal
                        + " for data axis " + selectedAxis + " is out of bounds: value=" + value
                        + ", extent=" + extent;
                case GATHER_ELEMENTS -> "GATHER_ELEMENTS index at logical position " + ordinal
                        + " for data axis " + selectedAxis + " is out of bounds: value=" + value
                        + ", extent=" + extent;
                case GATHER_ND -> "GATHER_ND index at logical position " + ordinal
                        + " for data axis " + selectedAxis + " is out of bounds: value=" + value
                        + ", extent=" + extent;
                case ONE_HOT -> "ONE_HOT index at logical position " + ordinal
                        + " is out of bounds: value=" + value + ", depth=" + extent;
            };
            return new IndexOutOfBoundsException(message);
        }
    }

    private static MemorySegment indexStorage(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Ints ints -> MemorySegment.ofArray(ints.carrier());
            case CpuBufferArgument.Longs longs -> MemorySegment.ofArray(longs.carrier());
            case CpuBufferArgument.Segment segment -> segment.segment();
            default -> throw new IllegalArgumentException("index requires integral storage");
        };
    }

    private static long readIndex(MemorySegment storage, long address, int indexBytes) {
        return indexBytes == Integer.BYTES
                ? storage.get(ValueLayout.JAVA_INT, Math.multiplyExact(address, Integer.BYTES))
                : storage.get(ValueLayout.JAVA_LONG, Math.multiplyExact(address, Long.BYTES));
    }

    /**
     * Checks the complete eligible index range before any output, history or scratch write.
     * The domain is the index domain, even if the numerical output is empty. GATHER_ND bounds
     * are component-specific. Cold wrappers and the mutable cursor are retained by the invocation.
     *
     * @param storage borrowed prebound integral storage
     * @param indexBytes represented width, four or eight bytes
     * @param base index layout element offset, including any carrier offset
     * @param extents immutable cold index extents
     * @param strides immutable cold index element strides
     * @param coordinates invocation-private rank-sized cursor, reset on every call
     * @param start inclusive scalar index ordinal
     * @param end exclusive ordinal for the entire unit validation domain
     * @param family cold ordinary indexing family
     * @param axis data axis for axis indexing, otherwise minus one
     * @param batch GATHER_ND batch dimensions, otherwise zero
     * @param tuple GATHER_ND tuple depth, otherwise zero
     * @param bound axis extent or ONE_HOT depth, unused for GATHER_ND
     * @param dataExtents immutable data extents for GATHER_ND, otherwise an empty array
     * @param temporalOffset additional index element displacement
     * @param ownerPresent structural occurrence presence, independent of empty domains
     * @param activity borrowed row bytes, or null for all-active
     * @param activityOffset byte address of row zero
     * @param rowStride scalar index ordinals per original row coordinate, or zero for a shared byte
     * @param rowExtent proved row extent, unused for a shared byte
     * @throws IndexOutOfBoundsException if an eligible index is outside its component bounds
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateTemporalIndices(MemorySegment storage, int indexBytes, long base,
            long[] extents, long[] strides, long[] coordinates, long start, long end,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family family,
            int axis, int batch, int tuple, long bound, long[] dataExtents, long temporalOffset,
            boolean ownerPresent, MemorySegment activity, long activityOffset,
            long rowStride, long rowExtent) {
        if (!ownerPresent) return;
        ScatterValidation.coordinates(start, extents, coordinates);
        long address = Math.addExact(base, temporalOffset);
        for (int dimension = 0; dimension < extents.length; dimension++)
            address = Math.addExact(address,
                    Math.multiplyExact(coordinates[dimension], strides[dimension]));
        for (long ordinal = start; ordinal < end; ordinal++) {
            if (CpuBooleanInputValidator.isActive(ordinal, activity, activityOffset,
                    rowStride, rowExtent)) {
                long value = readIndex(storage, address, indexBytes);
                int selectedAxis = axis;
                long selectedBound = bound;
                if (family == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.GATHER_ND) {
                    selectedAxis = batch + (int) (ordinal % tuple);
                    selectedBound = dataExtents[selectedAxis];
                }
                if (value < 0 || value >= selectedBound)
                    throw IndexValidation.failure(family, ordinal, value, selectedAxis, selectedBound);
            }
            address = advanceAddress(address, extents, strides, coordinates);
        }
    }

    /** Reusable allocation-free complete scatter bounds and replacement-uniqueness validator. */
    private static final class ScatterValidation {
        private final CpuScatterLowering.Geometry geometry;
        private final io.github.pho001.synaptik.model.datatype.DataType indexType;
        private final long[] indexExtents;
        private final long[] indexStrides;
        private final long[] tuplePrefixExtents;
        private final long indexBase;
        private final long[] current;
        private final long[] earlier;
        private final MemorySegment storage;
        private final long[] dataExtents;
        private final long[] updateExtents;
        private final long indexCount;
        private final long candidateCount;

        ScatterValidation(List<CpuBufferArgument> arguments,
                CpuScatterLowering.Geometry geometry) {
            this.geometry = geometry;
            int indexBoundary = geometry.occurrenceToBoundary().get(1);
            CpuBufferArgument indices = arguments.get(indexBoundary);
            indexType = geometry.boundaryTypes().get(indexBoundary);
            storage = indexStorage(indices);
            var layout = geometry.boundaries().get(indexBoundary);
            indexExtents = layout.extents(); indexStrides = layout.strides();
            tuplePrefixExtents = java.util.Arrays.copyOf(indexExtents,
                    Math.max(0, indexExtents.length - 1));
            indexBase = Math.addExact(indices.byteOffset() / indexType.byteWidth(), layout.offset());
            current = new long[Math.max(indexExtents.length,
                    geometry.boundaries().get(geometry.occurrenceToBoundary().get(2)).extents().length)];
            earlier = current.clone();
            dataExtents = geometry.boundaries().get(geometry.occurrenceToBoundary().get(0)).extents();
            updateExtents = geometry.boundaries().get(geometry.occurrenceToBoundary().get(2)).extents();
            indexCount = elementCount(indexExtents);
            candidateCount = geometry.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ND
                    ? elementCount(tuplePrefixExtents) : elementCount(updateExtents);
        }

        void validate() {
            validateTemporalScatter(storage, indexType.byteWidth(), indexBase, indexExtents,
                    indexStrides, dataExtents, updateExtents, tuplePrefixExtents, current, earlier,
                    indexCount, candidateCount, geometry.family(), geometry.axis(),
                    geometry.batchDimensions(), geometry.tupleDepth(),
                    geometry.reduction() == io.github.pho001.synaptik.model.operation.index.ScatterReduction.NONE,
                    0L, true, null, 0L, 0L, 0L, 0L, 0L, 0L);
        }
        private static void coordinates(long ordinal,long[] extents,long[] target){
            java.util.Arrays.fill(target,0);
            for(int i=extents.length-1;i>=0;i--){if(extents[i]!=0){target[i]=ordinal%extents[i];ordinal/=extents[i];}}
        }
    }

    /**
     * Validates all eligible scatter bounds, then replacement uniqueness, before the unit's
     * initial copy or any update/scratch write. Inactive earlier and later candidates participate
     * in neither phase. This is also the ordinary scatter validator; it does not restrict checks
     * to the output range or allocate state during execution.
     *
     * @param storage borrowed prebound integral index storage
     * @param indexBytes represented index width, four or eight bytes
     * @param base index layout element offset including any carrier offset
     * @param indexExtents immutable cold scalar-index extents
     * @param indexStrides immutable cold index element strides
     * @param dataExtents immutable cold destination extents
     * @param updateExtents immutable cold update extents
     * @param tuplePrefixExtents immutable ND index extents excluding the component axis
     * @param current invocation-private cursor sized for both index and update ranks
     * @param earlier separate invocation-private cursor with the same capacity
     * @param indexCount complete scalar-index count, including when output is empty
     * @param candidateCount complete replacement-update count, or ND tuple count
     * @param family cold ordinary scatter family
     * @param axis selected destination axis for axis scatter
     * @param batch ND batch-dimension count
     * @param tuple ND tuple depth
     * @param replacement whether ordinary NONE reduction requires uniqueness checks
     * @param temporalOffset additional index element displacement for this step
     * @param ownerPresent structural occurrence presence, independent of empty domains
     * @param activity borrowed row bytes, or null for all-active
     * @param activityOffset byte address of row zero for the scalar-index domain
     * @param indexRowStride scalar-index ordinals per original row coordinate, or zero for a shared byte
     * @param indexRowExtent proved row extent in the scalar-index domain
     * @param candidateActivityOffset independently bound byte address of row zero for replacement candidates
     * @param candidateRowStride update/tuple ordinals per original row coordinate, or zero for a shared byte
     * @param candidateRowExtent proved row extent in the replacement-candidate domain
     * @throws IndexOutOfBoundsException if any eligible index is out of bounds
     * @throws IllegalArgumentException if eligible replacement candidates address the same destination
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateTemporalScatter(MemorySegment storage, int indexBytes, long base,
            long[] indexExtents, long[] indexStrides, long[] dataExtents, long[] updateExtents,
            long[] tuplePrefixExtents, long[] current, long[] earlier,
            long indexCount, long candidateCount,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family family,
            int axis, int batch, int tuple, boolean replacement, long temporalOffset,
            boolean ownerPresent, MemorySegment activity, long activityOffset,
            long indexRowStride, long indexRowExtent,
            long candidateActivityOffset, long candidateRowStride, long candidateRowExtent) {
        if (!ownerPresent) return;
        long origin = Math.addExact(base, temporalOffset);
        java.util.Arrays.fill(current, 0L);
        long address = origin;
        boolean nd = family
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ND;
        for (long ordinal = 0; ordinal < indexCount; ordinal++) {
            if (CpuBooleanInputValidator.isActive(ordinal, activity, activityOffset,
                    indexRowStride, indexRowExtent)) {
                long value = readIndex(storage, address, indexBytes);
                int selectedAxis = nd ? batch + (int) (ordinal % tuple) : axis;
                long extent = dataExtents[selectedAxis];
                if (value < 0 || value >= extent)
                    throw new IndexOutOfBoundsException(family.name() + " index at logical position "
                            + ordinal + " for data axis " + selectedAxis
                            + " is out of bounds: value=" + value + ", extent=" + extent);
            }
            address = advanceAddress(address, indexExtents, indexStrides, current);
        }
        if (!replacement || family
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ADD) return;
        long[] candidateExtents = nd ? tuplePrefixExtents : updateExtents;
        for (long later = 1; later < candidateCount; later++) {
            if (!CpuBooleanInputValidator.isActive(later, activity, candidateActivityOffset,
                    candidateRowStride, candidateRowExtent)) continue;
            ScatterValidation.coordinates(later, candidateExtents, current);
            for (long first = 0; first < later; first++) {
                if (!CpuBooleanInputValidator.isActive(first, activity, candidateActivityOffset,
                        candidateRowStride, candidateRowExtent)) continue;
                ScatterValidation.coordinates(first, candidateExtents, earlier);
                if (scatterTargetEqual(storage, indexBytes, origin, indexExtents, indexStrides,
                        current, earlier, nd, axis, batch, tuple)) {
                    if (nd) throw new IllegalArgumentException(
                            "SCATTER_ND duplicate target tuple at logical tuple position " + later
                            + "; first addressed at logical tuple position " + first);
                    throw new IllegalArgumentException(
                            "SCATTER_ELEMENTS duplicate target at logical update position " + later
                            + "; first addressed at logical update position " + first);
                }
            }
        }
    }

    private static boolean scatterTargetEqual(MemorySegment storage, int indexBytes, long base,
            long[] extents, long[] strides, long[] left, long[] right,
            boolean nd, int axis, int batch, int tuple) {
        if (!nd) {
            for (int dimension = 0; dimension < extents.length; dimension++)
                if (dimension != axis && left[dimension] != right[dimension]) return false;
            return scatterIndexAt(storage, indexBytes, base, extents, strides, left, -1)
                    == scatterIndexAt(storage, indexBytes, base, extents, strides, right, -1);
        }
        for (int dimension = 0; dimension < batch; dimension++)
            if (left[dimension] != right[dimension]) return false;
        for (int component = 0; component < tuple; component++)
            if (scatterIndexAt(storage, indexBytes, base, extents, strides, left, component)
                    != scatterIndexAt(storage, indexBytes, base, extents, strides, right, component))
                return false;
        return true;
    }

    private static long scatterIndexAt(MemorySegment storage, int indexBytes, long base,
            long[] extents, long[] strides, long[] coordinate, int component) {
        long address = base;
        int prefix = extents.length - 1;
        for (int dimension = 0; dimension < (component < 0 ? extents.length : prefix); dimension++)
            address = Math.addExact(address, Math.multiplyExact(coordinate[dimension], strides[dimension]));
        if (component >= 0)
            address = Math.addExact(address, Math.multiplyExact(component, strides[prefix]));
        return readIndex(storage, address, indexBytes);
    }

    private final class Invocation extends BoundInvocation {
        private final IndexValidation validation;
        private final ScatterValidation scatterValidation;
        private final Runnable[] valueValidations;
        private final KernelCall prologue;
        private final KernelCall call;
        private final CpuWorkerGroup.RangeCall[] calls;
        Invocation(RunState state, IndexValidation validation,
                ScatterValidation scatterValidation, Runnable[] valueValidations,
                KernelCall prologue, KernelCall call,
                CpuWorkerGroup.RangeCall[] calls) {
            super(state); this.validation = validation;
            this.scatterValidation=scatterValidation; this.prologue=prologue;
            this.valueValidations = valueValidations;
            this.call = call; this.calls = calls;
        }
        @Override protected void executeBound() {
            CpuConcurrencyBudget.Lease lease = call != null && concurrencyBudget != null
                    ? concurrencyBudget.acquire(1) : null;
            try {
                for (Runnable valueValidation : valueValidations) valueValidation.run();
                if (validation != null) validation.validate();
                if (scatterValidation != null) scatterValidation.validate();
                if (prologue != null) prologue.invoke();
                if (call != null) call.invoke();
                else if (calls != null) workerGroup.execute(calls);
            }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("generated CPU invocation failed", failure); }
            finally { if (lease != null) lease.close(); }
        }
    }
}
