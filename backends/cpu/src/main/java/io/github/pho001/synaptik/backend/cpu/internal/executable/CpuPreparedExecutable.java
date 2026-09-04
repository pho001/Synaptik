package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
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
 * For an affine plan, cold binding additionally validates the exact two boundary carriers and
 * represented address spans, rejects source/result overlap, and retains the composed address
 * pairs directly. Runtime invokes only the prepared range and never interprets the view chain.
 * For an indexing plan, cold binding creates one direct typed scalar validator. Each execution
 * scans the complete logical index domain in deterministic row-major order before any generated
 * call or worker submission. A zero output still validates a non-empty index domain, while a
 * valid zero-output execution makes no generated call and submits no worker work.
 * For scatter, binding validates every index and then replacement uniqueness before any output
 * call. Successful ranges own disjoint output coordinates and, when floating multiplication is
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
 * For ordinary aggregate execution, binding validates complete input/output spans, every canonical
 * Boolean input, and any exact-state workspace size, alignment, accessibility, and physical
 * non-overlap before mutation. Floating numerical parallel ranges receive disjoint workspace
 * slices. All ranges own disjoint complete output cells; no range splits, partially reduces, or
 * combines a selected domain.
 * Masked SUM/MEAN binding additionally validates the complete canonical Boolean mask before any
 * output write or worker submission, permits only physical input/input aliasing, and rejects the
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
 * output, and complete output/input span before it scans index targets. The index scan compares
 * ignore before bounds and completes before it constructs a generated call or submits workers;
 * MSE and dense categorical loss require no target-value scan. Losses declare no workspace.
 * Scaled-dot-product-attention binding validates its optional canonical Boolean mask and all
 * input/output/workspace overlap before mutation. Each range owns complete broadcast-batch/query
 * rows and a disjoint aligned score slice; zero-score and zero-work forms retain the scratch-shaped
 * entry without requiring a workspace.
 */
public final class CpuPreparedExecutable extends PreparedExecutable {
    private final CpuGeneratedKernel artifact;
    private final List<CpuAccessPlan.Binding> bindings;
    private final List<CarrierAccess> carrierPattern;
    private final List<CarrierAccess> generatedCarrierPattern;
    private final long start;
    private final long end;
    private final int selectedRangeCount;
    private final long minimumElementsPerWorker;
    private final CpuWorkerGroup workerGroup;
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
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), outputCount);
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
            Optional<CpuLossLowering.Geometry> lossGeometry, int outputCount) {
        super(memoryPlan, selections, java.util.stream.Stream.concat(workspaceSelection.stream(),
                representationWorkspaceSelections.stream()).toList(),
                accesses(selections.size(), outputCount));
        if (selections.isEmpty()) throw new IllegalArgumentException("at least one buffer required");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
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
        long count = this.lossGeometry.isPresent() ? end : this.attentionGeometry.isPresent()
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
                : this.bindings.getFirst().elementCount();
        if (start < 0 || end < start || end > count) throw new IllegalArgumentException("invalid range");
        this.start = start;
        this.end = end;
        if (selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                || (selectedRangeCount >= 2) != (workerGroup != null)) {
            throw new IllegalArgumentException("parallel execution facts are inconsistent");
        }
        this.selectedRangeCount = selectedRangeCount;
        this.minimumElementsPerWorker = minimumElementsPerWorker;
        this.workerGroup = workerGroup;
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
                || this.orderingGeometry.isPresent())) {
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
                pool2dGeometry, pool3dGeometry, Optional.empty(), Optional.empty(), outputCount);
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
                pool2dGeometry, pool3dGeometry, attentionGeometry, lossGeometry, outputCount);
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
     * Returns the iteration boundary's geometry adjusted to this recipe's range.
     * For movement this is the output boundary; other routes retain the first boundary contract.
     *
     * @return a new immutable ranged binding
     */
    public CpuAccessPlan.Binding binding() {
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
                attentionGeometry, lossGeometry, outputCount);
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

    @Override protected boolean acceptsBufferRepresentation(int index, BufferRepresentation value) {
        if (!(value instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != artifact.specialization().boundaryDataTypes().get(index)) return false;
        var entry = memoryPlan().buffers().get(bufferSelection(index).bufferIndex());
        if (cpu.byteSize() != entry.byteSize()) return false;
        try {
            CpuBufferArgument argument = cpu.argument();
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
        boolean hasWorkspace=scatterGeometry.filter(g->g.scratchSliceBytes()>0).isPresent()
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
        for (BufferRepresentation buffer : buffers) {
            arguments.add(((CpuBufferRepresentation) buffer).argument());
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
                            || lossGeometry.isPresent()
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
        attentionGeometry.filter(g -> g.mask().isPresent()).ifPresent(g -> {
            int mask = g.roleBoundaryPositions().get(3);
            CpuAttentionMaskValidator.validate(arguments.get(mask), bindings.get(mask));
        });
        validateCanonicalBooleanInputs(arguments);
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
        if (softmaxGeometry.isPresent()) {
            CpuSoftmaxInputValidator.validate(arguments.getFirst(), softmaxGeometry.orElseThrow());
        }
        if (lossGeometry.filter(geometry -> geometry.axis() >= 0
                && geometry.targetRank() == geometry.extents().length - 1).isPresent()) {
            // Index loss has two distinct read boundaries: logits then integral targets. This
            // cold scan intentionally precedes call construction and all worker submission.
            CpuLossInputValidator.validate(arguments.get(1), lossGeometry.orElseThrow());
        }
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
        long length = end - start;
        KernelCall prologue = randomGeometry.isPresent() ? callFor(artifact.entryPoint(), arguments,
                null, geometry(arguments, 0, 0, 0), 0, 0) : null;
        int chunkCount = length == 0 ? 0 : Math.min(selectedRangeCount,
                Math.toIntExact(1 + (length - 1) / minimumElementsPerWorker));
        if (chunkCount <= 1) {
            long[] geometry = geometry(arguments, start, end, 0);
            KernelCall call = length == 0 ? null : callFor(lossEntryPoint(geometry, start, end),
                    arguments, scratchArgument(workspaces, 0), geometry, start, end);
            return new Invocation(state, validation, scatterValidation, prologue, call, null);
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
        return new Invocation(state, validation, scatterValidation, prologue, null, calls);
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

    private static long advanceAddress(long address, long[] extents, long[] strides,
            long[] coordinates) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            coordinates[axis]++;
            address = Math.addExact(address, strides[axis]);
            if (coordinates[axis] < extents[axis]) break;
            coordinates[axis] = 0;
            address = Math.subtractExact(address,
                    Math.multiplyExact(extents[axis], strides[axis]));
        }
        return address;
    }

    private long[] geometry(List<CpuBufferArgument> arguments, long rangeStart, long rangeEnd,
            int rangeIndex) {
        if (attentionGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return attentionGeometry.orElseThrow().pack(bases);
        }
        if (pool3dGeometry.isPresent()) {
            int width = artifact.specialization().boundaryDataTypes().getFirst().byteWidth();
            return pool3dGeometry.orElseThrow().pack(
                    arguments.getFirst().byteOffset() / width,
                    arguments.getLast().byteOffset() / width);
        }
        if (pool2dGeometry.isPresent()) {
            int width = artifact.specialization().boundaryDataTypes().getFirst().byteWidth();
            return pool2dGeometry.orElseThrow().pack(
                    arguments.getFirst().byteOffset() / width,
                    arguments.getLast().byteOffset() / width);
        }
        if (matmulGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
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
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return conv3dGeometry.orElseThrow().pack(bases);
        }
        if (conv2dGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return conv2dGeometry.orElseThrow().pack(bases);
        }
        if (batchNormTrainingGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return batchNormTrainingGeometry.orElseThrow().pack(bases, 0);
        }
        if (batchNormInferenceGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return batchNormInferenceGeometry.orElseThrow().pack(bases);
        }
        if (trailingNormalizationGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < bases.length; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return trailingNormalizationGeometry.orElseThrow().pack(bases);
        }
        if (softmaxGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return softmaxGeometry.orElseThrow().pack(bases);
        }
        if (lossGeometry.isPresent()) {
            int output = bindings.size() - outputCount;
            if (outputCount != 1 || bindings.size() < 2 || bindings.size() > 3) {
                throw new IllegalArgumentException("loss boundaries must have two inputs and one output");
            }
            long[] bases = new long[3];
            for (int index = 0; index < bindings.size() - 1; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            if (bindings.size() == 2) bases[1] = bases[0];
            int width = artifact.specialization().boundaryDataTypes().get(output).byteWidth();
            bases[2] = arguments.get(output).byteOffset() / width;
            return lossGeometry.orElseThrow().pack(bases);
        }
        if (advancedReductionGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return advancedReductionGeometry.orElseThrow().pack(bases);
        }
        if (maskedReductionGeometry.isPresent()) {
            long[] bases = new long[3];
            for (int index = 0; index < 3; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            // The worker call already receives its private slice of the run-owned workspace.
            return maskedReductionGeometry.orElseThrow().pack(bases, 0);
        }
        if (argExtremaGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return argExtremaGeometry.orElseThrow().pack(bases);
        }
        if (aggregateGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return aggregateGeometry.orElseThrow().pack(bases, 0);
        }
        if (scanGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return scanGeometry.orElseThrow().pack(bases);
        }
        if (randomGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < arguments.size(); index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return randomGeometry.orElseThrow().pack(bases);
        }
        if (orderingGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < arguments.size(); index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return orderingGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd, rangeIndex);
        }
        if (foldGeometry.isPresent()) {
            long[] bases = new long[2];
            for (int index = 0; index < 2; index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return foldGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (scatterGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < arguments.size(); index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return scatterGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd, rangeIndex);
        }
        if (indexingGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < arguments.size(); index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return indexingGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (movementGeometry.isPresent()) {
            long[] bases = new long[arguments.size()];
            for (int index = 0; index < arguments.size(); index++) {
                int width = artifact.specialization().boundaryDataTypes().get(index).byteWidth();
                bases[index] = arguments.get(index).byteOffset() / width;
            }
            return movementGeometry.orElseThrow().pack(bases, rangeStart, rangeEnd);
        }
        if (affineCopy) {
            long[] geometry = affineAddressPairs.clone();
            int sourceWidth = dataType(arguments.get(0)).byteWidth();
            int resultWidth = dataType(arguments.get(1)).byteWidth();
            long sourceBase = arguments.get(0).byteOffset() / sourceWidth;
            long resultBase = arguments.get(1).byteOffset() / resultWidth;
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
            int width = artifact.specialization().boundaryDataTypes().get(value).byteWidth();
            long carrierBase = arguments.get(value).byteOffset() / width;
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

    private static boolean overlaps(CpuBufferArgument left, CpuAccessPlan.Binding leftBinding,
            CpuBufferArgument right, CpuAccessPlan.Binding rightBinding) {
        if (leftBinding.elementCount() == 0 || rightBinding.elementCount() == 0) return false;
        Object leftCarrier = carrier(left);
        Object rightCarrier = carrier(right);
        if (leftCarrier instanceof MemorySegment a && rightCarrier instanceof MemorySegment b
                && a.asOverlappingSlice(b).isEmpty()) return false;
        if (leftCarrier != rightCarrier && !(leftCarrier instanceof MemorySegment
                && rightCarrier instanceof MemorySegment)) return false;
        int leftWidth = dataType(left).byteWidth();
        int rightWidth = dataType(right).byteWidth();
        long leftStart = Math.addExact(carrierBase(left),
                Math.multiplyExact(leftBinding.accessedElementStart(), leftWidth));
        long leftEnd = Math.addExact(carrierBase(left),
                Math.multiplyExact(leftBinding.accessedElementEnd(), leftWidth));
        long rightStart = Math.addExact(carrierBase(right),
                Math.multiplyExact(rightBinding.accessedElementStart(), rightWidth));
        long rightEnd = Math.addExact(carrierBase(right),
                Math.multiplyExact(rightBinding.accessedElementEnd(), rightWidth));
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private boolean affineOverlaps(CpuBufferArgument left, CpuBufferArgument right,
            int leftIndex, int rightIndex) {
        Object leftCarrier = carrier(left), rightCarrier = carrier(right);
        if (leftCarrier instanceof MemorySegment a && rightCarrier instanceof MemorySegment b
                && a.asOverlappingSlice(b).isEmpty()) return false;
        if (leftCarrier != rightCarrier && !(leftCarrier instanceof MemorySegment
                && rightCarrier instanceof MemorySegment)) return false;
        int width = artifact.specialization().boundaryDataTypes().get(leftIndex).byteWidth();
        long leftBase = carrierBase(left), rightBase = carrierBase(right);
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

    private static long carrierBase(CpuBufferArgument argument) {
        return argument instanceof CpuBufferArgument.Segment segment
                ? segment.segment().address() : argument.byteOffset();
    }

    @FunctionalInterface private interface KernelCall { void invoke() throws Throwable; }

    private static KernelCall callFor(MethodHandle h, List<CpuBufferArgument> a,
            MemorySegment scratch, long[] g, long start, long end) {
        Object[] carriers = a.stream().map(CpuPreparedExecutable::carrier).toArray();
        MethodHandle bound = MethodHandles.insertArguments(h, 0, carriers);
        if (scratch != null) bound = MethodHandles.insertArguments(bound, 0, scratch);
        bound = MethodHandles.insertArguments(bound, 0, (Object) g);
        MethodHandle target = bound;
        return () -> invokeVoid(target, start, end);
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

    private static io.github.pho001.synaptik.model.datatype.DataType dataType(
            CpuBufferArgument argument) {
        if (argument instanceof CpuBufferArgument.Doubles) return io.github.pho001.synaptik.model.datatype.DataType.FLOAT64;
        if (argument instanceof CpuBufferArgument.Floats) return io.github.pho001.synaptik.model.datatype.DataType.FLOAT32;
        if (argument instanceof CpuBufferArgument.Shorts) return io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16;
        if (argument instanceof CpuBufferArgument.Ints) return io.github.pho001.synaptik.model.datatype.DataType.INT32;
        if (argument instanceof CpuBufferArgument.Longs) return io.github.pho001.synaptik.model.datatype.DataType.INT64;
        if (argument instanceof CpuBufferArgument.Bytes) return io.github.pho001.synaptik.model.datatype.DataType.BOOL;
        return ((CpuBufferArgument.Segment) argument).dataType();
    }

    private void validateCanonicalBooleanInputs(List<CpuBufferArgument> arguments) {
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
                for (long logical = start; logical < end; logical++) {
                    int pair = Math.toIntExact(Math.multiplyExact(logical, 2));
                    long address = affineAddressPairs[pair];
                    byte value = readByte(arguments.get(index), address);
                    if (value != 0 && value != 1) throw new IllegalArgumentException(
                            "BOOL condition must use canonical bytes");
                }
                continue;
            }
            CpuAccessPlan.Binding binding = movementGeometry.isPresent() || indexingGeometry.isPresent()
                    || scatterGeometry.isPresent() || aggregateGeometry.isPresent()
                    || maskedReductionGeometry.isPresent() || attentionGeometry.isPresent()
                    ? bindings.get(index) : ranged(bindings.get(index));
            long[] extents = binding.extents().stream().mapToLong(Long::longValue).toArray();
            long[] strides = binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
            long[] coordinates = binding.startCoordinates().stream().mapToLong(Long::longValue).toArray();
            long address = binding.startAddress();
            for (long logical = binding.start(); logical < binding.end(); logical++) {
                byte value = readByte(arguments.get(index), address);
                if (value != 0 && value != 1) {
                    throw new IllegalArgumentException("BOOL condition must use canonical bytes");
                }
                address = advanceAddress(address, extents, strides, coordinates);
            }
        }
    }

    private static byte readByte(CpuBufferArgument argument, long address) {
        if (argument instanceof CpuBufferArgument.Bytes bytes) {
            return bytes.carrier()[Math.toIntExact(bytes.byteOffset() + address)];
        }
        return ((CpuBufferArgument.Segment) argument).segment().get(ValueLayout.JAVA_BYTE, address);
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

        IndexValidation(List<CpuBufferArgument> arguments, CpuIndexingLowering.Geometry geometry) {
            family = geometry.family();
            int indexBoundary = family
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.ONE_HOT
                    ? geometry.occurrenceToBoundary().getFirst()
                    : geometry.occurrenceToBoundary().get(1);
            indexArgument = arguments.get(indexBoundary);
            indexType = geometry.boundaryTypes().get(indexBoundary);
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
            java.util.Arrays.fill(coordinates, 0);
            long address = base;
            int component = 0;
            for (long ordinal = 0; ordinal < indexCount; ordinal++) {
                long value = readIndex(indexArgument, address, indexType);
                int currentAxis = axis;
                long currentBound = bound;
                if (family == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.GATHER_ND) {
                    currentAxis = batch + component;
                    currentBound = dataExtents[currentAxis];
                }
                if (value < 0 || value >= currentBound) throw failure(ordinal, value,
                        currentAxis, currentBound);
                address = advanceAddress(address, extents, strides, coordinates);
                if (family == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr.Family.GATHER_ND
                        && ++component == tuple) component = 0;
            }
        }

        private IndexOutOfBoundsException failure(long ordinal, long value, int selectedAxis,
                long extent) {
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

    private static long readIndex(CpuBufferArgument argument, long address,
            io.github.pho001.synaptik.model.datatype.DataType type) {
        if (type == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            if (argument instanceof CpuBufferArgument.Ints ints) {
                return ints.carrier()[Math.toIntExact(address)];
            }
            return ((CpuBufferArgument.Segment) argument).segment().get(ValueLayout.JAVA_INT,
                    Math.multiplyExact(address, Integer.BYTES));
        }
        if (argument instanceof CpuBufferArgument.Longs longs) {
            return longs.carrier()[Math.toIntExact(address)];
        }
        return ((CpuBufferArgument.Segment) argument).segment().get(ValueLayout.JAVA_LONG,
                Math.multiplyExact(address, Long.BYTES));
    }

    /** Reusable allocation-free complete scatter bounds and replacement-uniqueness validator. */
    private static final class ScatterValidation {
        private final List<CpuBufferArgument> arguments;
        private final CpuScatterLowering.Geometry geometry;
        private final int indexBoundary;
        private final CpuBufferArgument indices;
        private final io.github.pho001.synaptik.model.datatype.DataType indexType;
        private final long[] indexExtents;
        private final long[] indexStrides;
        private final long[] tuplePrefixExtents;
        private final long indexBase;
        private final long[] current;
        private final long[] earlier;

        ScatterValidation(List<CpuBufferArgument> arguments,
                CpuScatterLowering.Geometry geometry) {
            this.arguments = arguments;
            this.geometry = geometry;
            indexBoundary = geometry.occurrenceToBoundary().get(1);
            indices = arguments.get(indexBoundary);
            indexType = geometry.boundaryTypes().get(indexBoundary);
            var layout = geometry.boundaries().get(indexBoundary);
            indexExtents = layout.extents(); indexStrides = layout.strides();
            tuplePrefixExtents = java.util.Arrays.copyOf(indexExtents,
                    Math.max(0, indexExtents.length - 1));
            indexBase = Math.addExact(indices.byteOffset() / indexType.byteWidth(), layout.offset());
            current = new long[Math.max(indexExtents.length,
                    geometry.boundaries().get(geometry.occurrenceToBoundary().get(2)).extents().length)];
            earlier = current.clone();
        }

        void validate() {
            bounds();
            if (geometry.reduction()
                    == io.github.pho001.synaptik.model.operation.index.ScatterReduction.NONE) {
                duplicates();
            }
        }

        private void bounds() {
            java.util.Arrays.fill(current, 0);
            long address=indexBase;
            long count=elementCount(indexExtents);
            for(long ordinal=0;ordinal<count;ordinal++){
                long value=readIndex(indices,address,indexType);
                int axis=geometry.family()
                        ==io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ND
                        ?geometry.batchDimensions()+(int)(ordinal%geometry.tupleDepth())
                        :geometry.axis();
                long extent=geometry.boundaries().get(geometry.occurrenceToBoundary().get(0))
                        .extents()[axis];
                if(value<0||value>=extent)throw boundsFailure(ordinal,axis,value,extent);
                address=advanceAddress(address,indexExtents,indexStrides,current);
            }
        }

        private void duplicates() {
            if (geometry.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ADD) return;
            if (geometry.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ELEMENTS) {
                long[] updateExtents=geometry.boundaries().get(
                        geometry.occurrenceToBoundary().get(2)).extents();
                long count=elementCount(updateExtents);
                for(long later=1;later<count;later++){
                    coordinates(later,updateExtents,current);
                    for(long first=0;first<later;first++){
                        coordinates(first,updateExtents,earlier);
                        if(elementTargetEqual(current,earlier))throw new IllegalArgumentException(
                                "SCATTER_ELEMENTS duplicate target at logical update position "+later
                                +"; first addressed at logical update position "+first);
                    }
                }
                return;
            }
            long tupleCount=1;
            for(int i=0;i<indexExtents.length-1;i++)tupleCount=Math.multiplyExact(tupleCount,indexExtents[i]);
            for(long later=1;later<tupleCount;later++){
                coordinates(later,tuplePrefixExtents,current);
                for(long first=0;first<later;first++){
                    coordinates(first,tuplePrefixExtents,earlier);
                    if(ndTargetEqual(current,earlier))throw new IllegalArgumentException(
                            "SCATTER_ND duplicate target tuple at logical tuple position "+later
                            +"; first addressed at logical tuple position "+first);
                }
            }
        }

        private boolean elementTargetEqual(long[] left,long[] right){
            int axis=geometry.axis();
            for(int i=0;i<indexExtents.length;i++)if(i!=axis&&left[i]!=right[i])return false;
            return indexAt(left,-1)==indexAt(right,-1);
        }
        private boolean ndTargetEqual(long[] left,long[] right){
            for(int i=0;i<geometry.batchDimensions();i++)if(left[i]!=right[i])return false;
            for(int k=0;k<geometry.tupleDepth();k++)if(indexAt(left,k)!=indexAt(right,k))return false;
            return true;
        }
        private long indexAt(long[] coordinate,int component){
            long address=indexBase;
            int prefix=indexExtents.length-1;
            for(int i=0;i<(component<0?indexExtents.length:prefix);i++)
                address=Math.addExact(address,Math.multiplyExact(coordinate[i],indexStrides[i]));
            if(component>=0)address=Math.addExact(address,Math.multiplyExact(component,indexStrides[prefix]));
            return readIndex(indices,address,indexType);
        }
        private IndexOutOfBoundsException boundsFailure(long ordinal,int axis,long value,long extent){
            String family=geometry.family().name();
            return new IndexOutOfBoundsException(family+" index at logical position "+ordinal
                    +" for data axis "+axis+" is out of bounds: value="+value+", extent="+extent);
        }
        private static void coordinates(long ordinal,long[] extents,long[] target){
            java.util.Arrays.fill(target,0);
            for(int i=extents.length-1;i>=0;i--){if(extents[i]!=0){target[i]=ordinal%extents[i];ordinal/=extents[i];}}
        }
    }

    private final class Invocation extends BoundInvocation {
        private final IndexValidation validation;
        private final ScatterValidation scatterValidation;
        private final KernelCall prologue;
        private final KernelCall call;
        private final CpuWorkerGroup.RangeCall[] calls;
        Invocation(RunState state, IndexValidation validation,
                ScatterValidation scatterValidation, KernelCall prologue, KernelCall call,
                CpuWorkerGroup.RangeCall[] calls) {
            super(state); this.validation = validation;
            this.scatterValidation=scatterValidation; this.prologue=prologue;
            this.call = call; this.calls = calls;
        }
        @Override protected void executeBound() {
            try {
                if (validation != null) validation.validate();
                if (scatterValidation != null) scatterValidation.validate();
                if (prologue != null) prologue.invoke();
                if (call != null) call.invoke();
                else if (calls != null) workerGroup.execute(calls);
            }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("generated CPU invocation failed", failure); }
        }
    }
}
