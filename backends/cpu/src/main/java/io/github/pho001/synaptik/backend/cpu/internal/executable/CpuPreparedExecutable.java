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
            Optional<CpuOrderingLowering.Geometry> orderingGeometry) {
        this(memoryPlan, selections, artifact, bindings, carrierPattern, generatedCarrierPattern,
                start, end, selectedRangeCount, minimumElementsPerWorker, workerGroup,
                materialization, workspaceSelection, affineAddressPairs, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty());
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
                Optional.empty());
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
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry) {
        super(memoryPlan, selections, workspaceSelection.map(List::of).orElseGet(List::of),
                accesses(selections.size(), randomGeometry.map(g -> g.family()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT
                        ? 3 : 1).orElseGet(() -> orderingGeometry.filter(g ->
                        g.family() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr.Family.TOP_K)
                        .isPresent() ? 2 : 1)));
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
        long count = this.maskedReductionGeometry.isPresent()
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
        this.materialization = Objects.requireNonNull(materialization, "materialization");
        this.workspaceSelection = Objects.requireNonNull(workspaceSelection, "workspaceSelection");
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
        if (geometryCount>1) {
            throw new IllegalArgumentException("affine, movement, indexing, scatter, and fold geometry are exclusive");
        }
        boolean scatterScratch=this.scatterGeometry.filter(g->g.scratchSliceBytes()>0).isPresent();
        boolean aggregateScratch=this.aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent();
        boolean maskedScratch=this.maskedReductionGeometry.filter(g -> g.outputCount() > 0)
                .isPresent();
        if (workspaceSelection.isPresent() != (materialization.isPresent() || scatterScratch
                || aggregateScratch || maskedScratch || this.orderingGeometry.isPresent())) {
            throw new IllegalArgumentException("workspace selection purpose is inconsistent");
        }
        int materializedPosition = materialization
                .map(CpuMaterializationPlan::sourceBoundaryIndex).orElse(-1);
        if (artifact.specialization().materializedSourcePosition() != materializedPosition
                || materialization.isPresent() && !bindings.get(materializedPosition)
                        .equals(materialization.orElseThrow().consumerBinding())) {
            throw new IllegalArgumentException(
                    "materialization and generated specialization must agree");
        }
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
        return ranged(movementGeometry.isPresent() || indexingGeometry.isPresent()
                || scatterGeometry.isPresent() || foldGeometry.isPresent() || orderingGeometry.isPresent()
                || randomGeometry.isPresent()
                || scanGeometry.isPresent() || aggregateGeometry.isPresent()
                || argExtremaGeometry.isPresent() || maskedReductionGeometry.isPresent()
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
        if (movementGeometry.isPresent() || indexingGeometry.isPresent()
                || scatterGeometry.isPresent() || foldGeometry.isPresent() || orderingGeometry.isPresent()
                || randomGeometry.isPresent() || scanGeometry.isPresent()
                || aggregateGeometry.isPresent() || argExtremaGeometry.isPresent()
                || maskedReductionGeometry.isPresent()) {
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
                minimumElementsPerWorker, workerGroup, materialization,
                workspaceSelection, affineCopy ? affineAddressPairs : null, movementGeometry,
                indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry,
                scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry);
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
            int firstOutput = orderingGeometry.filter(g -> g.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr.Family.TOP_K)
                    .isPresent() ? bindings.size() - 2 : randomGeometry.map(g -> g.family()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT
                        ? bindings.size() - 3 : 0).orElse(bindings.size() - 1);
            return actual == carrierPattern.get(index) && aligned
                    && (index < firstOutput || !argument.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        if (index != 0 || materialization.isEmpty() && scatterGeometry.isEmpty()
                    && orderingGeometry.isEmpty() && aggregateGeometry.isEmpty()
                    && maskedReductionGeometry.isEmpty()
                || !(representation instanceof CpuContiguousWorkspace workspace)
                || !workspace.isAccessible()) return false;
        long bytes=materialization.map(CpuMaterializationPlan::byteCount)
                .orElseGet(()->scatterGeometry.filter(g -> g.scratchSliceBytes() > 0)
                        .map(g -> g.workspaceBytes(selectedRangeCount))
                        .orElseGet(() -> aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0)
                        .map(g -> g.workspaceBytes(selectedRangeCount))
                        .orElseGet(() -> maskedReductionGeometry
                            .map(g -> g.workspaceBytes(selectedRangeCount))
                            .orElseGet(() -> orderingGeometry.orElseThrow()
                                .workspaceBytes(selectedRangeCount)))));
        long alignment=materialization.map(CpuMaterializationPlan::byteAlignment).orElse(8L);
        return workspace.byteSize() == bytes && workspace.byteAlignment() == alignment
                && workspace.writableSegment().address() % alignment == 0;
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        boolean hasWorkspace=materialization.isPresent()
                || scatterGeometry.filter(g->g.scratchSliceBytes()>0).isPresent()
                || aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                    && g.outputCount() > 0).isPresent()
                || maskedReductionGeometry.filter(g -> g.outputCount() > 0).isPresent()
                || orderingGeometry.isPresent();
        if (workspaces.length != (hasWorkspace ? 1 : 0)) {
            throw new IllegalArgumentException("workspace count disagrees with prepared use");
        }
        var arguments = new ArrayList<CpuBufferArgument>(bindings.size());
        for (BufferRepresentation buffer : buffers) {
            arguments.add(((CpuBufferRepresentation) buffer).argument());
        }
        int firstOutputIndex = orderingGeometry.filter(g ->
                g.family() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr.Family.TOP_K)
                .isPresent() ? bindings.size() - 2 : randomGeometry.map(g -> g.family()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr.Family.DROPOUT
                    ? bindings.size() - 3 : 0).orElse(bindings.size() - 1);
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
            CpuAccessPlan.Binding inputBinding = materialization.isPresent()
                    && materialization.orElseThrow().sourceBoundaryIndex() == input
                    ? materialization.orElseThrow().sourceBinding() : bindings.get(input);
            boolean overlap = affineCopy
                    ? affineOverlaps(arguments.get(input), arguments.get(firstOutputIndex), input,
                            firstOutputIndex)
                    : movementGeometry.isPresent() || indexingGeometry.isPresent()
                            || scatterGeometry.isPresent() || foldGeometry.isPresent()
                            || orderingGeometry.isPresent() || scanGeometry.isPresent()
                            || aggregateGeometry.isPresent() || argExtremaGeometry.isPresent()
                            || maskedReductionGeometry.isPresent()
                        ? overlaps(arguments.get(input), inputBinding,
                            arguments.get(firstOutputIndex), bindings.get(firstOutputIndex))
                        : overlaps(arguments.get(input), ranged(inputBinding),
                            arguments.get(firstOutputIndex), ranged(bindings.get(firstOutputIndex)));
            if (overlap) {
                throw new IllegalArgumentException("output accessed span must not overlap an input");
            }
        }
        if (firstOutputIndex + 1 < bindings.size()) {
            for (int output = firstOutputIndex; output < bindings.size(); output++) {
                if (overlaps(arguments.get(0), bindings.get(0), arguments.get(output),
                        bindings.get(output))) throw new IllegalArgumentException(
                                "output accessed span must not overlap an input");
            }
            if (overlaps(arguments.get(firstOutputIndex), bindings.get(firstOutputIndex),
                    arguments.get(firstOutputIndex + 1), bindings.get(firstOutputIndex + 1)))
                throw new IllegalArgumentException("ordering outputs must not overlap");
        }
        }
        validateCanonicalBooleanInputs(arguments);
        IndexValidation validation = indexingGeometry.map(g ->
                new IndexValidation(arguments, g)).orElse(null);
        ScatterValidation scatterValidation = scatterGeometry.map(g ->
                new ScatterValidation(arguments, g)).orElse(null);
        CopyCall copyCall = null;
        if (materialization.isPresent()) {
            var copy = materialization.orElseThrow();
            CpuBufferArgument source = arguments.get(copy.sourceBoundaryIndex());
            MemorySegment workspace = ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
            if (end > start) copyCall = copyCall(
                    source, copy.sourceBinding(), workspace, copy.elementCount());
            arguments.set(copy.sourceBoundaryIndex(), new CpuBufferArgument.Segment(
                    io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,
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
        long length = end - start;
        KernelCall prologue = randomGeometry.isPresent() ? callFor(artifact.entryPoint(), arguments,
                null, geometry(arguments, 0, 0, 0), 0, 0) : null;
        int chunkCount = length == 0 ? 0 : Math.min(selectedRangeCount,
                Math.toIntExact(1 + (length - 1) / minimumElementsPerWorker));
        if (chunkCount <= 1) {
            KernelCall call = length == 0 ? null : callFor(artifact.entryPoint(), arguments,
                    artifact.specialization().scratchParameter() ? scratch(workspaces, 0) : null,
                    geometry(arguments, start, end, 0), start, end);
            return new Invocation(state, copyCall, validation, scatterValidation, prologue, call, null);
        }
        CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[chunkCount];
        long quotient = length / chunkCount;
        long remainder = length % chunkCount;
        long chunkStart = start;
        for (int index = 0; index < chunkCount; index++) {
            long chunkEnd = chunkStart + quotient + (index < remainder ? 1 : 0);
            KernelCall call = callFor(artifact.entryPoint(), arguments,
                    artifact.specialization().scratchParameter() ? scratch(workspaces, index) : null,
                    geometry(arguments, chunkStart, chunkEnd, index), chunkStart, chunkEnd);
            calls[index] = call::invoke;
            chunkStart = chunkEnd;
        }
        return new Invocation(state, copyCall, validation, scatterValidation, prologue, null, calls);
    }

    @FunctionalInterface private interface CopyCall { void invoke(); }

    private static CopyCall copyCall(CpuBufferArgument source, CpuAccessPlan.Binding binding,
            MemorySegment target, long count) {
        int rank = binding.extents().size();
        long[] extents = binding.extents().stream().mapToLong(Long::longValue).toArray();
        long[] strides = binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
        long[] coordinates = new long[rank];
        if (source instanceof CpuBufferArgument.Doubles doubles) {
            long arrayBase = doubles.byteOffset() / Double.BYTES;
            return () -> copyArray(doubles.carrier(), arrayBase, binding.baseElementOffset(),
                    target, count, extents, strides, coordinates);
        }
        MemorySegment segment = ((CpuBufferArgument.Segment) source).segment();
        return () -> copySegment(segment, binding.baseElementOffset(), target, count,
                extents, strides, coordinates);
    }

    private static void copyArray(double[] source, long arrayBase, long initialAddress,
            MemorySegment target, long count, long[] extents, long[] strides,
            long[] coordinates) {
        java.util.Arrays.fill(coordinates, 0);
        long address = initialAddress;
        for (long logical = 0; logical < count; logical++) {
            double value = source[Math.toIntExact(arrayBase + address)];
            target.set(ValueLayout.JAVA_DOUBLE,
                    Math.multiplyExact(logical, Double.BYTES), value);
            address = advanceAddress(address, extents, strides, coordinates);
        }
    }

    private static void copySegment(MemorySegment source, long initialAddress,
            MemorySegment target, long count, long[] extents, long[] strides,
            long[] coordinates) {
            java.util.Arrays.fill(coordinates, 0);
            long address = initialAddress;
            for (long logical = 0; logical < count; logical++) {
                double value = source.get(ValueLayout.JAVA_DOUBLE,
                        Math.multiplyExact(address, Double.BYTES));
                target.set(ValueLayout.JAVA_DOUBLE,
                        Math.multiplyExact(logical, Double.BYTES), value);
                address = advanceAddress(address, extents, strides, coordinates);
            }
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
        return whole;
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
                    || maskedReductionGeometry.isPresent()
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
        private final CopyCall copy;
        private final IndexValidation validation;
        private final ScatterValidation scatterValidation;
        private final KernelCall prologue;
        private final KernelCall call;
        private final CpuWorkerGroup.RangeCall[] calls;
        Invocation(RunState state, CopyCall copy, IndexValidation validation,
                ScatterValidation scatterValidation, KernelCall prologue, KernelCall call,
                CpuWorkerGroup.RangeCall[] calls) {
            super(state); this.copy = copy; this.validation = validation;
            this.scatterValidation=scatterValidation; this.prologue=prologue;
            this.call = call; this.calls = calls;
        }
        @Override protected void executeBound() {
            try {
                if (copy != null) copy.invoke();
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
