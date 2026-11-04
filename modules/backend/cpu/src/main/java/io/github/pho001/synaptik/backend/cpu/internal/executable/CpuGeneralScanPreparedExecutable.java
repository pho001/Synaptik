package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.*;

/**
 * Immutable post-assignment recipe for one generated scan invocation. No captured body, graph,
 * operation or query reaches this recipe. Each cold binding creates its own typed segment array,
 * geometry and access owners under the caller's single RunState. Whole selected representation
 * capacity, including tails, participates in write/scratch overlap checks before any write.
 * Composite bindings preserve every boundary's independent dtype and bind ordinary numeric packs,
 * validator cursors, control slabs and helper scratch slices once per invocation. Generated
 * invocation-local readiness flags lazily stage immutable SHARED formals at most once under their
 * consuming scopes; sequences always retain their changing original-time versions.
 * The composite generated entry is called once and owns all direct helper calls. Length mode binds
 * a separate aligned INT64 control span. Validation honors the authenticated lengths offset and
 * stride after producers, before any generated output/history/scratch/control write, including T=0.
 * Mutable packs and cursors are invocation-private and reset by their generated owners on every
 * helper call, including retries after value-validation failure.
 * Low numerical composition uses these same typed bindings: FP8 bytes are not BOOL-normalized,
 * short formats remain shorts, and per-value width/alignment is authenticated independently.
 */
public final class CpuGeneralScanPreparedExecutable extends PreparedExecutable {
    /**
     * One RunState's synchronous bridge to the single selected row region. Arrays and calls
     * are cold-bound once; generated code updates only active-time operand/base facts. This is
     * neither a stage registry nor a second time-control owner. The invocation exclusively lends
     * its reservation until all started workers join, including failure/interruption/close paths.
     */
    public static final class StepDispatch {
        private final MemorySegment[] operands;
        private final long[] frame;
        private final CpuWorkerGroup.RangeCall[] calls;
        private final CpuWorkerGroup workerGroup;
        private CpuConcurrencyBudget.Lease reservation;

        /**
         * Retains exclusively owned run arrays and prebound calls without copying at dispatch.
         * @param operands fresh mutable run operand bindings
         * @param frame fresh mutable run numeric frame
         * @param calls fresh prebound ordered ranges, never modified after binding
         * @param workerGroup borrowed coordinated group, including inline/empty selected ranges
         * @throws NullPointerException if any argument is null
         * @throws IllegalArgumentException if the group has no coordinated budget
         */
        private StepDispatch(MemorySegment[] operands, long[] frame,
                CpuWorkerGroup.RangeCall[] calls, CpuWorkerGroup workerGroup) {
            this.operands = Objects.requireNonNull(operands);
            this.frame = Objects.requireNonNull(frame);
            this.calls = Objects.requireNonNull(calls);
            this.workerGroup = Objects.requireNonNull(workerGroup);
            if (workerGroup.concurrencyBudget() == null)
                throw new IllegalArgumentException("parallel Scan requires coordinated workers");
        }

        /**
         * Exposes this invocation's own bindings for generated active-time updates, without copy.
         * @return exclusively run-owned mutable operand array, never null
         */
        public MemorySegment[] operands() { return operands; }

        /**
         * Exposes this invocation's own frame for generated active-time base updates, without copy.
         * @return exclusively run-owned mutable numeric frame, never null; its header holds
         *     original time, total rows, private scratch byte base and actual range count,
         *     followed by per-value read/write byte bases and byte strides, then demanded-port byte bases
         */
        public long[] frame() { return frame; }

        /**
         * Executes prebound ranges synchronously using only the privately lent outer reservation.
         * The caller performs no concurrent math, reuses no scratch and cannot clear or close the
         * reservation before this call quiesces. Fewer than two ranges execute inline.
         * @throws IllegalStateException if no live sufficient reservation is installed, or a
         *     bound inline call throws a checked failure
         * @throws CpuWorkerGroup.CpuParallelExecutionException if dispatch is cancelled or
         *     interrupted, or a checked worker failure is wrapped; runtime failures/errors propagate
         */
        public void execute() {
            if (reservation == null || !reservation.covers(workerGroup.concurrencyBudget(),Math.max(1,calls.length)))
                throw new IllegalStateException("Scan reservation is not bound or sufficient");
            if (calls.length >= 2) { workerGroup.execute(calls,reservation); return; }
            try { if (calls.length == 1) calls[0].invoke(); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("Scan inline row range failed",failure); }
        }
    }
    private final List<Long> spans;
    private final int reads;
    private final DataType type;
    private final int width;
    private final Facts facts;
    private final Optional<WorkspaceSelection> selectedWorkspace;
    private final MethodHandle entry;
    private final CpuConcurrencyBudget budget;
    /** Cold policy and borrowed coordinated workers; never stored in canonical Scan facts. */
    private final CpuWorkerGroup workerGroup;
    private final long minimumElementsPerWorker;
    private final CpuGeneratedKernel artifact;

    /**
     * Constructs an immutable assigned recipe and adapts its generated entry once, without
     * binding a run, reading tensor storage or transferring resource ownership. Run binding
     * later creates the mutable argument/geometry arrays and validates complete capacities.
     * @param plan non-null exact immutable memory plan
     * @param buffers non-null unique read selections followed by distinct demanded writes;
     *     copied by the base recipe
     * @param workspace non-null optional exact workspace for independent carry banks/live cuts;
     *     symbolic mode requires its structural selection even for zero bytes
     * @param facts non-null immutable numeric IR/geometry/access recipes retained for cold binding;
     *     no Model body, query or mutable run geometry is retained
     * @param artifact non-null verified generated artifact with matching typed scan identity
     * @param budget borrowed inline admission budget, or null
     * @throws NullPointerException if a required reference or selection is null
     * @throws IllegalArgumentException if assignments or generated identity disagree
     */
    public CpuGeneralScanPreparedExecutable(PreparedMemoryPlan plan, List<BufferSelection> buffers,
            Optional<WorkspaceSelection> workspace, Facts facts, CpuGeneratedKernel artifact,
            CpuConcurrencyBudget budget) {
        this(plan,buffers,workspace,facts,artifact,budget,null,1);
    }

    /**
     * Constructs a selected Scan recipe without binding tensor storage. The run reserves the
     * coordinated budget once across time and lends it synchronously to prebound row calls.
     * @param plan exact assigned memory plan
     * @param buffers ordered immutable boundary selections
     * @param workspace exact declared carry/cut/staging/partial workspace
     * @param facts numeric Scan binding facts, without execution policy
     * @param artifact verified generated Scan artifact
     * @param budget borrowed admission budget, null only for uncoordinated inline execution
     * @param workerGroup borrowed coordinated workers for selected parallel rows, otherwise null
     * @param minimumElementsPerWorker positive minimum row count used only at cold binding
     * @throws IllegalArgumentException if policy, workers, budget or artifact disagree
     * @throws NullPointerException if a required reference is null
     */
    public CpuGeneralScanPreparedExecutable(PreparedMemoryPlan plan,List<BufferSelection> buffers,
            Optional<WorkspaceSelection> workspace,Facts facts,CpuGeneratedKernel artifact,
            CpuConcurrencyBudget budget,CpuWorkerGroup workerGroup,long minimumElementsPerWorker) {
        super(plan, buffers, workspace.stream().toList(), java.util.stream.IntStream.range(0, buffers.size())
                .mapToObj(p -> p < facts.ir().scalarReads().size() ? BufferAccess.READ_ONLY : BufferAccess.WRITE_ONLY).toList());
        if (!artifact.specialization().generalScanIr().equals(Optional.of(facts.ir()))
                || buffers.size() != (facts.ir().symbolic() ? facts.batchBindings().size() : facts.spans().size())
                || workspace.isPresent() != (facts.ir().composition().isPresent()
                    || facts.ir().symbolic() || facts.geometry().scratchBytes() > 0))
            throw new IllegalArgumentException("assigned scan selections/artifact disagree");
        if (minimumElementsPerWorker <= 0) throw new IllegalArgumentException("minimum row work must be positive");
        int maximum = artifact.specialization().rowFusionEmission().map(e -> e.maximumPartialCount()).orElse(1);
        if ((maximum > 1) != (workerGroup != null)
                || workerGroup != null && (budget == null || workerGroup.concurrencyBudget() != budget
                    || workerGroup.workerCount() < maximum || !workerGroup.isOpen() || budget.capacity() < maximum))
            throw new IllegalArgumentException("Scan row workers/reservation policy disagree");
        this.workerGroup = workerGroup; this.minimumElementsPerWorker = minimumElementsPerWorker;
        this.artifact = artifact;
        type = facts.ir().dataType(); width = type.byteWidth();
        spans = facts.spans(); reads = facts.ir().scalarReads().size();
        this.facts = facts; selectedWorkspace = workspace; this.budget = budget;
        entry = facts.ir().composition().isPresent() ? artifact.entryPoint()
                : artifact.entryPoint().asSpreader(0, MemorySegment[].class,
                    buffers.size() + (workspace.isPresent() ? 1 : 0));
    }

    /**
     * Checks selected boundary dtype, byte capacity/alignment, accessibility and output
     * writability without reading tensor values. A wrong dtype is rejected even when its carrier
     * has enough bytes; read-only input storage is allowed.
     * @param p boundary position in the immutable selection list
     * @param representation selected run representation, or null to reject
     * @return whether this boundary can be bound as a checked segment
     */
    @Override protected boolean acceptsBufferRepresentation(int p, BufferRepresentation representation) {
        if (facts.ir().symbolic()) return false;
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != facts.ir().boundaryType(p)) return false;
        try {
            var segment = cpu.segment();
            int boundaryWidth = facts.ir().boundaryType(p).byteWidth();
            return segment.byteSize() >= Math.multiplyExact(spans.get(p), (long) boundaryWidth)
                    && segment.address() % boundaryWidth == 0 && (p < reads || !segment.isReadOnly());
        } catch (IllegalArgumentException | IllegalStateException failure) { return false; }
    }

    /**
     * Checks the exact declared carry-bank/body-cut workspace, not an execution-time allocation request.
     * @param p sole workspace position, zero
     * @param representation selected run workspace
     * @return whether exact byte size, access, alignment and permission agree
     */
    @Override protected boolean acceptsWorkspaceRepresentation(int p, WorkspaceRepresentation representation) {
        if (facts.ir().symbolic()) return false;
        return representation instanceof CpuContiguousWorkspace cpu && cpu.isAccessible()
                && cpu.byteSize() == facts.geometry().scratchBytes()
                && cpu.byteAlignment() >= workspaceAlignment()
                && cpu.writableSegment().address() % workspaceAlignment() == 0 && !cpu.writableSegment().isReadOnly();
    }

    /**
     * Validates actual selected bytes before cold binding, retaining full capacity for overlap checks.
     * Fixed occurrences authenticate their own fixed declarations and retained spans even when
     * another source makes the complete plan batch-required; only symbolic IR consumes actual B.
     * @param state exact invocation owner and authenticated numeric geometry
     * @param p ordered boundary position
     * @param representation selected CPU carrier, or null to reject
     * @return whether dtype/access/alignment and actual capacity qualify
     */
    @Override protected boolean acceptsBufferRepresentation(RunState state,int p,BufferRepresentation representation) {
        if (!facts.ir().symbolic()) {
            if (!state.memoryPlan().requiresBatchSize()) return acceptsBufferRepresentation(p, representation);
            return representation instanceof CpuBufferRepresentation cpu
                    && cpu.acceptsFixedGeometry(state, memoryPlan(), bufferSelection(p).bufferIndex(),
                            Math.multiplyExact(spans.get(p), (long) facts.ir().boundaryType(p).byteWidth()))
                    && acceptsBufferRepresentation(p, representation);
        }
        if (!state.memoryPlan().requiresBatchSize()) return false;
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible() || cpu.dataType() != facts.ir().boundaryType(p)) return false;
        try {
            var segment = cpu.segment();
            return segment.byteSize() >= state.geometry().bufferByteSize(bufferSelection(p).bufferIndex())
                    && segment.address()%facts.ir().boundaryType(p).byteWidth() == 0 && (p < reads || !segment.isReadOnly());
        } catch (IllegalArgumentException | IllegalStateException failure) { return false; }
    }

    /**
     * Requires the exact declared usable workspace span, including structural zero bytes.
     * Fixed occurrences retain their own exact scratch bytes and cannot bind a dynamic declaration.
     * @param state sole run owner
     * @param p sole selected workspace position
     * @param representation actual workspace
     * @return whether numeric size, alignment and access agree
     */
    @Override protected boolean acceptsWorkspaceRepresentation(RunState state,int p,WorkspaceRepresentation representation) {
        if (!facts.ir().symbolic()) {
            if (!state.memoryPlan().requiresBatchSize()) return acceptsWorkspaceRepresentation(p, representation);
            return representation instanceof CpuContiguousWorkspace cpu
                    && CpuBufferRepresentation.acceptsFixedWorkspaceGeometry(state, memoryPlan(),
                            selectedWorkspace.orElseThrow().workspaceIndex(), facts.geometry().scratchBytes(), cpu)
                    && acceptsWorkspaceRepresentation(p, representation);
        }
        if (!state.memoryPlan().requiresBatchSize()) return false;
        return representation instanceof CpuContiguousWorkspace cpu && cpu.isAccessible()
                && cpu.byteSize() == state.geometry().workspaceByteSize(selectedWorkspace.orElseThrow().workspaceIndex())
                && cpu.byteAlignment() >= workspaceAlignment()
                && cpu.writableSegment().address()%workspaceAlignment() == 0 && !cpu.writableSegment().isReadOnly();
    }

    /**
     * Reports the alignment shared by declared control, typed values and helper scratch.
     * @return eight for composite storage, otherwise the retained homogeneous element width
     */
    private int workspaceAlignment() { return facts.ir().composition().isPresent() ? 8 : width; }

    /**
     * Authenticates a container's nonfixed scratch selection to this symbolic occurrence.
     * @param selection exact dense workspace position selected by the container
     * @return whether this symbolic unit will validate that workspace when the child binds
     */
    boolean hasDynamicWorkspaceSelection(WorkspaceSelection selection) {
        return facts.ir().symbolic() && selectedWorkspace.isPresent()
                && selectedWorkspace.orElseThrow().equals(selection);
    }

    /**
     * Creates one run-owned invocation after complete overlap validation; read/read aliasing is legal.
     * @param state exact sole run owner
     * @param buffers compatible nominal selections, checked once at this cold boundary
     * @param workspaces compatible optional workspace for the declared banks and live body cuts
     * @return fresh direct typed invocation; never null
     * @throws IllegalArgumentException if a writable selection overlaps any other selected capacity
     */
    @Override protected BoundInvocation bindCompatible(RunState state, BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        var owners = new CpuBufferRepresentation[buffers.length];
        var segments = new MemorySegment[buffers.length + workspaces.length];
        for (int p = 0; p < buffers.length; p++) {
            owners[p] = (CpuBufferRepresentation) buffers[p]; segments[p] = owners[p].segment();
        }
        var scratch = workspaces.length == 0 ? null : (CpuContiguousWorkspace) workspaces[0];
        if (scratch != null) segments[buffers.length] = scratch.writableSegment();
        for (int left = 0; left < segments.length; left++) for (int right = left + 1; right < segments.length; right++)
            if ((left >= reads || right >= reads) && overlaps(segments[left], segments[right]))
                throw new IllegalArgumentException("general scan writes/scratch overlap selected capacity");
        long batch = facts.ir().symbolic() ? state.geometry().batchSize().orElseThrow() : 0;
        var fullSegments = segments.clone();
        for (int p = 0; p < buffers.length; p++) {
            long span = facts.ir().symbolic() ? facts.batchBindings().get(p).bind(batch).referencedElementSpan() : spans.get(p);
            segments[p] = segments[p].asSlice(0,Math.multiplyExact(span,(long)facts.ir().boundaryType(p).byteWidth()));
        }
        if (facts.ir().composition().isPresent())
            return bindComposite(state, owners, scratch, segments, fullSegments, batch);
        long[] geometry = facts.pack(batch);
        StepDispatch dispatch = null;
        int demand = 1;
        if (artifact.specialization().rowFusionEmission().isPresent()) {
            var emission = artifact.specialization().rowFusionEmission().orElseThrow();
            var region = facts.ir().stages().orElseThrow().rowMapReductionIr().orElseThrow();
            int anchor = region.rowSites().getFirst(), axis = region.rowAxes().get(anchor);
            long rows = Math.addExact(region.accesses().get(anchor).extentConstants().get(axis),
                    Math.multiplyExact(batch,region.accesses().get(anchor).extentBatchFactors().get(axis)));
            int n = rows == 0 ? 0 : (int)Math.min(emission.maximumPartialCount(),1+(rows-1)/minimumElementsPerWorker);
            long staged = 0, terminal = 0;
            for (var list : List.of(emission.invariantTransposeValues(),emission.temporalTransposeValues()))
                for (int v : list) staged = Math.addExact(staged,fixedCells(region,v));
            for (int v : region.terminalOutputs()) terminal = Math.addExact(terminal,fixedCells(region,v));
            long partials = Math.addExact(Math.multiplyExact(staged,width),7)&~7L;
            long slice = Math.addExact(8,Math.addExact(Math.multiplyExact(terminal,width),7)&~7L);
            long extra = Math.addExact(partials,Math.multiplyExact(slice,emission.maximumPartialCount()));
            long privateBase = Math.subtractExact(segments[buffers.length].byteSize(),extra);
            if (privateBase < 0 || privateBase%8 != 0) throw new IllegalArgumentException("Scan private row workspace mismatch");
            geometry = Arrays.copyOf(geometry,geometry.length+2);
            geometry[geometry.length-2] = privateBase; geometry[geometry.length-1] = rows;
            if (workerGroup != null) {
                var operands = new MemorySegment[emission.stepOperandCount()];
                var frame = new long[emission.stepFrameLongCount()];
                frame[1] = rows; frame[2] = privateBase; frame[3] = n;
                var calls = new CpuWorkerGroup.RangeCall[n];
                var helper = artifact.scanRowRangeEntryPoint();
                long quotient = rows/Math.max(1,n), remainder = rows%Math.max(1,n), cursor = 0;
                for (int r = 0; r < n; r++) {
                    long begin = cursor, end = cursor+quotient+(r < remainder ? 1 : 0);
                    long offset = privateBase+partials+r*slice; cursor = end;
                    calls[r] = () -> { helper.invokeExact(operands,frame,begin,end,offset); };
                }
                dispatch = new StepDispatch(operands,frame,calls,workerGroup);
                demand = Math.max(1,n);
            }
        }
        return new Invocation(state, owners, scratch, segments, fullSegments, geometry,dispatch,demand,batch);
    }
    /**
     * Binds direct composite arguments without consuming values or writing declared workspace.
     * Geometry and validator arrays are invocation-owned; no body-unit object reaches execution.
     * @param state sole invocation resource owner
     * @param owners exact typed outer boundary owners
     * @param scratch declared complete composite workspace owner, never null
     * @param boundaries cold-bound typed boundary segments followed by the workspace
     * @param fullSegments complete selected capacities for post-producer checks
     * @param batch actual named batch coordinate, or zero for fixed recipes
     * @return invocation with exclusively owned packs, wrappers and validator cursors
     * @throws IllegalArgumentException if workspace or ordinary inventory facts disagree
     * @throws ArithmeticException if numeric geometry or storage sizes overflow
     */
    private BoundInvocation bindComposite(RunState state, CpuBufferRepresentation[] owners,
            CpuContiguousWorkspace scratch, MemorySegment[] boundaries, MemorySegment[] fullSegments,
            long batch) {
        var composition = facts.ir().composition().orElseThrow();
        var workspace = Objects.requireNonNull(scratch, "composite Scan workspace").writableSegment();
        long expected = facts.geometry().workspaceSize().byteSize(batch);
        if (workspace.byteSize() != expected)
            throw new IllegalArgumentException("composite Scan workspace geometry mismatch");
        int unitCount = composition.units().size(), valueCount = composition.values().size();
        if (facts.ordinaryUnits().size() != unitCount
                || artifact.specialization().compositionUnits().size() != unitCount)
            throw new IllegalArgumentException("composite Scan ordinary inventory mismatch");
        long rows = 0;
        if (composition.lengthAware()) {
            rows = composition.boundaryMappings().get(composition.lengthRead()).bind(batch).extents().getFirst();
        } else {
            for (var value : composition.values()) if (value.rowAxis() >= 0) {
                rows = value.access().bind(batch).extents().get(value.rowAxis()); break;
            }
        }
        long roots = composition.activity().map(a -> (long) a.carries().size() + a.seedFormals().size()).orElse(0L);
        long controlBytes = Math.multiplyExact(Math.addExact(1, roots), Math.addExact(16, rows));
        var control = workspace.asSlice(composition.workspaceSize().byteSize(batch), controlBytes);
        var arguments = Arrays.copyOf(boundaries, owners.length + unitCount);
        var packs = new long[Math.addExact(unitCount,
                Math.addExact(Math.multiplyExact(3, valueCount), Math.multiplyExact(4, unitCount))) ][];
        long[] empty = new long[0]; Arrays.fill(packs, empty);
        for (var value : composition.values()) {
            var access = value.access().bind(batch);
            int position = unitCount + 3 * value.ordinal();
            packs[position] = access.extents().stream().mapToLong(Long::longValue).toArray();
            packs[position + 1] = access.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
            packs[position + 2] = new long[access.extents().size()];
        }
        for (int u = 0; u < unitCount; u++) {
            var unit = composition.units().get(u);
            var emission = artifact.specialization().compositionUnits().get(u);
            var ordinary = facts.ordinaryUnits().get(u);
            var lowered = ordinary.batchGeometry().isPresent()
                    ? ordinary.batchGeometry().orElseThrow().bind(batch) : ordinary;
            if (!lowered.portableKernelIr().structuralKey().equals(unit.ir().structuralKey())
                    || !lowered.boundaryValues().equals(ordinary.boundaryValues())
                    || !lowered.boundaryDataTypes().equals(emission.specialization().boundaryDataTypes())
                    || lowered.accessBindings().size() != unit.boundaries().size()
                    || lowered.batchGeometry().isPresent())
                throw new IllegalArgumentException("composite Scan bound ordinary geometry changed declarations");
            long[] bases = new long[unit.boundaries().size()];
            for (int b = 0; b < bases.length; b++) {
                var value = composition.values().get(unit.boundaries().get(b));
                int width = value.descriptor().dataType().byteWidth();
                long offset = value.offset().byteSize(batch);
                if (offset % width != 0)
                    throw new IllegalArgumentException("composite Scan typed base is misaligned");
                bases[b] = offset / width;
            }
            long[] geometry = CpuPreparedExecutable.packGeometry(lowered, emission.specialization(), bases, batch);
            int resetCount = emission.resets().size();
            packs[u] = Arrays.copyOf(geometry, Math.addExact(geometry.length, Math.addExact(resetCount, 1)));
            // Snapshot actual-B cursor seeds once; numerical helper packs retain their exact prefix.
            for (int r = 0; r < resetCount; r++)
                packs[u][geometry.length + r] = geometry[emission.resets().get(r).cell()];
            packs[u][packs[u].length - 1] = lowered.elementCount();
            long start = emission.scratchOffset().byteSize(batch);
            long end = u + 1 == unitCount ? expected
                    : artifact.specialization().compositionUnits().get(u + 1).scratchOffset().byteSize(batch);
            arguments[owners.length + u] = workspace.asSlice(start, Math.subtractExact(end, start));
            bindValidatorPacks(packs, unitCount, valueCount, u, unit, lowered, composition);
        }
        return new CompositeInvocation(state, owners, scratch, arguments, fullSegments, packs,
                workspace, control, facts.geometry().time(), batch, rows);
    }

    /**
     * Binds cold ordinary index/scatter parameters without a second pack-offset registry.
     * @param packs fresh invocation-owned geometry and validator array bank
     * @param units ordinary unit count, defining the numeric geometry bank prefix
     * @param values physical value count, defining the per-unit validator bank origin
     * @param ordinal current ordinary unit index
     * @param unit exact helper boundary ordering
     * @param lowered matching ordinary family geometry
     * @param composition immutable physical values and original row-coordinate facts
     * @throws ArithmeticException if a logical validation-domain size overflows
     */
    private static void bindValidatorPacks(long[][] packs, int units, int values, int ordinal,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.PhysicalUnit unit,
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering.LoweredPartition lowered,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Composition composition) {
        int extra = units + 3 * values + 4 * ordinal;
        if (lowered.indexingGeometry().isPresent()) {
            var geometry = lowered.indexingGeometry().orElseThrow();
            long axis = 0, batch = 0, tuple = 0, bound = -1;
            if (geometry.variant() instanceof io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering.Geometry.Axis a) {
                axis = a.axis();
                int data = unit.boundaries().get(geometry.occurrenceToBoundary().getFirst());
                bound = packs[units + 3 * data][a.axis()];
            } else if (geometry.variant() instanceof io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering.Geometry.Nd nd) {
                batch = nd.batchDimensions(); tuple = nd.tupleDepth();
            } else if (geometry.variant() instanceof io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering.Geometry.Hot hot) {
                bound = hot.depth();
            }
            packs[extra] = new long[] {axis, batch, tuple, bound};
        }
        if (lowered.scatterGeometry().isPresent()) {
            var geometry = lowered.scatterGeometry().orElseThrow();
            int index = unit.boundaries().get(geometry.occurrenceToBoundary().get(1));
            int update = unit.boundaries().get(geometry.occurrenceToBoundary().get(2));
            long[] indices = packs[units + 3 * index], updates = packs[units + 3 * update];
            boolean nd = geometry.family() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr.Family.SCATTER_ND;
            long[] prefix = nd ? Arrays.copyOf(indices, indices.length - 1) : indices;
            long candidates = product(nd ? prefix : updates, 0);
            int rowAxis = composition.values().get(nd ? index : update).rowAxis();
            long rowStride = rowAxis < 0 ? 0 : product(nd ? prefix : updates, rowAxis + 1);
            packs[extra] = new long[] {geometry.axis(), geometry.batchDimensions(), geometry.tupleDepth(),
                    product(indices, 0), candidates, rowStride};
            packs[extra + 1] = prefix;
            int rank = Math.max(indices.length, updates.length);
            packs[extra + 2] = new long[rank]; packs[extra + 3] = new long[rank];
        }
    }

    /**
     * Counts a logical suffix, preserving an empty-domain zero before multiplication.
     * @param extents resolved nonnegative logical extents
     * @param start inclusive suffix axis in the range zero through rank
     * @return checked suffix cell count, or one for a rank-zero suffix
     * @throws ArithmeticException if a nonempty product overflows
     */
    private static long product(long[] extents, int start) {
        for (int axis = start; axis < extents.length; axis++) if (extents[axis] == 0) return 0;
        long count = 1;
        for (int axis = start; axis < extents.length; axis++) count = Math.multiplyExact(count, extents[axis]);
        return count;
    }

    /** Owns one direct generated composite entry and all cold-bound mutable argument storage. */
    private final class CompositeInvocation extends BoundInvocation {
        private final RunState state;
        private final CpuBufferRepresentation[] owners;
        private final CpuContiguousWorkspace scratch;
        private final MemorySegment[] arguments, fullSegments;
        private final long[][] packs;
        private final MemorySegment workspace, control;
        private final long time, batch, rows;

        /**
         * Retains cold-bound arguments without copying or transferring resource ownership.
         * @param state sole invocation resource owner
         * @param owners typed outer boundary owners
         * @param scratch declared complete workspace owner
         * @param arguments typed boundaries followed by per-helper scratch wrappers
         * @param fullSegments original full selected capacities for length revalidation
         * @param packs invocation-private numeric geometry and validator cursors
         * @param workspace complete declared numeric workspace segment
         * @param control declared disjoint helper/root control slabs
         * @param time fixed source time count, including zero
         * @param batch actual named numeric recipe coordinate, or zero in fixed recipes
         * @param rows independently proved original row count, including no-row zero
         */
        private CompositeInvocation(RunState state, CpuBufferRepresentation[] owners,
                CpuContiguousWorkspace scratch, MemorySegment[] arguments, MemorySegment[] fullSegments,
                long[][] packs, MemorySegment workspace, MemorySegment control, long time, long batch, long rows) {
            super(state); this.state = state; this.owners = owners; this.scratch = scratch;
            this.arguments = arguments; this.fullSegments = fullSegments; this.packs = packs;
            this.workspace = workspace; this.control = control;
            this.time = time; this.batch = batch; this.rows = rows;
        }

        /** Revalidates produced lengths before the sole entry can touch any Scan-local storage. */
        @Override protected void executeBound() {
            CpuConcurrencyBudget.Lease lease = budget == null ? null : budget.acquire(1);
            try {
                for (var owner : owners) if (!owner.isAccessible())
                    throw new IllegalArgumentException("general scan representation is no longer accessible");
                if (!scratch.isAccessible())
                    throw new IllegalArgumentException("general scan scratch is no longer accessible");
                validateProducedLengths(state, owners, scratch, fullSegments, arguments, time, rows);
                entry.invokeExact(arguments, packs, workspace, control, time, batch);
            } catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("generated composite Scan invocation failed", failure); }
            finally { if (lease != null) lease.close(); }
        }
    }

    /**
     * Revalidates complete borrowed capacities and every logical produced length before Scan writes.
     * Composite lengths use the original boundary's offset/stride; bound segments are not rebased.
     * Cold-bound full-capacity segments and typed wrappers are reused, including on failure retry.
     * @param state sole invocation owner and authenticated resource geometry
     * @param owners exact typed boundary owners
     * @param scratch declared workspace owner, or null on an old no-workspace recipe
     * @param full retained original full selected capacities
     * @param arguments exact typed boundary segments used by the generated entry
     * @param time fixed maximum accepted sequence length
     * @param rows complete length-vector element count
     * @throws IllegalArgumentException if a capacity, access, alias or produced length is invalid
     */
    private void validateProducedLengths(RunState state, CpuBufferRepresentation[] owners,
            CpuContiguousWorkspace scratch, MemorySegment[] full, MemorySegment[] arguments,
            long time, long rows) {
        if (!facts.ir().lengthAware()) return;
        for (int p = 0; p < owners.length; p++) if (!acceptsBufferRepresentation(state, p, owners[p]))
            throw new IllegalArgumentException("length-aware scan boundary changed after binding");
        if (scratch != null && !acceptsWorkspaceRepresentation(state, 0, scratch))
            throw new IllegalArgumentException("length-aware scan workspace changed after binding");
        for (int l = 0; l < full.length; l++) for (int r = l + 1; r < full.length; r++)
            if ((l >= reads || r >= reads) && overlaps(full[l], full[r]))
                throw new IllegalArgumentException("length-aware scan writable overlap after producers");
        var lengths = arguments[facts.ir().lengthRead()];
        long base = 0, stride = 1;
        if (facts.ir().composition().isPresent()) {
            var composition = facts.ir().composition().orElseThrow();
            var mapping = composition.boundaryMappings().get(composition.lengthRead());
            base = mapping.baseElementOffset();
            stride = Math.addExact(mapping.strideConstants().getFirst(), Math.multiplyExact(
                    facts.ir().symbolic() ? state.geometry().batchSize().orElseThrow() : 0,
                    mapping.strideBatchFactors().getFirst()));
        }
        for (long row = 0; row < rows; row++) {
            long element = Math.addExact(base, Math.multiplyExact(row, stride));
            long length = lengths.get(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,
                    Math.multiplyExact(element, Long.BYTES));
            if (length < 0 || length > time)
                throw new IllegalArgumentException("length-aware scan length outside [0,T] at row " + row);
        }
    }


    /**
     * Counts fixed feature cells in an admitted shared value, never multiplying by B.
     * @param region exact access proof
     * @param v shared value ordinal
     * @return checked fixed cell count
     * @throws ArithmeticException if a malformed proof overflows
     */
    private static long fixedCells(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int v) {
        long cells = 1;
        for (long extent : region.accesses().get(v).extentConstants()) cells = Math.multiplyExact(cells,extent);
        return cells;
    }

    /**
     * Checks complete selected capacities, retaining legal empty and read/read cases.
     * @param a non-null accessible selected segment
     * @param b non-null accessible selected segment
     * @return whether the nonempty physical capacities overlap
     */
    private static boolean overlaps(MemorySegment a, MemorySegment b) {
        return a.byteSize() != 0 && b.byteSize() != 0 && a.asOverlappingSlice(b).isPresent();
    }

    /** All invocation-mutated bindings belong to one RunState, never shared prepared state. */
    private final class Invocation extends BoundInvocation {
        private final RunState state;
        private final CpuBufferRepresentation[] owners;
        private final CpuContiguousWorkspace scratch;
        private final MemorySegment[] segments;
        private final long[] geometry;
        private final MemorySegment[] fullSegments;
        private final StepDispatch dispatch;
        private final int demand;
        private final long batch;

        /**
         * Retains fresh cold-bound arrays without transferring caller storage ownership.
         * @param state sole logical run owner
         * @param owners direct typed access owners
         * @param scratch run workspace or null
         * @param segments fresh typed argument array including optional scratch
         * @param fullSegments cold-bound full capacities retained for post-producer length checks
         * @param geometry fresh run-owned time and coordinate counts
         * @param dispatch exclusively run-owned row dispatcher, or null for inline emission
         * @param demand fixed positive outer reservation demand
         * @param batch authenticated invocation B before selected geometry extension
         */
        private Invocation(RunState state, CpuBufferRepresentation[] owners, CpuContiguousWorkspace scratch,
                MemorySegment[] segments, MemorySegment[] fullSegments, long[] geometry,StepDispatch dispatch,int demand,long batch) {
            super(state); this.state = state; this.owners = owners; this.scratch = scratch;
            this.segments = segments; this.geometry = geometry;
            this.fullSegments = fullSegments;
            this.dispatch = dispatch; this.demand = demand; this.batch = batch;
        }

        /**
         * Rechecks liveness and performs exactly one generated entry invocation, not one per step.
         * One whole-entry reservation covers max(1,N) participants for selected parallel rows,
         * otherwise one inline participant. It is lent only synchronously and released on success
         * or failure after all started workers quiesce; the private dispatch binding is cleared
         * before release. Generated runtime failures and errors propagate without output rollback.
         * @throws IllegalArgumentException if a borrowed carrier is no longer accessible, a
         *     length-mode typed span/alignment/overlap changed, or any produced length is
         *     outside [0,T], or selected workers are closed or cannot access a carrier/workspace;
         *     length-mode validation precedes all scan-local writes
         * @throws CpuConcurrencyBudget.CpuCoordinationException if whole-entry admission is interrupted;
         *     the thread's interrupt status is restored
         * @throws CpuWorkerGroup.CpuParallelExecutionException if synchronous row dispatch is
         *     cancelled/interrupted or wraps a checked worker failure, after joining started work
         * @throws IllegalStateException if the generated invocation throws a checked failure
         */
        @Override protected void executeBound() {
            CpuConcurrencyBudget.Lease lease = budget == null ? null : budget.acquire(demand);
            try {
                for (var owner : owners) if (!owner.isAccessible())
                    throw new IllegalArgumentException("general scan representation is no longer accessible");
                if (scratch != null && !scratch.isAccessible())
                    throw new IllegalArgumentException("general scan scratch is no longer accessible");
                if (dispatch != null && demand >= 2 && !workerGroup.isOpen())
                    throw new IllegalArgumentException("Scan row workers are closed");
                if (dispatch != null && demand >= 2) for (var segment : segments)
                    if (!workerGroup.workersCanAccess(segment))
                        throw new IllegalArgumentException("Scan row carriers are not worker-accessible");
                if (facts.ir().lengthAware())
                    validateProducedLengths(state, owners, scratch, fullSegments, segments, geometry[0],
                            facts.ir().symbolic() ? batch : facts.ir().stages().orElseThrow().batch());
                if (dispatch == null) entry.invokeExact(segments, geometry, 0L, geometry[1]);
                else {
                    dispatch.reservation = lease;
                    entry.invokeExact(segments,geometry,dispatch,0L,geometry[1]);
                }
            } catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("generated scan invocation failed", failure); }
            finally {
                if (dispatch != null) dispatch.reservation = null;
                if (lease != null) lease.close();
            }
        }
    }
}
