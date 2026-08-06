package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
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

    /**
     * Creates a direct four-boundary recipe for one exact half-open logical range.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered selections for inputs {@code a}, {@code b}, {@code c},
     *     and the output; copied by the superclass
     * @param artifact non-null verified artifact matching {@code carrierPattern}
     * @param bindings non-null full-range normalized boundary bindings; copied defensively
     * @param carrierPattern non-null ordered direct carrier pattern; copied defensively
     * @param start non-negative inclusive logical element bound
     * @param end exclusive logical element bound no greater than the iteration element count
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if counts, specialization, or range disagree
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
     * @param selections non-null ordered selections for inputs {@code a}, {@code b}, {@code c},
     *     and output; copied by the superclass
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
     * @param selections non-null four ordered boundary selections
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
                workspaceSelection);
    }

    /**
     * Creates the complete recipe with separate original and adjusted generated carrier patterns.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null four ordered boundary selections
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
        super(memoryPlan, selections, workspaceSelection.map(List::of).orElseGet(List::of), List.of(BufferAccess.READ_ONLY,
                BufferAccess.READ_ONLY, BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        if (selections.size() != 4) throw new IllegalArgumentException("four buffers required");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
        this.bindings = List.copyOf(bindings);
        this.carrierPattern = List.copyOf(carrierPattern);
        this.generatedCarrierPattern = List.copyOf(generatedCarrierPattern);
        if (this.bindings.size() != 4 || this.carrierPattern.size() != 4
                || this.generatedCarrierPattern.size() != 4
                || !artifact.specialization().carrierPattern().equals(this.generatedCarrierPattern)) {
            throw new IllegalArgumentException("binding and specialization patterns must agree");
        }
        long count = this.bindings.getFirst().elementCount();
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
        if (materialization.isPresent() != workspaceSelection.isPresent()) {
            throw new IllegalArgumentException("materialization and workspace selection must agree");
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
     * Returns the first boundary's geometry adjusted to this recipe's range.
     *
     * @return a new immutable ranged binding
     */
    public CpuAccessPlan.Binding binding() { return ranged(bindings.getFirst()); }
    /**
     * Returns the exact ranged geometry for every ordered boundary.
     *
     * @return a new immutable four-entry list whose bindings share this recipe's range
     */
    public List<CpuAccessPlan.Binding> accessBindings() {
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
        var selections = new ArrayList<BufferSelection>(4);
        for (int i = 0; i < 4; i++) selections.add(bufferSelection(i));
        return new CpuPreparedExecutable(memoryPlan(), selections, artifact, bindings,
                carrierPattern, generatedCarrierPattern, rangeStart, rangeEnd, selectedRangeCount,
                minimumElementsPerWorker, workerGroup, materialization,
                workspaceSelection);
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
                || cpu.dataType() != io.github.pho001.synaptik.model.datatype.DataType.FLOAT64) return false;
        var entry = memoryPlan().buffers().get(bufferSelection(index).bufferIndex());
        if (cpu.byteSize() != entry.byteSize()) return false;
        try {
            CpuBufferArgument argument = cpu.argument();
            CarrierAccess actual = argument instanceof CpuBufferArgument.Doubles
                    ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.MEMORY_SEGMENT;
            boolean aligned = !(argument instanceof CpuBufferArgument.Segment segment)
                    || segment.segment().address() % entry.byteAlignment() == 0;
            return actual == carrierPattern.get(index) && aligned
                    && (index < 3 || !argument.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        if (index != 0 || materialization.isEmpty()
                || !(representation instanceof CpuContiguousWorkspace workspace)
                || !workspace.isAccessible()) return false;
        var copy = materialization.orElseThrow();
        return workspace.byteSize() == copy.byteCount()
                && workspace.byteAlignment() == copy.byteAlignment()
                && workspace.writableSegment().address() % copy.byteAlignment() == 0;
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (workspaces.length != (materialization.isPresent() ? 1 : 0)) {
            throw new IllegalArgumentException("workspace count disagrees with materialization");
        }
        var arguments = new ArrayList<CpuBufferArgument>(4);
        for (BufferRepresentation buffer : buffers) {
            arguments.add(((CpuBufferRepresentation) buffer).argument());
        }
        for (int input = 0; input < 3; input++) {
            CpuAccessPlan.Binding inputBinding = materialization.isPresent()
                    && materialization.orElseThrow().sourceBoundaryIndex() == input
                    ? materialization.orElseThrow().sourceBinding() : bindings.get(input);
            if (overlaps(arguments.get(input), ranged(inputBinding),
                    arguments.get(3), ranged(bindings.get(3)))) {
                throw new IllegalArgumentException("output accessed span must not overlap an input");
            }
        }
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
        long length = end - start;
        int chunkCount = length == 0 ? 0 : Math.min(selectedRangeCount,
                Math.toIntExact(1 + (length - 1) / minimumElementsPerWorker));
        if (chunkCount <= 1) {
            KernelCall call = callFor(artifact.entryPoint(), arguments,
                    geometry(arguments, start, end), start, end);
            return new Invocation(state, copyCall, call, null);
        }
        CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[chunkCount];
        long quotient = length / chunkCount;
        long remainder = length % chunkCount;
        long chunkStart = start;
        for (int index = 0; index < chunkCount; index++) {
            long chunkEnd = chunkStart + quotient + (index < remainder ? 1 : 0);
            KernelCall call = callFor(artifact.entryPoint(), arguments,
                    geometry(arguments, chunkStart, chunkEnd), chunkStart, chunkEnd);
            calls[index] = call::invoke;
            chunkStart = chunkEnd;
        }
        return new Invocation(state, copyCall, null, calls);
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

    private long[] geometry(List<CpuBufferArgument> arguments, long rangeStart, long rangeEnd) {
        int rank = bindings.getFirst().plan().iterationRank();
        long[] geometry = new long[2 * rank + 4 + 4 * rank + 8];
        CpuAccessPlan.Binding first = ranged(bindings.getFirst(), rangeStart, rangeEnd);
        for (int axis = 0; axis < rank; axis++) {
            geometry[axis] = first.extents().get(axis);
            geometry[rank + axis] = first.startCoordinates().get(axis);
        }
        for (int value = 0; value < 4; value++) {
            CpuAccessPlan.Binding binding = ranged(bindings.get(value), rangeStart, rangeEnd);
            long carrierBase = arguments.get(value).byteOffset() / Double.BYTES;
            geometry[2 * rank + value] = Math.addExact(carrierBase, binding.startAddress());
            for (int axis = 0; axis < rank; axis++) geometry[2 * rank + 4 + value * rank + axis]
                    = binding.effectiveStrides().get(axis);
            long innerPosition = 0;
            long innerSize = 1;
            for (int axis = rank - binding.plan().contiguousSuffix(); axis < rank; axis++) {
                innerPosition = Math.addExact(Math.multiplyExact(innerPosition,
                        binding.extents().get(axis)), binding.startCoordinates().get(axis));
                innerSize = Math.multiplyExact(innerSize, binding.extents().get(axis));
            }
            geometry[2 * rank + 4 + 4 * rank + value] = innerPosition;
            geometry[2 * rank + 4 + 4 * rank + 4 + value] = innerSize;
        }
        return geometry;
    }

    private static boolean overlaps(CpuBufferArgument left, CpuAccessPlan.Binding leftBinding,
            CpuBufferArgument right, CpuAccessPlan.Binding rightBinding) {
        if (leftBinding.elementCount() == 0 || rightBinding.elementCount() == 0) return false;
        Object leftCarrier = left instanceof CpuBufferArgument.Doubles d ? d.carrier()
                : ((CpuBufferArgument.Segment) left).segment();
        Object rightCarrier = right instanceof CpuBufferArgument.Doubles d ? d.carrier()
                : ((CpuBufferArgument.Segment) right).segment();
        if (leftCarrier instanceof MemorySegment a && rightCarrier instanceof MemorySegment b
                && a.asOverlappingSlice(b).isEmpty()) return false;
        if (leftCarrier != rightCarrier && !(leftCarrier instanceof MemorySegment
                && rightCarrier instanceof MemorySegment)) return false;
        long leftStart = Math.addExact(carrierBase(left),
                Math.multiplyExact(leftBinding.accessedElementStart(), Double.BYTES));
        long leftEnd = Math.addExact(carrierBase(left),
                Math.multiplyExact(leftBinding.accessedElementEnd(), Double.BYTES));
        long rightStart = Math.addExact(carrierBase(right),
                Math.multiplyExact(rightBinding.accessedElementStart(), Double.BYTES));
        long rightEnd = Math.addExact(carrierBase(right),
                Math.multiplyExact(rightBinding.accessedElementEnd(), Double.BYTES));
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static long carrierBase(CpuBufferArgument argument) {
        return argument instanceof CpuBufferArgument.Doubles
                ? argument.byteOffset() : ((CpuBufferArgument.Segment) argument).segment().address();
    }

    @FunctionalInterface private interface KernelCall { void invoke() throws Throwable; }

    private static KernelCall callFor(MethodHandle h, List<CpuBufferArgument> a, long[] g,
            long start, long end) {
        int mask = 0;
        for (int i = 0; i < 4; i++) if (a.get(i) instanceof CpuBufferArgument.Doubles) mask |= 1 << i;
        Object x0 = carrier(a.get(0)), x1 = carrier(a.get(1)), x2 = carrier(a.get(2)), x3 = carrier(a.get(3));
        return switch (mask) {
            case 0 -> { var p0=(MemorySegment)x0; var p1=(MemorySegment)x1; var p2=(MemorySegment)x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 1 -> { var p0=(double[])x0; var p1=(MemorySegment)x1; var p2=(MemorySegment)x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 2 -> { var p0=(MemorySegment)x0; var p1=(double[])x1; var p2=(MemorySegment)x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 3 -> { var p0=(double[])x0; var p1=(double[])x1; var p2=(MemorySegment)x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 4 -> { var p0=(MemorySegment)x0; var p1=(MemorySegment)x1; var p2=(double[])x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 5 -> { var p0=(double[])x0; var p1=(MemorySegment)x1; var p2=(double[])x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 6 -> { var p0=(MemorySegment)x0; var p1=(double[])x1; var p2=(double[])x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 7 -> { var p0=(double[])x0; var p1=(double[])x1; var p2=(double[])x2; var p3=(MemorySegment)x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 8 -> { var p0=(MemorySegment)x0; var p1=(MemorySegment)x1; var p2=(MemorySegment)x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 9 -> { var p0=(double[])x0; var p1=(MemorySegment)x1; var p2=(MemorySegment)x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 10 -> { var p0=(MemorySegment)x0; var p1=(double[])x1; var p2=(MemorySegment)x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 11 -> { var p0=(double[])x0; var p1=(double[])x1; var p2=(MemorySegment)x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 12 -> { var p0=(MemorySegment)x0; var p1=(MemorySegment)x1; var p2=(double[])x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 13 -> { var p0=(double[])x0; var p1=(MemorySegment)x1; var p2=(double[])x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 14 -> { var p0=(MemorySegment)x0; var p1=(double[])x1; var p2=(double[])x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            case 15 -> { var p0=(double[])x0; var p1=(double[])x1; var p2=(double[])x2; var p3=(double[])x3; yield () -> { h.invokeExact(p0,p1,p2,p3,g,start,end); }; }
            default -> throw new AssertionError(mask);
        };
    }

    private static Object carrier(CpuBufferArgument argument) {
        return argument instanceof CpuBufferArgument.Doubles d ? d.carrier()
                : ((CpuBufferArgument.Segment) argument).segment();
    }

    private final class Invocation extends BoundInvocation {
        private final CopyCall copy;
        private final KernelCall call;
        private final CpuWorkerGroup.RangeCall[] calls;
        Invocation(RunState state, CopyCall copy, KernelCall call, CpuWorkerGroup.RangeCall[] calls) {
            super(state); this.copy = copy; this.call = call; this.calls = calls;
        }
        @Override protected void executeBound() {
            try {
                if (copy != null) copy.invoke();
                if (calls == null) call.invoke();
                else workerGroup.execute(calls);
            }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException("generated CPU invocation failed", failure); }
        }
    }
}
