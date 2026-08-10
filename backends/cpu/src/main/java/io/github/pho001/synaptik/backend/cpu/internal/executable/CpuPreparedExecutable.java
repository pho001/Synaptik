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

    /**
     * Creates a direct derived-boundary recipe for one exact half-open logical range.
     *
     * @param memoryPlan non-null exact plan against which selections are resolved
     * @param selections non-null ordered derived-boundary selections; copied by the superclass
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
        super(memoryPlan, selections, workspaceSelection.map(List::of).orElseGet(List::of),
                accesses(selections.size()));
        if (selections.size() < 2) throw new IllegalArgumentException("at least two buffers required");
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
        this.affineCopy = affineAddressPairs != null;
        this.affineAddressPairs = affineAddressPairs == null ? new long[0] : affineAddressPairs.clone();
        if (affineCopy && (this.affineAddressPairs.length % 2 != 0
                || this.affineAddressPairs.length < Math.multiplyExact(end, 2))) {
            throw new IllegalArgumentException("affine address pairs must cover the selected copy range");
        }
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
     * @return a new immutable list whose entries share this recipe's range and remain in boundary
     *     order
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
        var selections = new ArrayList<BufferSelection>(bindings.size());
        for (int i = 0; i < bindings.size(); i++) selections.add(bufferSelection(i));
        return new CpuPreparedExecutable(memoryPlan(), selections, artifact, bindings,
                carrierPattern, generatedCarrierPattern, rangeStart, rangeEnd, selectedRangeCount,
                minimumElementsPerWorker, workerGroup, materialization,
                workspaceSelection, affineCopy ? affineAddressPairs : null);
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
            return actual == carrierPattern.get(index) && aligned
                    && (index < bindings.size() - 1 || !argument.readOnly());
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
        var arguments = new ArrayList<CpuBufferArgument>(bindings.size());
        for (BufferRepresentation buffer : buffers) {
            arguments.add(((CpuBufferRepresentation) buffer).argument());
        }
        int outputIndex = bindings.size() - 1;
        for (int input = 0; input < outputIndex; input++) {
            CpuAccessPlan.Binding inputBinding = materialization.isPresent()
                    && materialization.orElseThrow().sourceBoundaryIndex() == input
                    ? materialization.orElseThrow().sourceBinding() : bindings.get(input);
            boolean overlap = affineCopy
                    ? affineOverlaps(arguments.get(input), arguments.get(outputIndex), input,
                            outputIndex)
                    : overlaps(arguments.get(input), ranged(inputBinding),
                            arguments.get(outputIndex), ranged(bindings.get(outputIndex)));
            if (overlap) {
                throw new IllegalArgumentException("output accessed span must not overlap an input");
            }
        }
        validateCanonicalBooleanInputs(arguments);
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

    private static KernelCall callFor(MethodHandle h, List<CpuBufferArgument> a, long[] g,
            long start, long end) {
        Object[] carriers = a.stream().map(CpuPreparedExecutable::carrier).toArray();
        MethodHandle bound = MethodHandles.insertArguments(h, 0, carriers);
        bound = MethodHandles.insertArguments(bound, 0, (Object) g);
        MethodHandle target = bound;
        return () -> invokeVoid(target, start, end);
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

    private static List<BufferAccess> accesses(int count) {
        if (count < 2) throw new IllegalArgumentException("at least two buffers required");
        var result = new ArrayList<BufferAccess>(count);
        for (int i = 0; i < count - 1; i++) result.add(BufferAccess.READ_ONLY);
        result.add(BufferAccess.WRITE_ONLY);
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
        int output = arguments.size() - 1;
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
            CpuAccessPlan.Binding binding = ranged(bindings.get(index));
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
