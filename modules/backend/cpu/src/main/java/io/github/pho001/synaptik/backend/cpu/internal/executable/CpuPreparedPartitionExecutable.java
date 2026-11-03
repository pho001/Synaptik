package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuWorkspaceRecipeProvider;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

/**
 * CPU-private atomic recipe for a fixed, topologically ordered partition computation-unit list.
 * The recipe is immutable and reusable across runs; it retains child recipes and diagnostic
 * dependency indices but performs no dynamic dependency scheduling. Cold binding validates every
 * selected CPU buffer and workspace, rejects any overlap involving a write, and binds all children
 * before returning an invocation. Hot execution then invokes those already-bound children strictly
 * in list order, stopping at the first failure. An explicitly selected representation candidate
 * contributes at most two generated affine-copy units. Binding validates all copy resources
 * before mutation; execution invokes each copy exactly once before all represented consumers, so
 * compatible repeated and cross-unit uses share the completed workspace. A segment-form copy
 * source binds the exact selected segment even when its primitive heap carrier is observable;
 * explicit array-form copies keep their classified carrier. A child may be the
 * narrow native MATMUL recipe or a oneDNN child with run-owned native invocation workspaces.
 * Native scratch participates in the same alias checks; generated copy targets remain ordinary
 * contiguous workspaces. Runtime owns the surrounding atomic validity transition and each run's
 * resources and never selects a candidate.
 * Complete batch mode authenticates each selection against the exact run geometry. Fixed child
 * units retain their own fixed declarations; nonfixed boundaries must select an actual child
 * binding, whose genuine dynamic route is checked before the composite invocation is returned.
 * Fixed generated copy units retain their original affine and workspace facts.
 */
public final class CpuPreparedPartitionExecutable extends PreparedExecutable
        implements CpuWorkspaceRecipeProvider {
    private final List<PreparedExecutable> children;
    private final List<List<Integer>> dependencies;
    private final List<CopyUnit> copies;
    private final List<WorkspaceSelection> selectedWorkspaces;

    /**
     * Creates one immutable fixed partition composite.
     *
     * @param memoryPlan non-null exact shared memory plan retained by every child
     * @param buffers non-null deduplicated partition buffer selections; copied by the superclass
     * @param workspaces non-null exact unit-local workspace selections in unit order; copied by
     *     the superclass
     * @param accesses non-null read-only or write-only declarations aligned with {@code buffers}
     * @param children non-null, non-empty stable topological child list; copied defensively
     * @param dependencies non-null direct producer-index lists aligned with {@code children};
     *     every inner list is copied defensively and may be empty
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if list cardinalities, access kinds, child memory-plan
     *     identity, or strictly-earlier dependency indices disagree
     */
    public CpuPreparedPartitionExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            List<BufferAccess> accesses, List<? extends PreparedExecutable> children,
            List<List<Integer>> dependencies) {
        this(memoryPlan, buffers, workspaces, accesses, List.of(), children, dependencies);
    }

    /**
     * Creates an atomic composite with generated copy units followed by computation units.
     *
     * @param memoryPlan non-null exact shared memory plan retained by every child
     * @param buffers non-null deduplicated partition buffer selections; copied by the superclass
     * @param workspaces non-null exact unit-local and copy workspace selections; copied by the
     *     superclass
     * @param accesses non-null read-only or write-only declarations aligned with {@code buffers}
     * @param copies non-null ordered zero, one, or two explicit generated materialization units;
     *     copied defensively
     * @param children non-null, non-empty stable topological child list; copied defensively
     * @param dependencies non-null direct producer-index lists aligned with {@code children};
     *     every inner list is copied defensively and may be empty
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if list cardinalities, access kinds, copy facts, child
     *     memory-plan identity, or strictly-earlier dependency indices disagree
     */
    public CpuPreparedPartitionExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            List<BufferAccess> accesses, List<CopyUnit> copies,
            List<? extends PreparedExecutable> children, List<List<Integer>> dependencies) {
        super(memoryPlan, buffers, workspaces, accesses);
        selectedWorkspaces = List.copyOf(workspaces);
        this.copies = List.copyOf(copies);
        this.children = List.copyOf(children);
        this.dependencies = dependencies.stream().map(List::copyOf).toList();
        if (this.children.isEmpty() || this.children.size() != this.dependencies.size()
                || accesses.stream().anyMatch(access -> access == BufferAccess.READ_WRITE)) {
            throw new IllegalArgumentException("CPU partition composite facts disagree");
        }
        for (int index = 0; index < this.children.size(); index++) {
            PreparedExecutable child = this.children.get(index);
            if (child.memoryPlan() != memoryPlan) {
                throw new IllegalArgumentException("CPU child memory plan disagrees");
            }
            int unitIndex = index;
            if (this.dependencies.get(index).stream().distinct().count()
                    != this.dependencies.get(index).size()
                    || this.dependencies.get(index).stream().anyMatch(value -> value < 0
                        || value >= unitIndex)) {
                throw new IllegalArgumentException("CPU child dependencies are not topological");
            }
        }
        if (this.copies.size() > 2) throw new IllegalArgumentException(
                "CPU partition permits at most two copy units");
    }

    /** Resolves one cold child recipe, rejecting ambiguous native workspace ownership. */
    @Override public Optional<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreator(int workspaceIndex) {
        PreparedRepresentationPlan.WorkspaceCreator selected = null;
        for (PreparedExecutable child : children) {
            if (!(child instanceof CpuWorkspaceRecipeProvider provider)) continue;
            var candidate = provider.workspaceCreator(workspaceIndex);
            if (candidate.isEmpty()) continue;
            if (selected != null) throw new IllegalArgumentException(
                    "CPU partition children claim the same workspace");
            selected = candidate.orElseThrow();
        }
        return Optional.ofNullable(selected);
    }

    /**
     * One explicit generated affine-copy unit and its outer resource positions.
     *
     * @param plan non-null complete copy plan
     * @param artifact non-null generated affine-copy artifact matching {@code plan}
     * @param bufferPosition non-negative outer source-buffer position
     * @param workspacePosition non-negative outer destination-workspace position
     */
    public record CopyUnit(CpuMaterializationPlan plan, CpuGeneratedKernel artifact,
            int bufferPosition, int workspacePosition) {
        /**
         * Validates the exact copy artifact and non-negative outer positions.
         *
         * @throws NullPointerException if {@code plan} or {@code artifact} is {@code null}
         * @throws IllegalArgumentException if a position is negative or artifact specialization
         *     differs from the plan
         */
        public CopyUnit {
            Objects.requireNonNull(plan, "plan");
            Objects.requireNonNull(artifact, "artifact");
            if (bufferPosition < 0 || workspacePosition < 0
                    || !artifact.specialization().equals(plan.copySpecialization())) {
                throw new IllegalArgumentException("CPU generated copy unit facts disagree");
            }
        }
    }

    /**
     * Returns the retained child recipes, including direct recurrent and native children.
     *
     * @return the non-null immutable, non-empty child list in execution order
     */
    public List<PreparedExecutable> children() {
        return children;
    }

    /** Returns diagnostic topology facts; hot execution does not schedule from this graph.
     * @return the non-null immutable direct producer-index lists aligned with {@link #children()}
     */
    public List<List<Integer>> dependencies() { return dependencies; }

    /**
     * Checks complete-plan geometry without confusing a dynamic child with a fixed sibling.
     * @param state exact open run with authenticated sizing
     * @param index outer boundary selection position
     * @param representation selected CPU boundary, or an incompatible representation to reject
     * @return whether physical bytes/alignment agree and any nonfixed selection reaches a child
     */
    @Override protected boolean acceptsBufferRepresentation(RunState state, int index,
            BufferRepresentation representation) {
        if (!state.memoryPlan().requiresBatchSize()) return super.acceptsBufferRepresentation(state, index, representation);
        if (!acceptsBufferRepresentation(index, representation)
                || state.memoryPlan() != memoryPlan() || state.geometry().memoryPlan() != memoryPlan()) return false;
        var cpu = (CpuBufferRepresentation) representation;
        int position = bufferSelection(index).bufferIndex();
        var declaration = memoryPlan().buffers().get(position);
        if (cpu.byteSize() != state.geometry().bufferByteSize(position)) return false;
        var fixed = declaration.size().fixedByteSize();
        if (fixed.isPresent()) return cpu.acceptsFixedGeometry(state, memoryPlan(), position, fixed.getAsLong());
        if (!hasChildSelection(bufferSelection(index))) return false;
        try {
            var argument = cpu.argument();
            return argument.byteOffset() % cpu.dataType().byteWidth() == 0
                    && argument.byteSize() % cpu.dataType().byteWidth() == 0
                    && cpu.segment().address() % declaration.byteAlignment() == 0;
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    /**
     * Preserves dynamic child scratch while authenticating fixed scratch declarations explicitly.
     * @param state exact open run with authenticated sizing
     * @param index outer workspace selection position
     * @param representation selected CPU scratch, or an incompatible representation to reject
     * @return whether declared physical geometry and writable lifetime agree
     */
    @Override protected boolean acceptsWorkspaceRepresentation(RunState state, int index,
            WorkspaceRepresentation representation) {
        if (!state.memoryPlan().requiresBatchSize()) return super.acceptsWorkspaceRepresentation(state, index, representation);
        if (!acceptsWorkspaceRepresentation(index, representation)
                || state.memoryPlan() != memoryPlan() || state.geometry().memoryPlan() != memoryPlan()) return false;
        int position = selectedWorkspaces.get(index).workspaceIndex();
        var declaration = memoryPlan().workspaces().get(position);
        var fixed = declaration.size().fixedByteSize();
        if (representation instanceof CpuOneDnnWorkspace nativeWorkspace) {
            if (fixed.isEmpty() || fixed.getAsLong() != nativeWorkspace.byteSize()
                    || state.geometry().workspaceByteSize(position) != nativeWorkspace.byteSize()
                    || declaration.byteAlignment() > 64) return false;
            try {
                var segment = nativeWorkspace.writableSegment();
                return !segment.isReadOnly() && segment.address() % declaration.byteAlignment() == 0;
            } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
        }
        var cpu = (CpuContiguousWorkspace) representation;
        if (fixed.isPresent()) return CpuBufferRepresentation.acceptsFixedWorkspaceGeometry(
                state, memoryPlan(), position, fixed.getAsLong(), cpu);
        if (!hasDynamicWorkspaceSelection(selectedWorkspaces.get(index))) return false;
        try {
            var segment = cpu.writableSegment();
            return cpu.byteSize() == state.geometry().workspaceByteSize(position)
                    && cpu.byteAlignment() >= declaration.byteAlignment() && !segment.isReadOnly()
                    && segment.address() % declaration.byteAlignment() == 0;
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    /**
     * Requires an exact scratch selection on a genuine dynamic child, including nested composites.
     * @param selection dense workspace position selected by this container
     * @return whether a retained dynamic child authenticates this exact workspace at binding
     */
    private boolean hasDynamicWorkspaceSelection(WorkspaceSelection selection) {
        for (PreparedExecutable child : children) {
            if (child instanceof CpuPreparedExecutable ordinary
                    && ordinary.hasDynamicWorkspaceSelection(selection)) return true;
            if (child instanceof CpuGeneralScanPreparedExecutable scan
                    && scan.hasDynamicWorkspaceSelection(selection)) return true;
            if (child instanceof CpuPreparedPartitionExecutable nested
                    && nested.hasDynamicWorkspaceSelection(selection)) return true;
        }
        return false;
    }

    /**
     * Finds an exact child resource selection, not just a matching logical buffer position.
     * @param selection outer buffer and representation positions
     * @return whether an already-retained child authenticates this exact selection at binding
     */
    private boolean hasChildSelection(BufferSelection selection) {
        for (PreparedExecutable child : children)
            for (int p = 0; p < child.bufferSelectionCount(); p++)
                if (child.bufferSelection(p).equals(selection)) return true;
        return false;
    }

    @Override protected boolean acceptsBufferRepresentation(int index,
            BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible())
            return false;
        try {
            return bufferAccess(index) != BufferAccess.WRITE_ONLY || !cpu.argument().readOnly();
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        if (representation instanceof CpuOneDnnWorkspace workspace) {
            for (CopyUnit copy : copies) if (copy.workspacePosition() == index) return false;
            return workspace.isAccessible();
        }
        return representation instanceof CpuContiguousWorkspace workspace && workspace.isAccessible();
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        var arguments = new CpuBufferArgument[buffers.length];
        for (int index = 0; index < buffers.length; index++)
            arguments[index] = ((CpuBufferRepresentation) buffers[index]).argument();
        for (int left = 0; left < arguments.length; left++) {
            for (int right = left + 1; right < arguments.length; right++) {
                if (bufferAccess(left) == BufferAccess.READ_ONLY
                        && bufferAccess(right) == BufferAccess.READ_ONLY) continue;
                if (overlaps(arguments[left], arguments[right])) {
                    throw new IllegalArgumentException(
                            "CPU partition writes must not overlap another selected span");
                }
            }
        }
        for (int workspace = 0; workspace < workspaces.length; workspace++) {
            MemorySegment segment = workspaceSegment(workspaces[workspace]);
            for (CpuBufferArgument argument : arguments)
                if (segment.asOverlappingSlice(segment(argument)).isPresent())
                    throw new IllegalArgumentException(
                            "CPU partition workspace must not overlap a buffer");
            for (int other = workspace + 1; other < workspaces.length; other++)
                if (workspaces[workspace] == workspaces[other]
                        || segment.asOverlappingSlice(workspaceSegment(workspaces[other])).isPresent())
                    throw new IllegalArgumentException(
                            "CPU partition workspaces must be distinct and disjoint");
        }
        var copyCalls = new ArrayList<GeneratedCopyCall>(copies.size());
        for (CopyUnit copy : copies) {
            CpuBufferRepresentation sourceBuffer = (CpuBufferRepresentation)
                    buffers[copy.bufferPosition()];
            CpuBufferArgument source = copy.plan().sourceCarrier()
                    == io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                    ? sourceBuffer.segmentArgument() : arguments[copy.bufferPosition()];
            if (carrierAccess(source) != copy.plan().sourceCarrier()) throw new IllegalArgumentException(
                    "CPU copy source carrier changed after analysis");
            CpuContiguousWorkspace target = (CpuContiguousWorkspace)
                    workspaces[copy.workspacePosition()];
            if (state.memoryPlan().requiresBatchSize()
                    && (!sourceBuffer.acceptsFixedGeometry(state, memoryPlan(),
                            bufferSelection(copy.bufferPosition()).bufferIndex(),
                            Math.multiplyExact(copy.plan().sourceBinding().referencedElementSpan(),
                                    (long) copy.plan().dataType().byteWidth()))
                        || !CpuBufferRepresentation.acceptsFixedWorkspaceGeometry(state, memoryPlan(),
                            selectedWorkspaces.get(copy.workspacePosition()).workspaceIndex(),
                            copy.plan().byteCount(), target))) {
                throw new IllegalArgumentException("CPU fixed copy declarations disagree with retained geometry");
            }
            MemorySegment segment = target.writableSegment();
            if (segment.asOverlappingSlice(segment(source)).isPresent()) throw new IllegalArgumentException(
                    "CPU materialization workspace overlaps its source");
            long[] geometry = copy.plan().affineAddressPairs();
            long base = source.byteOffset() / copy.plan().dataType().byteWidth();
            for (int index = 0; index < geometry.length; index += 2)
                geometry[index] = Math.addExact(geometry[index], base);
            copyCalls.add(copyCall(copy, source, segment, geometry));
        }
        List<GeneratedCopyCall> generatedCopies = List.copyOf(copyCalls);
        var bound = new ArrayList<BoundInvocation>(children.size());
        for (PreparedExecutable child : children) bound.add(child.bind(state));
        List<BoundInvocation> invocations = List.copyOf(bound);
        return new BoundInvocation(state) {
            @Override protected void executeBound() {
                try {
                    for (GeneratedCopyCall copy : generatedCopies) copy.invoke();
                    for (BoundInvocation invocation : invocations) invocation.execute();
                } catch (RuntimeException | Error failure) { throw failure; }
                catch (Throwable failure) { throw new IllegalStateException(
                        "generated CPU copy invocation failed", failure); }
            }
        };
    }

    private static MemorySegment workspaceSegment(WorkspaceRepresentation representation) {
        return switch (representation) {
            case CpuContiguousWorkspace workspace -> workspace.writableSegment();
            case CpuOneDnnWorkspace workspace -> workspace.writableSegment();
            default -> throw new IllegalArgumentException("CPU partition workspace is incompatible");
        };
    }

    @FunctionalInterface private interface GeneratedCopyCall { void invoke() throws Throwable; }

    private static GeneratedCopyCall copyCall(CopyUnit copy, CpuBufferArgument source,
            MemorySegment target, long[] geometry) {
        MethodHandle handle = copy.artifact().entryPoint();
        handle = MethodHandles.insertArguments(handle, 0, carrier(source), target,
                geometry);
        MethodHandle targetHandle = handle;
        long end = copy.plan().elementCount();
        return () -> invokeVoid(targetHandle, 0L, end);
    }

    private static void invokeVoid(MethodHandle handle, long start, long end) throws Throwable {
        handle.invokeExact(start, end);
    }

    private static Object carrier(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> value.carrier();
            case CpuBufferArgument.Floats value -> value.carrier();
            case CpuBufferArgument.Shorts value -> value.carrier();
            case CpuBufferArgument.Ints value -> value.carrier();
            case CpuBufferArgument.Longs value -> value.carrier();
            case CpuBufferArgument.Bytes value -> value.carrier();
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess
            carrierAccess(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case CpuBufferArgument.Floats ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case CpuBufferArgument.Shorts ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case CpuBufferArgument.Ints ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case CpuBufferArgument.Longs ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case CpuBufferArgument.Bytes ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
            case CpuBufferArgument.Segment ignored -> io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        };
    }

    private static boolean overlaps(CpuBufferArgument left, CpuBufferArgument right) {
        if (left.byteSize() == 0 || right.byteSize() == 0) return false;
        return segment(left).asOverlappingSlice(segment(right)).isPresent();
    }

    private static MemorySegment segment(CpuBufferArgument argument) {
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
}
