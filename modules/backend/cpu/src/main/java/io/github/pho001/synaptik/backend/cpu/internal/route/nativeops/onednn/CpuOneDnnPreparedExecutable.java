package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuWorkspaceRecipeProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable native child using Runtime-owned persistent kernels and run-owned invocation state.
 * Shared assignment already owns all boundary/ingress/scratch slots. Cold binding authenticates
 * native spans and aliases, while hot execution only refreshes declared ingress, executes and
 * waits under one shared CPU-budget permit. It performs no route selection or fallback.
 */
public final class CpuOneDnnPreparedExecutable extends PreparedExecutable
        implements CpuWorkspaceRecipeProvider {
    private final CpuOneDnnRoutePlan.Unit route;
    private final CpuOneDnnKernel kernel;
    private final CpuConcurrencyBudget budget;
    private final List<WorkspaceSelection> selections;
    private final long[] requiredBytes;
    private final boolean[] direct;
    private final DataType[] dataTypes;
    private final PreparedRepresentationPlan.WorkspaceCreator invocationCreator;

    /**
     * Retains the exact selected facts without acquiring native resources or taking their close duty.
     * The finalizer separately transfers kernel ownership through PreparedResource; workspace
     * factories acquire per-run native leases before binding.
     *
     * @param memoryPlan non-null complete shared memory plan after slot assignment
     * @param unitBuffers non-null common boundary selections in projection order; copied by the base
     * @param routeWorkspaces non-null ingress selections followed by invocation scratch; copied
     * @param route non-null retained selected numeric/resource recipe
     * @param kernel non-null freshly finalized kernel borrowing Runtime's persistent ownership;
     *     spec, scratch and implementation must exactly match analysis
     * @param budget non-null shared CPU budget borrowed for synchronous demand-one execution
     * @throws NullPointerException if a required argument or list element is null
     * @throws IllegalArgumentException if kernel facts, boundaries or assigned declarations disagree
     * @throws ArithmeticException if checked boundary origin/span arithmetic overflows
     */
    public CpuOneDnnPreparedExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> unitBuffers, List<WorkspaceSelection> routeWorkspaces,
            CpuOneDnnRoutePlan.Unit route, CpuOneDnnKernel kernel, CpuConcurrencyBudget budget) {
        super(memoryPlan, unitBuffers, routeWorkspaces, accesses(unitBuffers, route));
        this.route = Objects.requireNonNull(route, "route");
        this.kernel = Objects.requireNonNull(kernel, "kernel");
        this.budget = Objects.requireNonNull(budget, "budget");
        selections = List.copyOf(routeWorkspaces);
        if (!kernel.spec().equals(route.projection().spec())
                || kernel.scratchpadBytes() != route.scratchpadBytes()
                || !kernel.implementation().equals(route.implementation())
                || selections.size() != route.workspaceRequirements().size()) {
            throw new IllegalArgumentException("oneDNN finalized kernel or workspaces disagree with route");
        }
        var workspaceIndices = new HashSet<Integer>();
        var requirements = route.workspaceRequirements();
        for (int position = 0; position < selections.size(); position++) {
            int index = selections.get(position).workspaceIndex();
            var declaration = memoryPlan.workspaces().get(index);
            var requirement = requirements.get(position);
            if (!workspaceIndices.add(index) || declaration.size().fixedByteSize().isEmpty()
                    || declaration.size().fixedByteSize().getAsLong() != requirement.byteSize()
                    || declaration.byteAlignment() != requirement.byteAlignment()) {
                throw new IllegalArgumentException("oneDNN workspace declaration disagrees with route");
            }
        }
        requiredBytes = new long[unitBuffers.size()];
        direct = new boolean[unitBuffers.size()];
        dataTypes = new DataType[unitBuffers.size()];
        java.util.Arrays.fill(direct, true);
        var projection = route.projection();
        for (int input = 0; input < projection.spec().inputs().size(); input++) {
            int boundary = projection.inputBoundaryPositions().get(input);
            dataTypes[boundary] = projection.spec().inputs().get(input).dataType();
            requiredBytes[boundary] = Math.max(requiredBytes[boundary], Math.addExact(
                    projection.byteOffsets().get(input), projection.spec().inputs().get(input).byteSize()));
        }
        int output = projection.outputBoundaryPosition();
        dataTypes[output] = projection.spec().output().dataType();
        requiredBytes[output] = Math.addExact(projection.byteOffsets().getLast(),
                projection.spec().output().byteSize());
        for (var copy : route.inputCopies()) {
            for (int input : copy.inputPositions()) {
                direct[projection.inputBoundaryPositions().get(input)] = false;
            }
        }
        // A repeated boundary with any direct native role still requires native storage.
        for (int input = 0; input < projection.spec().inputs().size(); input++) {
            int position = input;
            if (route.inputCopies().stream().noneMatch(copy -> copy.inputPositions().contains(position))) {
                direct[projection.inputBoundaryPositions().get(input)] = true;
            }
        }
        for (int boundary = 0; boundary < unitBuffers.size(); boundary++) {
            var fixed = memoryPlan.buffers().get(unitBuffers.get(boundary).bufferIndex())
                    .size().fixedByteSize();
            if (fixed.isEmpty() || fixed.getAsLong() < requiredBytes[boundary]) {
                throw new IllegalArgumentException("oneDNN boundary requires a sufficient fixed declaration");
            }
        }
        invocationCreator = (byteSize, byteAlignment) -> {
            if (byteSize != route.scratchpadBytes() || byteAlignment != 64) {
                throw new IllegalArgumentException("oneDNN invocation workspace request disagrees with route");
            }
            return kernel.createWorkspace();
        };
    }

    private static List<BufferAccess> accesses(List<BufferSelection> buffers,
            CpuOneDnnRoutePlan.Unit route) {
        Objects.requireNonNull(route, "route");
        var projection = route.projection();
        var accesses = new ArrayList<BufferAccess>(buffers.size());
        for (int boundary = 0; boundary < buffers.size(); boundary++) {
            if (boundary == projection.outputBoundaryPosition()) accesses.add(BufferAccess.WRITE_ONLY);
            else if (projection.inputBoundaryPositions().contains(boundary)) accesses.add(BufferAccess.READ_ONLY);
            else throw new IllegalArgumentException("oneDNN projection does not cover common boundary");
        }
        if (projection.outputBoundaryPosition() >= buffers.size()
                || projection.inputBoundaryPositions().stream().anyMatch(position -> position >= buffers.size())
                || !alignedOrigins(projection)) {
            throw new IllegalArgumentException("oneDNN projection boundary or origin disagrees");
        }
        return accesses;
    }

    private static boolean alignedOrigins(CpuOneDnnProjection projection) {
        for (int input = 0; input < projection.spec().inputs().size(); input++) {
            int width = projection.spec().inputs().get(input).dataType() == DataType.BOOL ? 1 : Float.BYTES;
            if (projection.byteOffsets().get(input) % width != 0) return false;
        }
        int width = projection.spec().output().dataType() == DataType.BOOL ? 1 : Float.BYTES;
        return projection.byteOffsets().getLast() % width == 0;
    }

    /**
     * Supplies native control-object creation only for this child's invocation workspace.
     * Shared raw ingress slots keep the ordinary CPU contiguous-workspace factory.
     *
     * @param workspaceIndex zero-based complete-plan workspace index, not child-local position
     * @return non-null optional retained factory; present only for the exact invocation slot,
     *     creating fresh run-owned resources and retaining the kernel/library lease on invocation
     */
    @Override public Optional<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreator(int workspaceIndex) {
        return workspaceIndex == selections.getLast().workspaceIndex()
                ? Optional.of(invocationCreator) : Optional.empty();
    }

    /**
     * Authenticates one fixed typed F32 or canonical one-byte BOOL boundary without changing handles.
     *
     * @param index zero-based child-local boundary selection
     * @param representation candidate borrowed representation; null or incompatible values fail closed
     * @return true only for a live sufficient dtype-aligned span with native storage when direct and
     *     write access when required; successful admission transfers no ownership
     * @throws IndexOutOfBoundsException if index is outside the retained boundary selections
     */
    @Override protected boolean acceptsBufferRepresentation(int index, BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != dataTypes[index] || cpu.byteSize() < requiredBytes[index]) return false;
        try {
            var argument = cpu.argument(); // Preserve observable-carrier authentication even for raw copies.
            MemorySegment segment = cpu.segment();
            int width = dataTypes[index] == DataType.BOOL ? 1 : Float.BYTES;
            return argument.byteOffset() % width == 0
                    && (!direct[index] || segment.isNative())
                    && segment.address() % width == 0
                    && (bufferAccess(index) != BufferAccess.WRITE_ONLY || !argument.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    /**
     * Also authenticates fixed boundary geometry within a run's complete memory plan.
     *
     * @param state non-null run state borrowing this executable's immutable plan
     * @param index zero-based child-local boundary selection
     * @param representation borrowed candidate, with no ownership transfer
     * @return true only when both intrinsic representation and applicable fixed geometry match
     * @throws NullPointerException if state is null after intrinsic admission
     * @throws IndexOutOfBoundsException if index is outside the retained boundary selections
     */
    @Override protected boolean acceptsBufferRepresentation(RunState state, int index,
            BufferRepresentation representation) {
        if (!acceptsBufferRepresentation(index, representation)) return false;
        if (!state.memoryPlan().requiresBatchSize()) return true;
        int buffer = bufferSelection(index).bufferIndex();
        long bytes = memoryPlan().buffers().get(buffer).size().fixedByteSize().orElseThrow();
        return ((CpuBufferRepresentation) representation).acceptsFixedGeometry(state, memoryPlan(), buffer, bytes);
    }

    /**
     * Distinguishes shared ingress storage from this exact kernel's private invocation state.
     *
     * @param index zero-based child-local workspace selection
     * @param representation borrowed candidate, null/incompatible values fail closed
     * @return true only for declared writable aligned native storage and the exact kernel identity
     *     when this is the invocation slot; no handles or leases are acquired
     * @throws IndexOutOfBoundsException if index is outside the retained workspace selections
     */
    @Override protected boolean acceptsWorkspaceRepresentation(int index, WorkspaceRepresentation representation) {
        try {
            long bytes = route.workspaceRequirements().get(index).byteSize();
            MemorySegment segment;
            if (index == selections.size() - 1) {
                if (!(representation instanceof CpuOneDnnWorkspace nativeWorkspace)
                        || nativeWorkspace.kernel() != kernel || !nativeWorkspace.isAccessible()
                        || nativeWorkspace.byteSize() != bytes) return false;
                segment = nativeWorkspace.writableSegment();
            } else {
                if (!(representation instanceof CpuContiguousWorkspace copy) || !copy.isAccessible()
                        || copy.byteSize() != bytes || copy.byteAlignment() < 64) return false;
                segment = copy.writableSegment();
            }
            return segment.isNative() && !segment.isReadOnly() && segment.address() % 64 == 0;
        } catch (IllegalArgumentException | IllegalStateException incompatible) { return false; }
    }

    /**
     * Requires workspace storage and numeric geometry from the same RunState memory plan.
     *
     * @param state non-null owning run state
     * @param index zero-based child-local workspace selection
     * @param representation borrowed candidate with no ownership transfer
     * @return true only for the exact plan association and declared run byte size
     * @throws NullPointerException if state is null after intrinsic admission
     * @throws IndexOutOfBoundsException if index is outside the retained workspace selections
     */
    @Override protected boolean acceptsWorkspaceRepresentation(RunState state, int index,
            WorkspaceRepresentation representation) {
        if (!acceptsWorkspaceRepresentation(index, representation)) return false;
        return state.memoryPlan() == memoryPlan() && state.geometry().memoryPlan() == memoryPlan()
                && state.geometry().workspaceByteSize(selections.get(index).workspaceIndex())
                    == route.workspaceRequirements().get(index).byteSize();
    }

    /**
     * Cold-binds precreated native handles and captures exact per-run copy/execute work.
     * Original data origins are sliced once; copied inputs bind at origin zero. All boundaries
     * and workspaces are checked for output/scratch aliasing before native handle updates.
     *
     * @param state non-null owning run state, borrowed by the bound invocation
     * @param buffers non-null already-admitted child-local boundaries, borrowed until run close
     * @param workspaces non-null admitted shared ingress slots followed by the native invocation
     *     workspace, all owned by state rather than this executable
     * @return non-null run-bound invocation refreshing only first-consumer ingress spans and
     *     synchronously executing under the shared budget with demand one
     * @throws NullPointerException if a required array, state or representation is null
     * @throws IllegalArgumentException if output/input/scratch alias or span geometry is invalid
     * @throws IllegalStateException if a representation is inaccessible or native binding fails
     */
    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        var sources = new MemorySegment[buffers.length];
        for (int boundary = 0; boundary < buffers.length; boundary++) {
            sources[boundary] = ((CpuBufferRepresentation) buffers[boundary]).segment();
        }
        int outputBoundary = route.projection().outputBoundaryPosition();
        for (int boundary = 0; boundary < sources.length; boundary++) {
            if (boundary != outputBoundary && sources[outputBoundary].asOverlappingSlice(sources[boundary]).isPresent()) {
                throw new IllegalArgumentException("oneDNN output overlaps an input");
            }
        }
        var targets = new MemorySegment[workspaces.length];
        for (int position = 0; position < workspaces.length; position++) {
            targets[position] = position == workspaces.length - 1
                    ? ((CpuOneDnnWorkspace) workspaces[position]).writableSegment()
                    : ((CpuContiguousWorkspace) workspaces[position]).writableSegment();
            for (MemorySegment source : sources) {
                if (targets[position].asOverlappingSlice(source).isPresent()) {
                    throw new IllegalArgumentException("oneDNN workspace overlaps a boundary");
                }
            }
            for (int earlier = 0; earlier < position; earlier++) {
                if (workspaces[earlier] == workspaces[position]
                        || targets[position].asOverlappingSlice(targets[earlier]).isPresent()) {
                    throw new IllegalArgumentException("oneDNN workspaces must be distinct and disjoint");
                }
            }
        }
        var projection = route.projection();
        var nativeInputs = new MemorySegment[projection.spec().inputs().size()];
        for (int input = 0; input < nativeInputs.length; input++) {
            nativeInputs[input] = sources[projection.inputBoundaryPositions().get(input)]
                    .asSlice(projection.byteOffsets().get(input), projection.spec().inputs().get(input).byteSize());
        }
        int refreshCount = (int) route.inputCopies().stream().filter(
                CpuOneDnnRoutePlan.InputCopy::refresh).count();
        var copySources = new MemorySegment[refreshCount];
        var copyTargets = new MemorySegment[refreshCount];
        int refresh = 0;
        for (int copy = 0; copy < route.inputCopies().size(); copy++) {
            var selectedCopy = route.inputCopies().get(copy);
            if (selectedCopy.refresh()) {
                copySources[refresh] = nativeInputs[selectedCopy.inputPositions().getFirst()];
                copyTargets[refresh++] = targets[copy];
            }
            for (int input : selectedCopy.inputPositions()) nativeInputs[input] = targets[copy];
        }
        MemorySegment output = sources[outputBoundary].asSlice(projection.byteOffsets().getLast(),
                projection.spec().output().byteSize());
        var invocationWorkspace = (CpuOneDnnWorkspace) workspaces[workspaces.length - 1];
        invocationWorkspace.bind(nativeInputs, output);
        Runnable executeNative = () -> {
            for (int copy = 0; copy < copySources.length; copy++) {
                MemorySegment.copy(copySources[copy], 0, copyTargets[copy], 0, copyTargets[copy].byteSize());
            }
            invocationWorkspace.execute();
        };
        return new BoundInvocation(state) {
            @Override protected void executeBound() { budget.execute(1, executeNative); }
        };
    }
}
