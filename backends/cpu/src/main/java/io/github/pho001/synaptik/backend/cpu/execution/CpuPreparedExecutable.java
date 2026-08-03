package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

/**
 * Performs the CPU-specific half of Runtime cold binding before a route builds direct fields.
 *
 * <p>The immutable recipe validates CPU representation type, lifetime, access, exact geometry,
 * data type, carrier, and native alignment. It performs no allocation, materialization, route
 * selection, or operation dispatch.</p>
 */
abstract class CpuPreparedExecutable extends PreparedExecutable {
    private final DataType[] bufferDataTypes;
    private final WorkspaceSelection[] workspaceSelections;

    /**
     * Creates an immutable reusable CPU recipe aligned with Runtime's ordered selections.
     *
     * @param memoryPlan exact non-null prepared plan retained by the Runtime base
     * @param bufferSelections non-null ordered selections; copied by the Runtime base
     * @param workspaceSelections non-null ordered selections; copied here and by the base
     * @param bufferAccesses non-null accesses aligned one-for-one with buffer selections
     * @param bufferDataTypes non-null logical data types aligned one-for-one with buffer selections
     * @throws NullPointerException if an aggregate or required entry is {@code null}
     * @throws IllegalArgumentException if list sizes, selections, accesses, or plan associations
     *         violate the Runtime contract
     */
    protected CpuPreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections,
            List<WorkspaceSelection> workspaceSelections,
            List<BufferAccess> bufferAccesses,
            List<DataType> bufferDataTypes) {
        super(memoryPlan, bufferSelections, workspaceSelections, bufferAccesses);
        Objects.requireNonNull(bufferDataTypes, "bufferDataTypes");
        if (bufferDataTypes.size() != bufferSelectionCount()) {
            throw new IllegalArgumentException(
                    "bufferDataTypes size must equal buffer selection count " + bufferSelectionCount());
        }
        this.bufferDataTypes = new DataType[bufferDataTypes.size()];
        for (int index = 0; index < this.bufferDataTypes.length; index++) {
            this.bufferDataTypes[index] = Objects.requireNonNull(
                    bufferDataTypes.get(index), "bufferDataTypes[" + index + "]");
        }
        this.workspaceSelections = workspaceSelections.toArray(WorkspaceSelection[]::new);
    }

    /**
     * Constructs one route-specific invocation from fresh cold-binding arrays.
     *
     * <p>The implementation must copy the needed argument and workspace references into direct
     * typed fields and must not retain either array. It may allocate the invocation and ordinary
     * fixed-size binding state, but owns and closes no selected representation.</p>
     *
     * @param runState exact non-null open state being bound; retained by the returned invocation
     * @param bufferArguments fresh non-null array of already validated direct arguments
     * @param workspaces fresh non-null array of already validated native workspaces
     * @return a non-null invocation associated with {@code runState}
     */
    protected abstract BoundInvocation bindCpu(
            RunState runState,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);

    /**
     * Checks one selected buffer at the CPU cold-binding boundary.
     *
     * @param selectionIndex zero-based buffer-selection position
     * @param representation non-null physical representation supplied by Runtime
     * @return whether type, lifetime, current-thread access, exact size, writability, logical data
     *         type, direct argument form, and any required exact-segment alignment are compatible
     */
    @Override
    protected final boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()) {
            return false;
        }
        PreparedMemoryPlan.BufferEntry entry = memoryPlan().buffers()
                .get(bufferSelection(selectionIndex).bufferIndex());
        if (cpu.byteSize() != entry.byteSize()) {
            return false;
        }
        MemorySegment segment = cpu.segment();
        if (bufferAccess(selectionIndex) != BufferAccess.READ_ONLY && segment.isReadOnly()) {
            return false;
        }
        if (cpu.dataType() != bufferDataTypes[selectionIndex]) return false;
        CpuBufferArgument argument;
        try {
            argument = cpu.argument();
        } catch (RuntimeException failure) {
            return false;
        }
        return segment.heapBase().isPresent() || segment.address() % entry.byteAlignment() == 0;
    }

    /**
     * Checks one selected workspace at the CPU cold-binding boundary.
     *
     * @param selectionIndex zero-based workspace-selection position
     * @param representation non-null physical representation supplied by Runtime
     * @return whether the workspace is an accessible open CPU workspace with exact size and
     *         compatible declared and actual alignment
     */
    @Override
    protected final boolean acceptsWorkspaceRepresentation(
            int selectionIndex, WorkspaceRepresentation representation) {
        if (!(representation instanceof CpuNativeWorkspace workspace) || !workspace.isAccessible()) {
            return false;
        }
        PreparedMemoryPlan.WorkspaceEntry entry = memoryPlan().workspaces()
                .get(workspaceSelections[selectionIndex].workspaceIndex());
        return workspace.byteSize() == entry.byteSize()
                && workspace.byteAlignment() >= entry.byteAlignment()
                && workspace.byteAlignment() % entry.byteAlignment() == 0
                && workspace.segment().address() % entry.byteAlignment() == 0;
    }

    /**
     * Converts the fully validated Runtime selections to fresh typed CPU arrays and delegates
     * exactly once to {@link #bindCpu(RunState, CpuBufferArgument[], CpuNativeWorkspace[])}.
     *
     * @param runState exact non-null open state already validated by Runtime
     * @param bufferRepresentations non-null compatible buffer selections in recipe order
     * @param workspaceRepresentations non-null compatible workspace selections in recipe order
     * @return the route-specific non-null invocation for {@code runState}
     */
    @Override
    protected final BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations) {
        var arguments = new CpuBufferArgument[bufferRepresentations.length];
        for (int index = 0; index < arguments.length; index++) {
            arguments[index] = ((CpuBufferRepresentation) bufferRepresentations[index]).argument();
        }
        var workspaces = new CpuNativeWorkspace[workspaceRepresentations.length];
        for (int index = 0; index < workspaces.length; index++) {
            workspaces[index] = (CpuNativeWorkspace) workspaceRepresentations[index];
        }
        return bindCpu(runState, arguments, workspaces);
    }
}
