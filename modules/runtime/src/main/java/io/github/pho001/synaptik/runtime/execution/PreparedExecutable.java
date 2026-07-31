package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;
import java.util.Objects;

/**
 * Defines one immutable reusable backend-owned recipe for a prepared computation region.
 *
 * <p>The recipe retains one exact {@link PreparedMemoryPlan} and immutable private snapshots of
 * ordered dense resource selections. Buffer and workspace indices address positions in the
 * prepared plan's lists, not numeric slot components. A buffer representation index addresses
 * the selected run state's ordered bindings for that buffer position. Empty and repeated
 * selections are valid; repetition can represent repeated operand roles without duplicating
 * ownership in the run state.
 *
 * <p>{@link #bind(RunState)} is the final cold boundary. Runtime validates plan identity and
 * selection ranges, and a concrete backend performs explicit checked representation
 * compatibility before constructing a typed {@link BoundInvocation} with direct references.
 * The executable and its subclass state must be immutable and thread-safe so one recipe can bind
 * concurrently to distinct run states. The recipe owns no selected representation or bound
 * invocation and provides no allocation, transfer, residency, schedule, publication, or cleanup
 * lifecycle.
 */
public abstract class PreparedExecutable {
    private final PreparedMemoryPlan memoryPlan;
    private final BufferSelection[] bufferSelections;
    private final WorkspaceSelection[] workspaceSelections;

    /**
     * Creates an immutable prepared recipe from ordered dense resource selections.
     *
     * <p>The three top-level references are validated in parameter order. Buffer selections are
     * then validated in supplied order and copied to a private array before workspace selections
     * are validated and copied. The exact immutable selection objects and plan reference are
     * retained; caller list containers are not retained. Construction allocates only ordinary JVM
     * arrays and performs no physical resource or project-identifier operation.
     *
     * @param memoryPlan the immutable prepared memory plan to retain exactly; must be non-null
     * @param bufferSelections ordered dense buffer and representation positions to snapshot; must
     *     be non-null, contain no null entry, and use only prepared buffer positions
     * @param workspaceSelections ordered dense workspace positions to snapshot; must be non-null,
     *     contain no null entry, and use only prepared workspace positions
     * @throws NullPointerException if a top-level argument or selection entry is {@code null};
     *     indexed failures identify the first invalid supplied position
     * @throws IllegalArgumentException if a selection refers to a buffer or workspace position
     *     outside the prepared plan
     */
    protected PreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<PreparedExecutable.BufferSelection> bufferSelections,
            List<PreparedExecutable.WorkspaceSelection> workspaceSelections) {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(bufferSelections, "bufferSelections");
        Objects.requireNonNull(workspaceSelections, "workspaceSelections");

        int bufferCount = memoryPlan.buffers().size();
        for (int index = 0; index < bufferSelections.size(); index++) {
            BufferSelection selection =
                    Objects.requireNonNull(
                            bufferSelections.get(index), "bufferSelections[" + index + "]");
            if (selection.bufferIndex() >= bufferCount) {
                throw new IllegalArgumentException(
                        "bufferSelections["
                                + index
                                + "].bufferIndex out of prepared-plan range: "
                                + selection.bufferIndex());
            }
        }
        BufferSelection[] copiedBufferSelections =
                bufferSelections.toArray(BufferSelection[]::new);

        int workspaceCount = memoryPlan.workspaces().size();
        for (int index = 0; index < workspaceSelections.size(); index++) {
            WorkspaceSelection selection =
                    Objects.requireNonNull(
                            workspaceSelections.get(index),
                            "workspaceSelections[" + index + "]");
            if (selection.workspaceIndex() >= workspaceCount) {
                throw new IllegalArgumentException(
                        "workspaceSelections["
                                + index
                                + "].workspaceIndex out of prepared-plan range: "
                                + selection.workspaceIndex());
            }
        }
        WorkspaceSelection[] copiedWorkspaceSelections =
                workspaceSelections.toArray(WorkspaceSelection[]::new);

        this.memoryPlan = memoryPlan;
        this.bufferSelections = copiedBufferSelections;
        this.workspaceSelections = copiedWorkspaceSelections;
    }

    /**
     * Returns the exact prepared-memory-plan reference associated with this recipe.
     *
     * @return the retained non-null immutable plan reference; never a copy or structural
     *     replacement
     */
    public final PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Cold-binds this reusable recipe to the exact resources of one open matching run state.
     *
     * <p>Selections are resolved in their original supplied order into fresh nominal arrays.
     * Buffer representations are checked first, then workspaces. Each compatibility hook is
     * called exactly once for its selected representation. Only after every check succeeds is
     * {@link #bindCompatible(RunState, BufferRepresentation[], WorkspaceRepresentation[])} called
     * once. The returned invocation must retain the exact supplied state.
     *
     * <p>Binding may allocate ordinary JVM arrays and the bound invocation. It acquires no
     * independently closeable or native auxiliary binding resource, changes no ownership, and
     * performs no cleanup on failure.
     *
     * @param runState the open per-run state whose exact plan reference and selected
     *     representations are used; must be non-null
     * @return the non-null backend-owned invocation associated with exactly {@code runState}
     * @throws NullPointerException if {@code runState} or the backend-created invocation is
     *     {@code null}
     * @throws IllegalStateException if {@code runState} is closed
     * @throws IllegalArgumentException if the run uses another plan object, a representation
     *     position is absent, a selected representation is incompatible, or the returned
     *     invocation belongs to another run state
     */
    public final BoundInvocation bind(RunState runState) {
        Objects.requireNonNull(runState, "runState");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        if (runState.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "run state memory plan does not match prepared executable memory plan");
        }

        var bufferRepresentations = new BufferRepresentation[bufferSelections.length];
        for (int index = 0; index < bufferSelections.length; index++) {
            BufferSelection selection = bufferSelections[index];
            int representationIndex = selection.representationIndex();
            if (representationIndex
                    >= runState.bufferRepresentationCount(selection.bufferIndex())) {
                throw new IllegalArgumentException(
                        "bufferSelections["
                                + index
                                + "].representationIndex out of run-state range: "
                                + representationIndex);
            }
            BufferRepresentation representation =
                    runState
                            .bufferRepresentation(
                                    selection.bufferIndex(), representationIndex)
                            .representation();
            bufferRepresentations[index] = representation;
            if (!acceptsBufferRepresentation(index, representation)) {
                throw new IllegalArgumentException(
                        "bufferSelections["
                                + index
                                + "] is incompatible with prepared executable");
            }
        }

        var workspaceRepresentations = new WorkspaceRepresentation[workspaceSelections.length];
        for (int index = 0; index < workspaceSelections.length; index++) {
            WorkspaceRepresentation representation =
                    runState.workspaceRepresentation(workspaceSelections[index].workspaceIndex());
            workspaceRepresentations[index] = representation;
            if (!acceptsWorkspaceRepresentation(index, representation)) {
                throw new IllegalArgumentException(
                        "workspaceSelections["
                                + index
                                + "] is incompatible with prepared executable");
            }
        }

        BoundInvocation invocation =
                Objects.requireNonNull(
                        bindCompatible(
                                runState, bufferRepresentations, workspaceRepresentations),
                        "boundInvocation");
        if (invocation.runState() != runState) {
            throw new IllegalArgumentException(
                    "bound invocation does not belong to supplied run state");
        }
        return invocation;
    }

    /**
     * Reports whether one resolved buffer selection is compatible with this backend recipe.
     *
     * <p>The concrete backend must use an explicit checked type test such as {@code instanceof}.
     * Returning {@code false} lets the final Runtime binding method issue the standardized
     * indexed failure. The hook is called once per resolved selection in selection order during
     * cold binding and never during bound execution.
     *
     * @param selectionIndex the zero-based position in the executable's ordered buffer selections
     * @param representation the exact non-null nominal run-state representation selected at that
     *     position
     * @return {@code true} if the representation has the concrete backend type and compatibility
     *     required by this selection; otherwise {@code false}
     */
    protected abstract boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation);

    /**
     * Reports whether one resolved workspace selection is compatible with this backend recipe.
     *
     * <p>The concrete backend must use an explicit checked type test such as {@code instanceof}.
     * Returning {@code false} lets the final Runtime binding method issue the standardized
     * indexed failure. The hook is called once per resolved selection in selection order during
     * cold binding and never during bound execution.
     *
     * @param selectionIndex the zero-based position in the executable's ordered workspace
     *     selections
     * @param representation the exact non-null nominal run-state representation selected at that
     *     position
     * @return {@code true} if the representation has the concrete backend type and compatibility
     *     required by this selection; otherwise {@code false}
     */
    protected abstract boolean acceptsWorkspaceRepresentation(
            int selectionIndex, WorkspaceRepresentation representation);

    /**
     * Constructs one backend-owned typed invocation after every selection is compatible.
     *
     * <p>The arrays are fresh, ordered like the executable selections, and contain the exact
     * nominal representation references resolved from {@code runState}. The implementation may
     * use ordinary checked casts justified by the completed compatibility pass, but the returned
     * invocation must retain direct concrete typed fields rather than either nominal array as its
     * hot-path access mechanism. It must retain exactly {@code runState} through its superclass.
     *
     * @param runState the exact non-null open state supplied to {@link #bind(RunState)}
     * @param bufferRepresentations fresh non-null array of exact selected buffer representations
     *     in original selection order; never retained as the invocation's hot-path access
     *     mechanism
     * @param workspaceRepresentations fresh non-null array of exact selected workspace
     *     representations in original selection order; never retained as the invocation's
     *     hot-path access mechanism
     * @return a non-null backend-owned invocation retaining exactly {@code runState} and direct
     *     concrete representation references
     * @throws RuntimeException if backend-specific invocation construction fails without
     *     acquiring an auxiliary closeable resource
     * @throws Error if backend-specific invocation construction reports an error
     */
    protected abstract BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations);

    /**
     * Selects one ordered buffer representation through dense prepared-plan and run-state
     * positions.
     *
     * <p>The value is deeply immutable. Indices are positions, not slot numeric components, and
     * repeated equal selections are valid. Ordinary record equality and hashing compare both
     * components; record text is diagnostic only.
     *
     * @param bufferIndex the non-negative dense position in {@code memoryPlan().buffers()}
     * @param representationIndex the non-negative dense position in the selected run-state
     *     buffer's ordered representation bindings
     */
    public record BufferSelection(int bufferIndex, int representationIndex) {
        /**
         * Creates one dense buffer-representation selection.
         *
         * @param bufferIndex the prepared buffer position; must be non-negative
         * @param representationIndex the per-run representation position; must be non-negative
         * @throws IllegalArgumentException if either component is negative; validation follows
         *     declaration order
         */
        public BufferSelection {
            if (bufferIndex < 0) {
                throw new IllegalArgumentException("bufferIndex must be non-negative");
            }
            if (representationIndex < 0) {
                throw new IllegalArgumentException(
                        "representationIndex must be non-negative");
            }
        }

        /**
         * Returns the dense prepared buffer position.
         *
         * @return the non-negative zero-based position in the prepared plan's ordered buffers
         */
        @Override
        public int bufferIndex() {
            return bufferIndex;
        }

        /**
         * Returns the dense representation position within the selected run-state buffer.
         *
         * @return the non-negative zero-based representation position
         */
        @Override
        public int representationIndex() {
            return representationIndex;
        }
    }

    /**
     * Selects one workspace representation through its dense prepared-plan position.
     *
     * <p>The value is deeply immutable. Its index is a position, not a workspace-slot numeric
     * component, and repeated equal selections are valid. Ordinary record equality and hashing
     * use the component; record text is diagnostic only.
     *
     * @param workspaceIndex the non-negative dense position in
     *     {@code memoryPlan().workspaces()}
     */
    public record WorkspaceSelection(int workspaceIndex) {
        /**
         * Creates one dense workspace selection.
         *
         * @param workspaceIndex the prepared workspace position; must be non-negative
         * @throws IllegalArgumentException if {@code workspaceIndex} is negative
         */
        public WorkspaceSelection {
            if (workspaceIndex < 0) {
                throw new IllegalArgumentException("workspaceIndex must be non-negative");
            }
        }

        /**
         * Returns the dense prepared workspace position.
         *
         * @return the non-negative zero-based position in the prepared plan's ordered workspaces
         */
        @Override
        public int workspaceIndex() {
            return workspaceIndex;
        }
    }
}
