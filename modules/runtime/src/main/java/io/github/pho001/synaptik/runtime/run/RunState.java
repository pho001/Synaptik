package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.util.List;
import java.util.Objects;

/**
 * Owns the representation bindings and cleanup lifecycle for one complete logical run.
 *
 * <p>One instance covers all backend partitions in that run. It retains the exact immutable
 * {@link PreparedMemoryPlan} reference and snapshots the supplied list structure into private
 * arrays in the plan's buffer and workspace encounter order. Buffer positions contain one or
 * more ordered bindings; workspace positions contain exactly one run-owned representation.
 * Indexed representation access uses those arrays directly. Position indices are dense
 * zero-based list positions, not numeric {@code BufferSlot} or {@code WorkspaceSlot} components.
 *
 * <p>Every bound representation is structurally resident until closure: its exact physical
 * object exists and remains bound to this run. Each buffer representation has one independent
 * validity bit stating whether that resident copy currently contains the logical slot value.
 * Borrowed buffers start valid and run-owned buffers start invalid. Any subset, including zero or
 * multiple copies, may be valid. Workspaces are resident backend-local scratch and have no
 * logical validity. Validity mutation records a fact only; it performs no storage access, copy,
 * transfer, kernel call, ownership change, or implicit change to another bit.
 *
 * <p>Every exact representation object may occur only once across both domains in one state,
 * preventing ambiguous cleanup ownership. Successful construction transfers responsibility for
 * {@link RunResourceOwnership#RUN_OWNED} buffers and every workspace; any constructor failure
 * transfers nothing and closes nothing.
 *
 * <p>{@link #close()} marks the logical state closed before physical cleanup, skips borrowed
 * buffers, and attempts every owned representation once in deterministic reverse order. Closing
 * is idempotent. Representation access is unavailable after closure, while immutable plan and
 * count inspection remains available. A prepared executable may cold-bind direct typed
 * references from this state while it is open. A resulting bound invocation retains this exact
 * state and rejects execution after closure; neither object acquires ownership of the state or
 * its representations.
 *
 * <p>This class is not thread-safe. A caller must not race access, validity mutation, or closure on
 * one instance. Separate concurrent states may share the immutable plan, but their run-owned
 * representation objects must be distinct. A borrowed representation may be shared only when its
 * caller guarantees lifetime, safe access, and external synchronization for every run using it.
 */
public final class RunState implements AutoCloseable {
    private final PreparedMemoryPlan memoryPlan;
    private final BufferRepresentationBinding[][] bufferBindings;
    private final boolean[][] bufferValidity;
    private final WorkspaceRepresentation[] workspaceRepresentations;
    private boolean closed;

    /**
     * Creates the complete representation state for one run of a prepared memory plan.
     *
     * <p>Top-level inputs and plan-sized counts are validated first. Buffer positions are then
     * scanned in increasing order, followed by workspace positions. All list structure is copied
     * into private arrays while exact binding and representation objects are retained. One
     * parallel validity array is initialized from ownership: borrowed buffers are valid and
     * run-owned buffers are invalid. Ownership transfers only after the complete input has passed
     * null, count, non-empty, and duplicate-identity validation.
     *
     * @param memoryPlan the immutable prepared plan to retain exactly; must be non-null
     * @param bufferBindings the outer plan-ordered buffer positions to snapshot; must be non-null,
     *     have one entry per prepared buffer, and contain a non-null, non-empty list of non-null
     *     bindings at each position
     * @param workspaceRepresentations the plan-ordered workspace representations to snapshot;
     *     must be non-null, have one non-null entry per prepared workspace, and each entry becomes
     *     run-owned only after successful construction
     * @throws NullPointerException if a top-level argument, inner buffer list, buffer binding, or
     *     workspace representation is {@code null}; the exception message identifies the first
     *     invalid argument or indexed position
     * @throws IllegalArgumentException if a top-level count differs from the prepared plan, an
     *     inner buffer list is empty, or an exact representation object occurs more than once;
     *     count and position failures identify the rejected input, while every repeated identity
     *     reports {@code representation is already bound to this run}
     */
    public RunState(
            PreparedMemoryPlan memoryPlan,
            List<List<BufferRepresentationBinding>> bufferBindings,
            List<WorkspaceRepresentation> workspaceRepresentations) {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(bufferBindings, "bufferBindings");
        Objects.requireNonNull(workspaceRepresentations, "workspaceRepresentations");

        int bufferCount = memoryPlan.buffers().size();
        if (bufferBindings.size() != bufferCount) {
            throw new IllegalArgumentException(
                    "bufferBindings size must equal prepared buffer count " + bufferCount);
        }

        int workspaceCount = memoryPlan.workspaces().size();
        if (workspaceRepresentations.size() != workspaceCount) {
            throw new IllegalArgumentException(
                    "workspaceRepresentations size must equal prepared workspace count "
                            + workspaceCount);
        }

        var copiedBufferBindings = new BufferRepresentationBinding[bufferCount][];
        for (int bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
            List<BufferRepresentationBinding> suppliedBindings =
                    Objects.requireNonNull(
                            bufferBindings.get(bufferIndex),
                            "bufferBindings[" + bufferIndex + "]");
            if (suppliedBindings.isEmpty()) {
                throw new IllegalArgumentException(
                        "bufferBindings[" + bufferIndex + "] must not be empty");
            }

            var copiedBindings =
                    new BufferRepresentationBinding[suppliedBindings.size()];
            copiedBufferBindings[bufferIndex] = copiedBindings;
            for (int representationIndex = 0;
                    representationIndex < copiedBindings.length;
                    representationIndex++) {
                BufferRepresentationBinding binding =
                        Objects.requireNonNull(
                                suppliedBindings.get(representationIndex),
                                "bufferBindings["
                                        + bufferIndex
                                        + "]["
                                        + representationIndex
                                        + "]");
                rejectRepeatedBufferIdentity(
                        copiedBufferBindings,
                        bufferIndex,
                        representationIndex,
                        binding.representation());
                copiedBindings[representationIndex] = binding;
            }
        }

        var copiedWorkspaceRepresentations = new WorkspaceRepresentation[workspaceCount];
        for (int workspaceIndex = 0; workspaceIndex < workspaceCount; workspaceIndex++) {
            WorkspaceRepresentation representation =
                    Objects.requireNonNull(
                            workspaceRepresentations.get(workspaceIndex),
                            "workspaceRepresentations[" + workspaceIndex + "]");
            rejectRepeatedWorkspaceIdentity(
                    copiedBufferBindings,
                    copiedWorkspaceRepresentations,
                    workspaceIndex,
                    representation);
            copiedWorkspaceRepresentations[workspaceIndex] = representation;
        }

        this.memoryPlan = memoryPlan;
        this.bufferBindings = copiedBufferBindings;
        this.bufferValidity = new boolean[bufferCount][];
        for (int bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
            BufferRepresentationBinding[] bindings = copiedBufferBindings[bufferIndex];
            boolean[] validity = new boolean[bindings.length];
            for (int representationIndex = 0;
                    representationIndex < bindings.length;
                    representationIndex++) {
                validity[representationIndex] =
                        bindings[representationIndex].ownership()
                                == RunResourceOwnership.BORROWED;
            }
            bufferValidity[bufferIndex] = validity;
        }
        this.workspaceRepresentations = copiedWorkspaceRepresentations;
    }

    /**
     * Returns the exact prepared-memory-plan reference supplied at construction.
     *
     * <p>This immutable inspection remains available after closure.
     *
     * @return the retained non-null plan reference; never a copy or replacement
     */
    public PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the number of dense buffer positions in prepared-plan encounter order.
     *
     * <p>This immutable count remains available after closure.
     *
     * @return the exact number of entries in {@code memoryPlan().buffers()}; never negative
     */
    public int bufferSlotCount() {
        return bufferBindings.length;
    }

    /**
     * Returns the number of representations bound at one dense buffer position.
     *
     * <p>The position is the zero-based index into {@code memoryPlan().buffers()}, not the
     * numeric component of its {@code BufferSlot}. This immutable count remains available after
     * closure.
     *
     * @param bufferIndex the dense zero-based prepared buffer position
     * @return the positive number of ordered representations at that position
     * @throws IndexOutOfBoundsException if {@code bufferIndex} is outside the prepared buffer
     *     positions
     */
    public int bufferRepresentationCount(int bufferIndex) {
        checkBufferIndex(bufferIndex);
        return bufferBindings[bufferIndex].length;
    }

    /**
     * Returns one exact buffer binding from the private array snapshot.
     *
     * <p>The buffer position follows prepared-plan encounter order, and the representation
     * position follows the supplied inner-list order. The returned binding and representation
     * are the exact objects supplied at construction. The binding's presence establishes
     * structural residency only; query {@link #isBufferRepresentationValid(int, int)} separately
     * for logical validity. Access performs no coherence, transfer, or backend compatibility
     * action.
     *
     * @param bufferIndex the dense zero-based prepared buffer position
     * @param representationIndex the zero-based representation position within that buffer
     * @return the retained non-null binding reference; never a copy or replacement
     * @throws IllegalStateException if this run state is closed; this check occurs before index
     *     validation
     * @throws IndexOutOfBoundsException if either index is outside its corresponding position
     *     range
     */
    public BufferRepresentationBinding bufferRepresentation(
            int bufferIndex, int representationIndex) {
        requireOpen();
        checkBufferIndex(bufferIndex);
        BufferRepresentationBinding[] representations = bufferBindings[bufferIndex];
        if (representationIndex < 0 || representationIndex >= representations.length) {
            throw new IndexOutOfBoundsException(
                    "representationIndex out of range: " + representationIndex);
        }
        return representations[representationIndex];
    }

    /**
     * Reports whether one resident buffer representation contains the logical slot value.
     *
     * <p>The query is a constant-time private-array read. It performs no storage inspection,
     * transfer, backend call, ownership transition, or validity inference.
     *
     * @param bufferIndex the dense zero-based prepared buffer position
     * @param representationIndex the dense zero-based representation position
     * @return the current validity bit for exactly the selected resident representation
     * @throws IllegalStateException if this run state is closed; this check occurs before index
     *     validation
     * @throws IndexOutOfBoundsException if either index is outside its corresponding range; the
     *     first rejected index is identified by the existing dense-index diagnostic
     */
    public boolean isBufferRepresentationValid(
            int bufferIndex, int representationIndex) {
        requireOpen();
        checkBufferIndex(bufferIndex);
        checkRepresentationIndex(bufferIndex, representationIndex);
        return bufferValidity[bufferIndex][representationIndex];
    }

    /**
     * Stores one explicit logical-validity fact without performing physical work.
     *
     * <p>The update is a constant-time write of exactly {@code valid}. It neither validates bytes
     * nor copies data, calls a backend or kernel, changes ownership, invalidates another copy, or
     * establishes a transition policy. Repeating the current value is permitted.
     *
     * @param bufferIndex the dense zero-based prepared buffer position
     * @param representationIndex the dense zero-based representation position
     * @param valid the exact validity value to store
     * @throws IllegalStateException if this run state is closed; this check occurs before index
     *     validation
     * @throws IndexOutOfBoundsException if either index is outside its corresponding range; the
     *     first rejected index is identified by the existing dense-index diagnostic
     */
    public void setBufferRepresentationValid(
            int bufferIndex, int representationIndex, boolean valid) {
        requireOpen();
        checkBufferIndex(bufferIndex);
        checkRepresentationIndex(bufferIndex, representationIndex);
        bufferValidity[bufferIndex][representationIndex] = valid;
    }

    /**
     * Returns the number of dense workspace positions in prepared-plan encounter order.
     *
     * <p>This immutable count remains available after closure.
     *
     * @return the exact number of entries in {@code memoryPlan().workspaces()}; never negative
     */
    public int workspaceSlotCount() {
        return workspaceRepresentations.length;
    }

    /**
     * Returns the exact representation bound to one dense workspace position.
     *
     * <p>The position is the zero-based index into {@code memoryPlan().workspaces()}, not the
     * numeric component of its {@code WorkspaceSlot}. The returned object is the exact supplied
     * run-owned representation. Its binding establishes structural residency until closure, but
     * workspace scratch has no logical validity. Access performs no storage, transfer, or backend
     * compatibility action.
     *
     * @param workspaceIndex the dense zero-based prepared workspace position
     * @return the retained non-null workspace representation; never a copy or replacement
     * @throws IllegalStateException if this run state is closed; this check occurs before index
     *     validation
     * @throws IndexOutOfBoundsException if {@code workspaceIndex} is outside the prepared
     *     workspace positions
     */
    public WorkspaceRepresentation workspaceRepresentation(int workspaceIndex) {
        requireOpen();
        checkWorkspaceIndex(workspaceIndex);
        return workspaceRepresentations[workspaceIndex];
    }

    /**
     * Reports whether this state has begun or completed its one cleanup lifecycle.
     *
     * <p>The value changes to {@code true} before any physical representation cleanup begins and
     * remains true even when cleanup reports a failure. This method does not make the class safe
     * for racing access and closure.
     *
     * @return {@code true} after the first close operation has claimed cleanup responsibility;
     *     otherwise {@code false}
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Closes every representation still owned by this run in deterministic reverse order.
     *
     * <p>The state becomes closed before cleanup begins. Workspace positions are attempted from
     * last to first, followed by buffer positions from last to first and each position's
     * representations from last to first. Borrowed buffers are skipped. Every owned
     * representation is attempted once even if another cleanup fails. The first encountered
     * {@link RuntimeException} or {@link Error} is rethrown after all attempts, with later
     * failures suppressed in encounter order. A repeated call performs no cleanup and does not
     * rethrow a prior failure.
     *
     * <p>Callers must not race this method with access, another close, or future mutation.
     *
     * @throws RuntimeException if an owned representation's cleanup first reports an unchecked
     *     exception; later cleanup failures are attached to that same exception as suppressed
     *     failures
     * @throws Error if an owned representation's cleanup first reports an error; later cleanup
     *     failures are attached to that same error as suppressed failures
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        Throwable firstFailure = null;
        for (int workspaceIndex = workspaceRepresentations.length - 1;
                workspaceIndex >= 0;
                workspaceIndex--) {
            try {
                workspaceRepresentations[workspaceIndex].close();
            } catch (RuntimeException | Error failure) {
                if (firstFailure == null) {
                    firstFailure = failure;
                } else {
                    firstFailure.addSuppressed(failure);
                }
            }
        }

        for (int bufferIndex = bufferBindings.length - 1; bufferIndex >= 0; bufferIndex--) {
            BufferRepresentationBinding[] representations = bufferBindings[bufferIndex];
            for (int representationIndex = representations.length - 1;
                    representationIndex >= 0;
                    representationIndex--) {
                BufferRepresentationBinding binding = representations[representationIndex];
                if (binding.ownership() == RunResourceOwnership.BORROWED) {
                    continue;
                }
                try {
                    binding.representation().close();
                } catch (RuntimeException | Error failure) {
                    if (firstFailure == null) {
                        firstFailure = failure;
                    } else {
                        firstFailure.addSuppressed(failure);
                    }
                }
            }
        }

        if (firstFailure instanceof RuntimeException failure) {
            throw failure;
        }
        if (firstFailure instanceof Error failure) {
            throw failure;
        }
    }

    private static void rejectRepeatedBufferIdentity(
            BufferRepresentationBinding[][] copiedBufferBindings,
            int bufferIndex,
            int representationIndex,
            BufferRepresentation representation) {
        for (int earlierBufferIndex = 0;
                earlierBufferIndex <= bufferIndex;
                earlierBufferIndex++) {
            BufferRepresentationBinding[] earlierBindings =
                    copiedBufferBindings[earlierBufferIndex];
            int earlierCount =
                    earlierBufferIndex == bufferIndex
                            ? representationIndex
                            : earlierBindings.length;
            for (int earlierRepresentationIndex = 0;
                    earlierRepresentationIndex < earlierCount;
                    earlierRepresentationIndex++) {
                if (earlierBindings[earlierRepresentationIndex].representation()
                        == representation) {
                    throw new IllegalArgumentException(
                            "representation is already bound to this run");
                }
            }
        }
    }

    private static void rejectRepeatedWorkspaceIdentity(
            BufferRepresentationBinding[][] copiedBufferBindings,
            WorkspaceRepresentation[] copiedWorkspaceRepresentations,
            int workspaceIndex,
            WorkspaceRepresentation representation) {
        for (BufferRepresentationBinding[] bindings : copiedBufferBindings) {
            for (BufferRepresentationBinding binding : bindings) {
                if (binding.representation() == representation) {
                    throw new IllegalArgumentException(
                            "representation is already bound to this run");
                }
            }
        }
        for (int earlierWorkspaceIndex = 0;
                earlierWorkspaceIndex < workspaceIndex;
                earlierWorkspaceIndex++) {
            if (copiedWorkspaceRepresentations[earlierWorkspaceIndex] == representation) {
                throw new IllegalArgumentException(
                        "representation is already bound to this run");
            }
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("run state is closed");
        }
    }

    private void checkBufferIndex(int bufferIndex) {
        if (bufferIndex < 0 || bufferIndex >= bufferBindings.length) {
            throw new IndexOutOfBoundsException("bufferIndex out of range: " + bufferIndex);
        }
    }

    private void checkRepresentationIndex(int bufferIndex, int representationIndex) {
        if (representationIndex < 0
                || representationIndex >= bufferBindings[bufferIndex].length) {
            throw new IndexOutOfBoundsException(
                    "representationIndex out of range: " + representationIndex);
        }
    }

    private void checkWorkspaceIndex(int workspaceIndex) {
        if (workspaceIndex < 0 || workspaceIndex >= workspaceRepresentations.length) {
            throw new IndexOutOfBoundsException(
                    "workspaceIndex out of range: " + workspaceIndex);
        }
    }
}
