package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Performs cold, all-or-cleaned creation of one complete run state.
 *
 * <p>This package-private operation is the setup seam for a later Runtime runner, not a public
 * lifecycle facade. It validates the complete caller-input list before invoking any callback,
 * creates buffer representations in dense buffer/representation order and workspaces afterward,
 * and transfers ownership only when {@link RunState} construction succeeds.
 */
final class RunStateCreation {
    private RunStateCreation() {}

    /**
     * Creates one run state from a reusable prepared plan and dense caller inputs.
     *
     * <p>Each caller input is retained exactly as borrowed and initially valid. Each creator must
     * return a fresh non-null identity; created buffers are run-owned and initially invalid, and
     * created workspaces are run-owned scratch without validity. If callback work, result
     * validation, or final state construction fails, all successfully created results are closed
     * once in reverse creation order. The original unchecked failure is rethrown unchanged and
     * cleanup failures are suppressed on it in cleanup encounter order. Borrowed inputs and a
     * duplicate callback result are never closed as newly owned resources.
     *
     * @param representationPlan the non-null immutable representation plan whose exact memory
     *     plan the returned state retains
     * @param callerInputs the non-null caller inputs in dense caller-preparation encounter order;
     *     elements must be non-null and identity-distinct and remain caller-owned
     * @return a new non-null open run state containing the exact borrowed inputs and fresh created
     *     results; cleanup ownership for created results has transferred to the state
     * @throws NullPointerException if a top-level input, caller-input element, or callback result
     *     is {@code null}; indexed failures identify the rejected dense position
     * @throws IllegalArgumentException if the caller count differs from the plan or an exact
     *     representation identity is repeated among callers and created results
     * @throws RuntimeException if a creator or final state construction reports an unchecked
     *     exception; successfully created results are rolled back and cleanup failures are
     *     suppressed on the original exception
     * @throws Error if a creator or final state construction reports an error; successfully
     *     created results are rolled back and cleanup failures are suppressed on the original
     *     error
     */
    static RunState create(
            PreparedRepresentationPlan representationPlan,
            List<BufferRepresentation> callerInputs) {
        Objects.requireNonNull(representationPlan, "representationPlan");
        Objects.requireNonNull(callerInputs, "callerInputs");

        int callerInputCount = 0;
        int createdBufferCount = 0;
        for (List<PreparedRepresentationPlan.BufferPreparation> preparations
                : representationPlan.bufferPreparations()) {
            for (PreparedRepresentationPlan.BufferPreparation preparation : preparations) {
                if (preparation instanceof CallerInput) {
                    callerInputCount++;
                } else {
                    createdBufferCount++;
                }
            }
        }
        if (callerInputs.size() != callerInputCount) {
            throw new IllegalArgumentException(
                    "callerInputs size must equal caller-input preparation count "
                            + callerInputCount);
        }
        for (int index = 0; index < callerInputCount; index++) {
            BufferRepresentation representation =
                    Objects.requireNonNull(callerInputs.get(index), "callerInputs[" + index + "]");
            for (int earlierIndex = 0; earlierIndex < index; earlierIndex++) {
                if (callerInputs.get(earlierIndex) == representation) {
                    throw new IllegalArgumentException(
                            "representation is already bound to this run");
                }
            }
        }

        var createdRepresentations =
                new AutoCloseable[
                        createdBufferCount + representationPlan.workspaceCreators().size()];
        int createdCount = 0;
        try {
            var bufferBindings =
                    new ArrayList<List<BufferRepresentationBinding>>(
                            representationPlan.bufferPreparations().size());
            int callerInputIndex = 0;
            for (int bufferIndex = 0;
                    bufferIndex < representationPlan.bufferPreparations().size();
                    bufferIndex++) {
                List<PreparedRepresentationPlan.BufferPreparation> preparations =
                        representationPlan.bufferPreparations().get(bufferIndex);
                var bindings = new ArrayList<BufferRepresentationBinding>(preparations.size());
                for (int representationIndex = 0;
                        representationIndex < preparations.size();
                        representationIndex++) {
                    PreparedRepresentationPlan.BufferPreparation preparation =
                            preparations.get(representationIndex);
                    if (preparation instanceof CallerInput) {
                        bindings.add(
                                new BufferRepresentationBinding(
                                        callerInputs.get(callerInputIndex++),
                                        RunResourceOwnership.BORROWED));
                    } else {
                        BufferRepresentation representation =
                                Objects.requireNonNull(
                                        ((CreatedBuffer) preparation).creator().create(),
                                        "bufferPreparations["
                                                + bufferIndex
                                                + "]["
                                                + representationIndex
                                                + "] creator result");
                        rejectRepeatedIdentity(
                                callerInputs,
                                createdRepresentations,
                                createdCount,
                                representation);
                        createdRepresentations[createdCount++] = representation;
                        bindings.add(
                                new BufferRepresentationBinding(
                                        representation, RunResourceOwnership.RUN_OWNED));
                    }
                }
                bufferBindings.add(List.copyOf(bindings));
            }

            var workspaces =
                    new ArrayList<WorkspaceRepresentation>(
                            representationPlan.workspaceCreators().size());
            for (int workspaceIndex = 0;
                    workspaceIndex < representationPlan.workspaceCreators().size();
                    workspaceIndex++) {
                WorkspaceRepresentation representation =
                        Objects.requireNonNull(
                                representationPlan.workspaceCreators().get(workspaceIndex).create(),
                                "workspaceCreators[" + workspaceIndex + "] result");
                rejectRepeatedIdentity(
                        callerInputs,
                        createdRepresentations,
                        createdCount,
                        representation);
                createdRepresentations[createdCount++] = representation;
                workspaces.add(representation);
            }

            return new RunState(
                    representationPlan.memoryPlan(), bufferBindings, workspaces);
        } catch (RuntimeException | Error failure) {
            rollback(createdRepresentations, createdCount, failure);
            throw failure;
        }
    }

    private static void rejectRepeatedIdentity(
            List<BufferRepresentation> callerInputs,
            AutoCloseable[] createdRepresentations,
            int createdCount,
            AutoCloseable representation) {
        for (BufferRepresentation callerInput : callerInputs) {
            if (callerInput == representation) {
                throw new IllegalArgumentException(
                        "representation is already bound to this run");
            }
        }
        for (int index = 0; index < createdCount; index++) {
            if (createdRepresentations[index] == representation) {
                throw new IllegalArgumentException(
                        "representation is already bound to this run");
            }
        }
    }

    private static void rollback(
            AutoCloseable[] createdRepresentations, int createdCount, Throwable failure) {
        for (int index = createdCount - 1; index >= 0; index--) {
            try {
                AutoCloseable representation = createdRepresentations[index];
                if (representation instanceof BufferRepresentation buffer) {
                    buffer.close();
                } else {
                    ((WorkspaceRepresentation) representation).close();
                }
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
        }
    }
}
