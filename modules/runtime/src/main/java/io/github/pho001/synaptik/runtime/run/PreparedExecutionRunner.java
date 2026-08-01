package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Runs one complete immutable prepared execution synchronously against an isolated mutable state.
 *
 * <p>Each call creates exactly one state, cold-binds every executable, transfer, and publication
 * occurrence before the first action, then traverses a private direct-reference array in schedule
 * order. Successful completion returns a {@link RunResult} that leases the still-open state;
 * every failure after state creation closes it once before the original unchecked failure is
 * rethrown.
 *
 * <p>This runner is stateless and thread-safe. Separate calls may share the immutable recipe but
 * share no runner-created mutable state. One call is synchronous and uses one orchestrating
 * thread. Runtime performs no backend discovery, graph inspection, route selection, physical
 * allocation, retry, tracing, or result conversion.
 */
public final class PreparedExecutionRunner {
    /**
     * Creates a stateless prepared-execution runner.
     *
     * <p>Construction acquires no resource and retains no configuration or mutable run state.
     */
    public PreparedExecutionRunner() {}

    /**
     * Executes the prepared schedule once and leases its complete state to a result.
     *
     * <p>Caller inputs are dense borrowed representations and remain caller-owned. A non-empty
     * memory plan requires the schedule's first occurrence to create representations. A wholly
     * empty plan and empty caller list may run without that occurrence. All remaining occurrences
     * bind before any invocation, transfer, or publication action begins.
     *
     * @param execution the immutable prepared execution to run; must be non-null
     * @param callerInputs dense borrowed inputs in caller-preparation encounter order; must be
     *     non-null and satisfy the representation-creation plan, or be empty when no creation
     *     occurrence exists
     * @return a non-null result leasing the exact open state and ordered publications; closing it
     *     releases resources still owned by the run
     * @throws NullPointerException if {@code execution}, {@code callerInputs}, or an existing
     *     creation, binding, or result contract input is {@code null}
     * @throws IllegalArgumentException if creation is absent for non-empty memory, caller inputs
     *     are supplied without creation, or an existing creation or binding contract rejects an
     *     input
     * @throws IllegalStateException if an executable read is invalid or an existing action,
     *     publication, or result contract rejects the current state
     * @throws RuntimeException if prepared backend work or cleanup reports an unchecked failure
     * @throws Error if prepared backend work or cleanup reports an error
     */
    public RunResult run(
            PreparedExecution execution,
            List<BufferRepresentation> callerInputs) {
        Objects.requireNonNull(execution, "execution");
        Objects.requireNonNull(callerInputs, "callerInputs");

        List<PreparedSchedule.Step> steps = execution.schedule().steps();
        RunState state;
        int firstBoundIndex;
        if (!steps.isEmpty()
                && steps.getFirst() instanceof PreparedSchedule.RepresentationCreationStep creation) {
            state = RunStateCreation.create(creation.representationPlan(), callerInputs);
            firstBoundIndex = 1;
        } else {
            if (!execution.memoryPlan().buffers().isEmpty()
                    || !execution.memoryPlan().workspaces().isEmpty()) {
                throw new IllegalArgumentException(
                        "non-empty prepared memory plan requires a representation creation occurrence");
            }
            if (!callerInputs.isEmpty()) {
                throw new IllegalArgumentException(
                        "callerInputs size must equal caller-input preparation count 0");
            }
            state = new RunState(execution.memoryPlan(), List.of(), List.of());
            firstBoundIndex = 0;
        }

        try {
            BoundStep[] boundSteps = new BoundStep[steps.size() - firstBoundIndex];
            BoundPublication[] publications =
                    new BoundPublication[execution.schedule().publicationCount()];
            int publicationIndex = 0;
            for (int index = firstBoundIndex; index < steps.size(); index++) {
                PreparedSchedule.Step step = steps.get(index);
                BoundStep boundStep;
                if (step instanceof PreparedSchedule.ExecutionStep executionStep) {
                    boundStep = bindExecutable(executionStep.executable(), state);
                } else if (step instanceof PreparedSchedule.BufferTransferStep transferStep) {
                    boundStep = new TransferStep(transferStep.transfer().bind(state));
                } else {
                    BoundPublication publication =
                            ((PreparedSchedule.PublicationStep) step).publication().bind(state);
                    publications[publicationIndex++] = publication;
                    boundStep = new PublicationAction(publication);
                }
                boundSteps[index - firstBoundIndex] = boundStep;
            }

            for (BoundStep boundStep : boundSteps) {
                boundStep.execute();
            }
            return new RunResult(state, Arrays.asList(publications));
        } catch (RuntimeException | Error failure) {
            closeAfterFailure(state, failure);
            throw failure;
        }
    }

    private static BoundExecutable bindExecutable(
            PreparedExecutable executable, RunState state) {
        BoundInvocation invocation = executable.bind(state);
        int selectionCount = executable.bufferSelectionCount();
        int[] readSelectionIndices = new int[selectionCount];
        int[] readBufferIndices = new int[selectionCount];
        int[] readRepresentationIndices = new int[selectionCount];
        int readCount = 0;
        int[] outputBufferIndices = new int[selectionCount];
        int outputCount = 0;
        int[] writtenBufferIndices = new int[selectionCount];
        int[] writtenRepresentationIndices = new int[selectionCount];
        int writtenCount = 0;

        for (int selectionIndex = 0; selectionIndex < selectionCount; selectionIndex++) {
            PreparedExecutable.BufferSelection selection =
                    executable.bufferSelection(selectionIndex);
            PreparedExecutable.BufferAccess access = executable.bufferAccess(selectionIndex);
            if (access != PreparedExecutable.BufferAccess.WRITE_ONLY) {
                readSelectionIndices[readCount] = selectionIndex;
                readBufferIndices[readCount] = selection.bufferIndex();
                readRepresentationIndices[readCount] = selection.representationIndex();
                readCount++;
            }
            if (access != PreparedExecutable.BufferAccess.READ_ONLY) {
                if (!contains(outputBufferIndices, outputCount, selection.bufferIndex())) {
                    outputBufferIndices[outputCount++] = selection.bufferIndex();
                }
                if (!containsPair(
                        writtenBufferIndices,
                        writtenRepresentationIndices,
                        writtenCount,
                        selection.bufferIndex(),
                        selection.representationIndex())) {
                    writtenBufferIndices[writtenCount] = selection.bufferIndex();
                    writtenRepresentationIndices[writtenCount] = selection.representationIndex();
                    writtenCount++;
                }
            }
        }
        return new BoundExecutable(
                state,
                invocation,
                Arrays.copyOf(readSelectionIndices, readCount),
                Arrays.copyOf(readBufferIndices, readCount),
                Arrays.copyOf(readRepresentationIndices, readCount),
                Arrays.copyOf(outputBufferIndices, outputCount),
                Arrays.copyOf(writtenBufferIndices, writtenCount),
                Arrays.copyOf(writtenRepresentationIndices, writtenCount));
    }

    private static boolean contains(int[] values, int count, int candidate) {
        for (int index = 0; index < count; index++) {
            if (values[index] == candidate) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsPair(
            int[] first, int[] second, int count, int firstCandidate, int secondCandidate) {
        for (int index = 0; index < count; index++) {
            if (first[index] == firstCandidate && second[index] == secondCandidate) {
                return true;
            }
        }
        return false;
    }

    private static void closeAfterFailure(RunState state, Throwable failure) {
        try {
            state.close();
        } catch (RuntimeException | Error cleanupFailure) {
            if (cleanupFailure != failure) {
                failure.addSuppressed(cleanupFailure);
            }
        }
    }

    private interface BoundStep {
        void execute();
    }

    private static final class BoundExecutable implements BoundStep {
        private final RunState state;
        private final BoundInvocation invocation;
        private final int[] readSelectionIndices;
        private final int[] readBufferIndices;
        private final int[] readRepresentationIndices;
        private final int[] outputBufferIndices;
        private final int[] writtenBufferIndices;
        private final int[] writtenRepresentationIndices;

        private BoundExecutable(
                RunState state,
                BoundInvocation invocation,
                int[] readSelectionIndices,
                int[] readBufferIndices,
                int[] readRepresentationIndices,
                int[] outputBufferIndices,
                int[] writtenBufferIndices,
                int[] writtenRepresentationIndices) {
            this.state = state;
            this.invocation = invocation;
            this.readSelectionIndices = readSelectionIndices;
            this.readBufferIndices = readBufferIndices;
            this.readRepresentationIndices = readRepresentationIndices;
            this.outputBufferIndices = outputBufferIndices;
            this.writtenBufferIndices = writtenBufferIndices;
            this.writtenRepresentationIndices = writtenRepresentationIndices;
        }

        @Override
        public void execute() {
            for (int index = 0; index < readSelectionIndices.length; index++) {
                if (!state.isBufferRepresentationValid(
                        readBufferIndices[index], readRepresentationIndices[index])) {
                    throw new IllegalStateException(
                            "executable buffer selection "
                                    + readSelectionIndices[index]
                                    + " requires a valid input representation");
                }
            }
            for (int bufferIndex : outputBufferIndices) {
                int representationCount = state.bufferRepresentationCount(bufferIndex);
                for (int representationIndex = 0;
                        representationIndex < representationCount;
                        representationIndex++) {
                    state.setBufferRepresentationValid(
                            bufferIndex, representationIndex, false);
                }
            }
            invocation.execute();
            for (int index = 0; index < writtenBufferIndices.length; index++) {
                state.setBufferRepresentationValid(
                        writtenBufferIndices[index],
                        writtenRepresentationIndices[index],
                        true);
            }
        }
    }

    private static final class TransferStep implements BoundStep {
        private final BoundBufferTransfer transfer;

        private TransferStep(BoundBufferTransfer transfer) {
            this.transfer = transfer;
        }

        @Override
        public void execute() {
            transfer.execute();
        }
    }

    private static final class PublicationAction implements BoundStep {
        private final BoundPublication publication;

        private PublicationAction(BoundPublication publication) {
            this.publication = publication;
        }

        @Override
        public void execute() {
            publication.publish();
        }
    }
}
