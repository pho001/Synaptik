package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.engine.HostTensorValue;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Immutable detached result of one successful {@link TrainingSession#run} call.
 *
 * <p>The execution number starts at one and counts successful forward/backward executions. The
 * optional optimizer step number is present only when the selected gradient mode committed a
 * parameter update; those numbers independently start at one. The objective owns detached
 * canonical host bytes, so every accessor remains usable after the session, Engine, caller input
 * storage, or parameter storage closes or becomes unreachable.</p>
 */
public final class TrainingStep {
    private final long executionNumber;
    private final OptionalLong optimizerStep;
    private final long accumulatedGradientRuns;
    private final HostTensorValue objective;

    TrainingStep(
            long executionNumber,
            OptionalLong optimizerStep,
            long accumulatedGradientRuns,
            HostTensorValue objective) {
        if (executionNumber <= 0) {
            throw new IllegalArgumentException("executionNumber must be positive");
        }
        this.executionNumber = executionNumber;
        this.optimizerStep = Objects.requireNonNull(optimizerStep, "optimizerStep");
        if (optimizerStep.isPresent() && optimizerStep.orElseThrow() <= 0) {
            throw new IllegalArgumentException("optimizerStep must be positive when present");
        }
        if (accumulatedGradientRuns < 0) {
            throw new IllegalArgumentException("accumulatedGradientRuns must be non-negative");
        }
        this.accumulatedGradientRuns = accumulatedGradientRuns;
        this.objective = Objects.requireNonNull(objective, "objective");
    }

    /**
     * Returns this successful forward/backward execution's sequence number.
     *
     * @return the positive one-based execution number
     */
    public long executionNumber() {
        return executionNumber;
    }

    /**
     * Returns the optimizer update number committed by this run, when any.
     *
     * @return a non-null optional containing the positive one-based optimizer step for an updating
     *     mode, or empty for {@link GradientMode#ACCUMULATE}
     */
    public OptionalLong optimizerStep() {
        return optimizerStep;
    }

    /**
     * Returns how many successful gradient runs remain pending after this call.
     *
     * @return a positive count after accumulate-only success, otherwise zero
     */
    public long accumulatedGradientRuns() {
        return accumulatedGradientRuns;
    }

    /**
     * Returns the detached scalar objective produced before any optional update.
     *
     * @return the exact non-null immutable host value retained by this result
     */
    public HostTensorValue objective() {
        return objective;
    }
}
