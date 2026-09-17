package io.github.pho001.synaptik.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable detached result of one scalar-objective backward convenience call.
 *
 * <p>The objective is the requested scalar forward value. Gradient position {@code i} is the
 * first derivative for target position {@code i} in the corresponding request. Every value owns
 * only immutable canonical host bytes, so the result has no close lifecycle and remains readable
 * after the Engine, temporary run result, and caller storage have closed. Distinct occurrences
 * remain distinct values even when their inward representations alias.</p>
 */
public final class ScalarObjectiveBackwardResult {
    private final HostTensorValue objective;
    private final List<HostTensorValue> gradients;

    /**
     * Retains already-detached objective and target-aligned gradient values.
     *
     * @param objective non-null detached scalar forward value retained by exact reference
     * @param gradients non-null ordered list of non-null detached gradient values; membership is
     *     snapshotted while each value is retained by exact reference
     * @throws NullPointerException if the objective, list, or an indexed gradient is {@code null},
     *     checked in that order
     */
    ScalarObjectiveBackwardResult(
            HostTensorValue objective, List<HostTensorValue> gradients) {
        this.objective = Objects.requireNonNull(objective, "objective");
        Objects.requireNonNull(gradients, "gradients");
        var snapshot = new ArrayList<HostTensorValue>(gradients.size());
        for (int index = 0; index < gradients.size(); index++) {
            snapshot.add(Objects.requireNonNull(
                    gradients.get(index), "gradients[" + index + "]"));
        }
        this.gradients = List.copyOf(snapshot);
    }

    /**
     * Returns the detached scalar forward occurrence.
     *
     * @return the exact non-null immutable host value retained at construction
     */
    public HostTensorValue objective() {
        return objective;
    }

    /**
     * Returns first-order gradients aligned with the original target-list positions.
     *
     * @return the same non-null immutable ordered list on every call
     */
    public List<HostTensorValue> gradients() {
        return gradients;
    }
}
