package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable opaque tuning transport associated with one exact planned partition.
 *
 * <p>The handoff retains the exact candidate-batch reference and, when present, the exact
 * selected-decision reference. It takes no ownership, snapshots no backend state, and performs no
 * candidate or decision interpretation. Its type parameters preserve the concrete backend's
 * typed collaboration for a caller that already owns both types.</p>
 *
 * <p>Record equality, hashing, and diagnostic text use exactly the three declared components.
 * The diagnostic text is not a serialization or persistent tuning format.</p>
 *
 * @param <C> concrete backend-owned immutable complete candidate-batch role
 * @param <D> concrete backend-owned immutable selected-decision role
 * @param partition non-null exact immutable planned-partition reference
 * @param candidateBatch non-null exact immutable backend-owned candidate-batch reference
 * @param selectedDecision non-null optional containing the exact immutable backend-owned
 *     decision reference, or empty when no decision has been selected
 */
public record BackendPartitionTuningHandoff<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision>(
        PlannedPartition partition,
        C candidateBatch,
        Optional<D> selectedDecision) {
    /**
     * Validates one exact-partition opaque handoff without copying or interpreting backend state.
     *
     * @param partition non-null exact planned-partition reference to retain
     * @param candidateBatch non-null exact backend candidate-batch reference to retain opaquely
     * @param selectedDecision non-null optional whose contained decision, when present, must be
     *     non-null and is retained by exact reference
     * @throws NullPointerException if any component is {@code null}; an {@link Optional} itself
     *     cannot be constructed with a null contained value
     */
    public BackendPartitionTuningHandoff {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(candidateBatch, "candidateBatch");
        Objects.requireNonNull(selectedDecision, "selectedDecision")
                .ifPresent(decision -> Objects.requireNonNull(decision, "selectedDecision value"));
    }
}
