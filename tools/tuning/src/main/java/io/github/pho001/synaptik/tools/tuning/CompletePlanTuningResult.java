package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable selected handoff, compact selected-plan record, and separate rich transaction
 * evidence.
 *
 * <p>The selected handoff contains the producer-owned decision associated with the request's
 * exact partition and candidate-batch references. The compact record carries compatibility,
 * winner identity, and timing summary without raw samples. Rich evidence is returned in memory
 * and is not written to the model-plan cache.
 *
 * @param <C> immutable complete candidate-batch type transported by the selected handoff
 * @param <D> immutable selected-decision type transported by the selected handoff
 */
public final class CompletePlanTuningResult<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision> {
    /** Source of the selected decision. */
    public enum Source {
        /** A persistent entry was fully validated and its decision authenticated by the producer. */
        CACHE_HIT,
        /** All candidates were checked and measured in the current transaction. */
        MEASURED
    }

    /** Correctness action recorded for a candidate. */
    public enum CorrectnessAction {
        /** The first candidate produced the caller-owned opaque reference. */
        REFERENCE_CAPTURED,
        /** A later candidate exactly matched that reference. */
        MATCH
    }

    /**
     * Compact bounded timing summary for one candidate.
     *
     * @param minimumNanos non-negative minimum elapsed time in nanoseconds
     * @param medianNanos non-negative integer-middle median elapsed time in nanoseconds
     * @param maximumNanos non-negative maximum elapsed time in nanoseconds
     * @param sampleCount positive number of timed samples represented by the summary
     */
    public record SampleSummary(long minimumNanos, long medianNanos, long maximumNanos,
            int sampleCount) {
        /**
         * Validates and creates a timing summary.
         *
         * @param minimumNanos non-negative minimum elapsed time in nanoseconds
         * @param medianNanos non-negative median not smaller than {@code minimumNanos}
         * @param maximumNanos non-negative maximum not smaller than {@code medianNanos}
         * @param sampleCount positive timed-sample count
         * @throws IllegalArgumentException if times are negative or unordered, or the sample count
         *     is not positive
         */
        public SampleSummary {
            if (minimumNanos < 0
                    || minimumNanos > medianNanos
                    || medianNanos > maximumNanos) {
                throw new IllegalArgumentException("summary values must be non-negative and ordered");
            }
            if (sampleCount <= 0) {
                throw new IllegalArgumentException("sampleCount must be positive");
            }
        }
    }

    /**
     * Rich immutable evidence for one measured candidate.
     *
     * @param identity non-null opaque identity of the measured candidate
     * @param correctnessAction non-null correctness action completed before timing
     * @param elapsedSamplesNanos non-null, non-empty elapsed-time list in measurement order; its
     *     structure is snapshotted and every value is non-null and non-negative
     * @param summary non-null summary exactly matching the copied sample values
     */
    public record CandidateEvidence(CompletePlanTuningRequest.CandidateIdentity identity,
            CorrectnessAction correctnessAction, List<Long> elapsedSamplesNanos,
            SampleSummary summary) {
        /**
         * Snapshots and validates measured-candidate evidence.
         *
         * @param identity non-null immutable candidate identity
         * @param correctnessAction non-null completed correctness action
         * @param elapsedSamplesNanos non-null, non-empty elapsed nanoseconds in measurement order
         * @param summary non-null summary matching count, minimum, integer-middle median, and
         *     maximum of the samples
         * @throws NullPointerException if an argument other than a sample element is null
         * @throws IllegalArgumentException if samples are empty, contain a null or negative value,
         *     or do not exactly match the summary
         */
        public CandidateEvidence {
            Objects.requireNonNull(identity, "identity");
            Objects.requireNonNull(correctnessAction, "correctnessAction");
            Objects.requireNonNull(summary, "summary");
            var copy = new ArrayList<>(
                    Objects.requireNonNull(elapsedSamplesNanos, "elapsedSamplesNanos"));
            if (copy.isEmpty()) {
                throw new IllegalArgumentException("elapsed samples must not be empty");
            }
            for (Long sample : copy) {
                if (sample == null || sample < 0) {
                    throw new IllegalArgumentException(
                            "elapsed samples must be non-null and non-negative");
                }
            }
            elapsedSamplesNanos = List.copyOf(copy);
            if (copy.size() != summary.sampleCount()) {
                throw new IllegalArgumentException("sample count mismatch");
            }
            copy.sort(Long::compare);
            if (copy.getFirst() != summary.minimumNanos()
                    || copy.get(copy.size() / 2) != summary.medianNanos()
                    || copy.getLast() != summary.maximumNanos()) {
                throw new IllegalArgumentException("samples do not match summary");
            }
        }
    }

    /**
     * Compact selected-plan state containing no raw samples or correctness payload.
     *
     * @param compatibility non-null immutable compatibility used for this selection
     * @param selectedCandidateIdentity non-null opaque identity of the winning candidate
     * @param timingSummary non-null compact winning timing summary
     */
    public record SelectedPlanRecord(CompletePlanTuningRequest.PlanCompatibility compatibility,
            CompletePlanTuningRequest.CandidateIdentity selectedCandidateIdentity,
            SampleSummary timingSummary) {
        /**
         * Creates a compact selected-plan record.
         *
         * @param compatibility non-null immutable plan compatibility
         * @param selectedCandidateIdentity non-null immutable winner identity
         * @param timingSummary non-null immutable winner summary
         * @throws NullPointerException if any component is null
         */
        public SelectedPlanRecord {
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(selectedCandidateIdentity, "selectedCandidateIdentity");
            Objects.requireNonNull(timingSummary, "timingSummary");
        }
    }

    /**
     * Rich immutable transaction evidence, never persisted in the compact cache.
     *
     * @param modelFingerprint non-null model fingerprint from the request
     * @param profileFingerprint non-null representative-profile fingerprint from the request
     * @param targetFingerprint non-null execution-target fingerprint from the request
     * @param objective non-null timing objective
     * @param correctnessPolicy non-null correctness policy
     * @param policyIdentity non-null constraint/policy identity
     * @param source non-null source of the selected decision
     * @param candidates measured candidates in encounter order, snapshotted as an immutable list;
     *     non-empty for {@link Source#MEASURED} and empty for {@link Source#CACHE_HIT}
     * @param selectedCandidateIdentity non-null selected candidate identity
     * @param selectedSummary non-null selected candidate timing summary
     */
    public record Evidence(CompletePlanTuningRequest.ModelFingerprint modelFingerprint,
            CompletePlanTuningRequest.ProfileFingerprint profileFingerprint,
            CompletePlanTuningRequest.TargetFingerprint targetFingerprint,
            CompletePlanTuningRequest.Objective objective,
            CompletePlanTuningRequest.CorrectnessPolicy correctnessPolicy,
            CompletePlanTuningRequest.PolicyIdentity policyIdentity,
            Source source, List<CandidateEvidence> candidates,
            CompletePlanTuningRequest.CandidateIdentity selectedCandidateIdentity,
            SampleSummary selectedSummary) {
        /**
         * Snapshots and validates rich transaction evidence.
         *
         * @param modelFingerprint non-null immutable model fingerprint
         * @param profileFingerprint non-null immutable representative-profile fingerprint
         * @param targetFingerprint non-null immutable target fingerprint
         * @param objective non-null timing objective
         * @param correctnessPolicy non-null correctness policy
         * @param policyIdentity non-null immutable constraint/policy identity
         * @param source non-null decision source
         * @param candidates non-null measured-candidate list with source-dependent emptiness
         * @param selectedCandidateIdentity non-null winner identity
         * @param selectedSummary non-null winner summary
         * @throws NullPointerException if any required component is null
         * @throws IllegalArgumentException if measured evidence has no candidates or cache-hit
         *     evidence contains candidates
         */
        public Evidence {
            Objects.requireNonNull(modelFingerprint, "modelFingerprint");
            Objects.requireNonNull(profileFingerprint, "profileFingerprint");
            Objects.requireNonNull(targetFingerprint, "targetFingerprint");
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(correctnessPolicy, "correctnessPolicy");
            Objects.requireNonNull(policyIdentity, "policyIdentity");
            Objects.requireNonNull(source, "source");
            candidates = List.copyOf(Objects.requireNonNull(candidates, "candidates"));
            if ((source == Source.MEASURED) != !candidates.isEmpty()) {
                throw new IllegalArgumentException("measured evidence requires candidates and cache hits require none");
            }
            Objects.requireNonNull(selectedCandidateIdentity, "selectedCandidateIdentity");
            Objects.requireNonNull(selectedSummary, "selectedSummary");
        }
    }

    private final BackendPartitionTuningHandoff<C, D> selectedHandoff;
    private final SelectedPlanRecord selectedPlan;
    private final Evidence evidence;

    /**
     * Creates a successful immutable transaction result.
     * @param selectedHandoff non-null decision-present exact handoff
     * @param selectedPlan non-null compact selected-plan record
     * @param evidence non-null separate rich in-memory evidence
     * @throws NullPointerException if any argument is null
     * @throws IllegalArgumentException if {@code selectedHandoff} contains no selected decision
     */
    public CompletePlanTuningResult(BackendPartitionTuningHandoff<C, D> selectedHandoff,
            SelectedPlanRecord selectedPlan, Evidence evidence) {
        this.selectedHandoff = Objects.requireNonNull(selectedHandoff, "selectedHandoff");
        if (selectedHandoff.selectedDecision().isEmpty()) {
            throw new IllegalArgumentException("selected handoff must contain a decision");
        }
        this.selectedPlan = Objects.requireNonNull(selectedPlan, "selectedPlan");
        this.evidence = Objects.requireNonNull(evidence, "evidence");
    }

    /**
     * Returns the selected Prepare handoff.
     *
     * @return the non-null decision-present exact typed handoff retained by reference
     */
    public BackendPartitionTuningHandoff<C, D> selectedHandoff() {
        return selectedHandoff;
    }

    /**
     * Returns compact selected-plan state.
     *
     * @return the non-null immutable compact selected-plan record retained by reference
     */
    public SelectedPlanRecord selectedPlan() {
        return selectedPlan;
    }

    /**
     * Returns rich transaction evidence.
     *
     * @return the non-null immutable rich evidence, separate from cache state
     */
    public Evidence evidence() {
        return evidence;
    }
}
