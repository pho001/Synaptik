package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Stateless entry point for one bounded complete-plan tuning transaction.
 *
 * <p>A session-scoped transaction performs no filesystem operation. A persistent transaction
 * completely validates its explicit cache before candidate enumeration; a hit performs no
 * enumeration, correctness execution, warmup, or sample after the producer authenticates the
 * decoded decision. A miss validates the complete batch and checked execution budget, completes
 * all correctness actions, then performs bounded measurement. No successful result is exposed
 * before any required persistent publication succeeds.
 */
public final class CompletePlanTuning {
    private static final int OBJECTIVE_MIN_MEDIAN = 1;
    private static final int CORRECTNESS_EXACT_BYTES = 1;
    private CompletePlanTuning() {}

    @FunctionalInterface
    interface NanoClock {
        long nanoTime();
    }

    /**
     * Signals a clean exact correctness mismatch after caller-owned comparison and cleanup
     * succeed. Caller execution or cleanup failures retain their original exception instead.
     */
    public static final class CorrectnessMismatchException extends Exception {
        private CorrectnessMismatchException() {
            super("complete-plan candidate correctness mismatch");
        }
    }

    /**
     * Authenticates a persistent hit or validates, checks, and measures a complete miss batch.
     *
     * <p>For a miss with {@code N} candidates, {@code W} warmups, and {@code S} timed samples,
     * this method preflights exactly {@code N * (1 + W + S)} complete executions before the first
     * correctness callback. Correctness capture/comparison for all candidates precedes every
     * warmup and timed sample. Selection minimizes the integer-middle median elapsed nanoseconds
     * and preserves encounter order on ties.
     *
     * @param request non-null immutable request
     * @param backend non-null backend-owned candidate and codec collaboration
     * @param correctness non-null caller-owned exact correctness collaboration
     * @param measurement non-null caller-owned fresh complete execution action
     * @param <C> immutable complete candidate-batch type
     * @param <D> immutable selected-decision type
     * @param <K> immutable complete candidate type
     * @param <R> opaque transient correctness-reference type
     * @return non-null immutable result containing the exact decision-present handoff, compact
     *     selected-plan record, and separate rich evidence; cache-hit evidence contains no raw
     *     candidate samples
     * @throws NullPointerException if the request or a collaboration is null, or a collaboration
     *     returns a required null value
     * @throws IllegalArgumentException if the enumerated candidates are empty, duplicated,
     *     identity-unstable, identity-duplicated, over the candidate ceiling, or over the checked
     *     total-execution ceiling
     * @throws IllegalStateException if elapsed time is negative or a persistent decision fails an
     *     equal compatible codec round trip
     * @throws CorrectnessMismatchException after a clean exact mismatch
     * @throws IOException if persistent cache validation or publication fails
     * @throws Exception if a caller collaboration reports another checked failure
     */
    public static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision, K, R>
            CompletePlanTuningResult<C, D> tune(
                    CompletePlanTuningRequest<C, D> request,
                    BackendCompletePlanTuning<C, D, K> backend,
                    CompletePlanCorrectness<C, K, R> correctness,
                    CompleteCandidateMeasurement<C, K> measurement)
                    throws Exception {
        return tune(request, backend, correctness, measurement, System::nanoTime);
    }

    static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision, K, R>
            CompletePlanTuningResult<C, D> tune(
                    CompletePlanTuningRequest<C, D> request,
                    BackendCompletePlanTuning<C, D, K> backend,
                    CompletePlanCorrectness<C, K, R> correctness,
                    CompleteCandidateMeasurement<C, K> measurement,
                    NanoClock clock)
                    throws Exception {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(correctness, "correctness");
        Objects.requireNonNull(measurement, "measurement");
        Objects.requireNonNull(clock, "clock");
        C batch = request.handoff().candidateBatch();
        CompletePlanTuningRequest.PlanCompatibility compatibility =
                Objects.requireNonNull(
                        backend.compatibility(batch), "compatibility result");
        ModelPlanCacheFile.Key key = key(request, compatibility);

        if (compatibility.reuseScope() == CompletePlanTuningRequest.ReuseScope.PERSISTENT) {
            ModelPlanCacheFile cache = new ModelPlanCacheFile();
            Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> loaded =
                    cache.load(request.modelPlanCache());
            ModelPlanCacheFile.Entry entry = loaded.get(key);
            if (entry != null) {
                Optional<D> decoded =
                        Objects.requireNonNull(
                                backend.decodeCompatibleDecision(batch, entry.decision()),
                                "decodeCompatibleDecision result");
                if (decoded.isPresent()) {
                    return result(
                            request,
                            compatibility,
                            Objects.requireNonNull(decoded.orElseThrow(), "decoded decision"),
                            new CompletePlanTuningRequest.CandidateIdentity(entry.winner()),
                            entry.summary(),
                            CompletePlanTuningResult.Source.CACHE_HIT,
                            List.of());
                }
            }
            MeasurementOutcome<D> outcome =
                    measure(request, backend, correctness, measurement, clock, batch);
            byte[] encoded = boundedEncoding(backend.encodeDecision(outcome.decision));
            Optional<D> roundTrip =
                    Objects.requireNonNull(
                            backend.decodeCompatibleDecision(batch, encoded),
                            "decodeCompatibleDecision result");
            if (roundTrip.isEmpty() || !outcome.decision.equals(roundTrip.orElseThrow())) {
                throw new IllegalStateException("persistent decision did not round trip compatibly");
            }
            Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> updated =
                    new LinkedHashMap<>(loaded);
            updated.put(
                    key,
                    new ModelPlanCacheFile.Entry(
                            key,
                            encoded,
                            outcome.winnerIdentity.bytes(),
                            outcome.summary));
            cache.publish(request.modelPlanCache(), updated);
            return result(
                    request,
                    compatibility,
                    outcome.decision,
                    outcome.winnerIdentity,
                    outcome.summary,
                    CompletePlanTuningResult.Source.MEASURED,
                    outcome.evidence);
        }

        MeasurementOutcome<D> outcome =
                measure(request, backend, correctness, measurement, clock, batch);
        return result(
                request,
                compatibility,
                outcome.decision,
                outcome.winnerIdentity,
                outcome.summary,
                CompletePlanTuningResult.Source.MEASURED,
                outcome.evidence);
    }

    private static <
                    C extends BackendTuningCandidateBatch,
                    D extends BackendTuningDecision,
                    K,
                    R>
            MeasurementOutcome<D> measure(
                    CompletePlanTuningRequest<C, D> request,
                    BackendCompletePlanTuning<C, D, K> backend,
                    CompletePlanCorrectness<C, K, R> correctness,
                    CompleteCandidateMeasurement<C, K> measurement,
                    NanoClock clock,
                    C batch)
                    throws Exception {
        List<K> candidates =
                List.copyOf(
                        Objects.requireNonNull(backend.candidates(batch), "candidates result"));
        int count = candidates.size();
        if (count == 0) {
            throw new IllegalArgumentException("candidate list must not be empty");
        }
        if (count > request.maximumPlanCandidates()) {
            throw new IllegalArgumentException("candidate count exceeds ceiling");
        }
        checkedTotalExecutions(
                count,
                request.warmupCount(),
                request.timedSampleCount(),
                request.maximumTotalPlanExecutions());

        List<CompletePlanTuningRequest.CandidateIdentity> identities = new ArrayList<>(count);
        Set<K> uniqueCandidates = new HashSet<>();
        Set<CompletePlanTuningRequest.CandidateIdentity> uniqueIdentities = new HashSet<>();
        for (K candidate : candidates) {
            Objects.requireNonNull(candidate, "candidate");
            if (!uniqueCandidates.add(candidate)) {
                throw new IllegalArgumentException("candidate is duplicated");
            }
            var first =
                    Objects.requireNonNull(
                            backend.candidateIdentity(candidate), "candidate identity");
            var second =
                    Objects.requireNonNull(
                            backend.candidateIdentity(candidate),
                            "candidate identity stability");
            if (!first.equals(second)) {
                throw new IllegalArgumentException("candidate identity is unstable");
            }
            if (!uniqueIdentities.add(first)) {
                throw new IllegalArgumentException("candidate identity is duplicated");
            }
            identities.add(first);
        }

        List<CompletePlanTuningResult.CorrectnessAction> actions = new ArrayList<>(count);
        R reference =
                Objects.requireNonNull(
                        correctness.capture(
                                batch,
                                candidates.getFirst(),
                                request.maximumAggregateCorrectnessBytes()),
                        "correctness reference");
        actions.add(CompletePlanTuningResult.CorrectnessAction.REFERENCE_CAPTURED);
        for (int i = 1; i < count; i++) {
            CompletePlanCorrectness.Outcome outcome =
                    Objects.requireNonNull(
                            correctness.compare(
                                    reference,
                                    batch,
                                    candidates.get(i),
                                    request.maximumAggregateCorrectnessBytes()),
                            "correctness outcome");
            if (outcome == CompletePlanCorrectness.Outcome.MISMATCH) {
                throw new CorrectnessMismatchException();
            }
            actions.add(CompletePlanTuningResult.CorrectnessAction.MATCH);
        }

        List<CompletePlanTuningResult.CandidateEvidence> evidence = new ArrayList<>(count);
        long best = Long.MAX_VALUE;
        int winner = -1;
        CompletePlanTuningResult.SampleSummary winnerSummary = null;
        for (int i = 0; i < count; i++) {
            K candidate = candidates.get(i);
            for (int warmup = 0; warmup < request.warmupCount(); warmup++) {
                measurement.execute(batch, candidate);
            }
            List<Long> samples = new ArrayList<>(request.timedSampleCount());
            for (int sample = 0; sample < request.timedSampleCount(); sample++) {
                long start = clock.nanoTime();
                measurement.execute(batch, candidate);
                long end = clock.nanoTime();
                long elapsed = Math.subtractExact(end, start);
                if (elapsed < 0) {
                    throw new IllegalStateException("monotonic clock moved backwards");
                }
                samples.add(elapsed);
            }
            CompletePlanTuningResult.SampleSummary summary = summary(samples);
            evidence.add(
                    new CompletePlanTuningResult.CandidateEvidence(
                            identities.get(i), actions.get(i), samples, summary));
            if (winner < 0 || summary.medianNanos() < best) {
                winner = i;
                best = summary.medianNanos();
                winnerSummary = summary;
            }
        }
        D decision =
                Objects.requireNonNull(
                        backend.selectedDecision(batch, candidates.get(winner)),
                        "selected decision");
        return new MeasurementOutcome<>(
                decision, identities.get(winner), winnerSummary, List.copyOf(evidence));
    }

    static long checkedTotalExecutions(
            long candidateCount,
            long warmupCount,
            long timedSampleCount,
            long maximumTotalExecutions) {
        long perCandidate =
                Math.addExact(1L, Math.addExact(warmupCount, timedSampleCount));
        long total = Math.multiplyExact(candidateCount, perCandidate);
        if (total > maximumTotalExecutions) {
            throw new IllegalArgumentException("total plan executions exceed ceiling");
        }
        return total;
    }

    private static CompletePlanTuningResult.SampleSummary summary(List<Long> samples) {
        List<Long> sorted = new ArrayList<>(samples);
        sorted.sort(Long::compare);
        return new CompletePlanTuningResult.SampleSummary(
                sorted.getFirst(),
                sorted.get(sorted.size() / 2),
                sorted.getLast(),
                sorted.size());
    }

    private static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision>
            CompletePlanTuningResult<C, D> result(
                    CompletePlanTuningRequest<C, D> request,
                    CompletePlanTuningRequest.PlanCompatibility compatibility,
                    D decision,
                    CompletePlanTuningRequest.CandidateIdentity winner,
                    CompletePlanTuningResult.SampleSummary summary,
                    CompletePlanTuningResult.Source source,
                    List<CompletePlanTuningResult.CandidateEvidence> candidates) {
        BackendPartitionTuningHandoff<C, D> handoff =
                new BackendPartitionTuningHandoff<>(
                        request.handoff().partition(),
                        request.handoff().candidateBatch(),
                        Optional.of(decision));
        var record =
                new CompletePlanTuningResult.SelectedPlanRecord(
                        compatibility, winner, summary);
        var evidence =
                new CompletePlanTuningResult.Evidence(
                        request.modelFingerprint(),
                        request.profileFingerprint(),
                        request.targetFingerprint(),
                        request.objective(),
                        request.correctnessPolicy(),
                        request.policyIdentity(),
                        source,
                        candidates,
                        winner,
                        summary);
        return new CompletePlanTuningResult<>(handoff, record, evidence);
    }

    private static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision>
            ModelPlanCacheFile.Key key(
                    CompletePlanTuningRequest<C, D> request,
                    CompletePlanTuningRequest.PlanCompatibility compatibility) {
        return new ModelPlanCacheFile.Key(
                compatibility.producerIdentity().schemaVersion(),
                compatibility.producerIdentity().bytes(),
                compatibility.decisionCodecIdentity().schemaVersion(),
                compatibility.decisionCodecIdentity().bytes(),
                request.modelFingerprint().schemaVersion(),
                request.modelFingerprint().bytes(),
                request.profileFingerprint().schemaVersion(),
                request.profileFingerprint().bytes(),
                request.targetFingerprint().schemaVersion(),
                request.targetFingerprint().bytes(),
                compatibility.schemaVersion(),
                compatibility.bytes(),
                OBJECTIVE_MIN_MEDIAN,
                CORRECTNESS_EXACT_BYTES,
                request.policyIdentity().schemaVersion(),
                request.policyIdentity().bytes(),
                request.warmupCount(),
                request.timedSampleCount());
    }

    private static byte[] boundedEncoding(byte[] bytes) {
        Objects.requireNonNull(bytes, "encoded decision");
        if (bytes.length == 0 || bytes.length > ModelPlanCacheFile.MAX_OPAQUE_BYTES) {
            throw new IllegalArgumentException("encoded decision length is invalid");
        }
        return bytes.clone();
    }

    private record MeasurementOutcome<D>(
            D decision,
            CompletePlanTuningRequest.CandidateIdentity winnerIdentity,
            CompletePlanTuningResult.SampleSummary summary,
            List<CompletePlanTuningResult.CandidateEvidence> evidence) {}
}
