package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Stateless entry point for one cache-first, bounded cold workload-tuning transaction.
 *
 * <p>The caller supplies stable evidence identities, opaque backend collaboration, and complete
 * candidate execution. This type validates and deduplicates work, reuses compatible persistent
 * decisions, measures misses, selects by integer median elapsed nanoseconds, and atomically
 * publishes compact cache state. It performs no model extraction, backend discovery, preparation,
 * Runtime execution orchestration, or graph/plan tuning.
 */
public final class WorkloadTuning {
    private static final int OBJECTIVE_MIN_MEDIAN_NANOS = 1;

    private WorkloadTuning() { }

    @FunctionalInterface
    interface NanoClock {
        long nanoTime();
    }

    /**
     * Reuses compatible persistent decisions and measures every complete candidate for cache misses.
     * Cache parsing and all complete-work validation finish before the first candidate execution.
     *
     * @param request non-null immutable validated request; the transaction does not mutate it
     * @param backend non-null typed backend-owned collaboration retained only for this call
     * @param measurement non-null one-complete-candidate cold execution action retained only for
     *     this call
     * @param <C> immutable complete candidate-batch type
     * @param <D> immutable selected-decision type
     * @param <K> immutable complete-candidate type
     * @return a non-null immutable result containing selected handoffs in original occurrence order
     *     and rich evidence separate from compact cache state
     * @throws NullPointerException if any argument or required collaborator result is {@code null}
     * @throws IllegalArgumentException if request-derived work, candidates, identities, elapsed
     *     samples, or codec output violates the documented bounds or invariants
     * @throws IllegalStateException if the monotonic clock moves backwards or a freshly encoded
     *     decision cannot be decoded to an equal compatible decision
     * @throws ArithmeticException if checked weight, invocation-count, or elapsed-time arithmetic
     *     overflows
     * @throws IOException if cache loading, validation, encoding, or atomic publication fails
     * @throws Exception if complete candidate execution reports another checked failure; every
     *     failure leaves prior cache state unpublished by this invocation
     */
    public static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> WorkloadTuningResult<C, D> tune(
                    WorkloadTuningRequest<C, D> request,
                    BackendWorkloadTuning<C, D, K> backend,
                    ColdCandidateMeasurement<C, K> measurement) throws Exception {
        return tune(request, backend, measurement, System::nanoTime);
    }

    static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> WorkloadTuningResult<C, D> tune(
                    WorkloadTuningRequest<C, D> request,
                    BackendWorkloadTuning<C, D, K> backend,
                    ColdCandidateMeasurement<C, K> measurement,
                    NanoClock clock) throws Exception {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(measurement, "measurement");
        Objects.requireNonNull(clock, "clock");

        List<Group<C, D, K>> groups = groupOccurrences(request, backend);
        WorkloadCacheFile cacheFile = new WorkloadCacheFile();
        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> loaded =
                cacheFile.load(request.workloadCache());

        int missCount = identifyHitsAndMisses(groups, request.budget(), backend, loaded);
        if (missCount > request.budget().maximumDistinctCacheMisses()) {
            throw new IllegalArgumentException("distinct cache misses exceed the request budget");
        }
        validateAllMissCandidates(groups, request.budget(), backend);
        validateCompleteInvocationCount(groups, request.budget());

        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> updated =
                new LinkedHashMap<>(loaded);
        boolean changed = false;
        for (Group<C, D, K> group : groups) {
            if (group.decision == null) {
                measure(group, request.budget(), backend, measurement, clock);
                if (group.compatibility.reuseScope()
                        == WorkloadTuningRequest.ReuseScope.PERSISTENT) {
                    byte[] encoded = boundedDecisionEncoding(backend.encodeDecision(group.decision));
                    Optional<D> decoded = Objects.requireNonNull(
                            backend.decodeCompatibleDecision(group.batch, encoded),
                            "decodeCompatibleDecision result");
                    if (decoded.isEmpty() || !group.decision.equals(decoded.orElseThrow())) {
                        throw new IllegalStateException(
                                "persistent decision did not round trip compatibly");
                    }
                    WorkloadCacheFile.Entry entry = new WorkloadCacheFile.Entry(
                            group.cacheKey,
                            encoded,
                            group.winnerIdentity.bytes(),
                            group.winnerSummary);
                    updated.put(group.cacheKey, entry);
                    changed = true;
                }
            }
        }
        if (changed) {
            cacheFile.publish(request.workloadCache(), updated);
        }
        return result(request, groups);
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> List<Group<C, D, K>> groupOccurrences(
                    WorkloadTuningRequest<C, D> request,
                    BackendWorkloadTuning<C, D, K> backend) {
        Map<GroupKey, Group<C, D, K>> grouped = new LinkedHashMap<>();
        List<WorkloadTuningRequest.Occurrence<C, D>> occurrences = request.occurrences();
        for (int index = 0; index < occurrences.size(); index++) {
            WorkloadTuningRequest.Occurrence<C, D> occurrence = occurrences.get(index);
            C batch = occurrence.handoff().candidateBatch();
            WorkloadTuningRequest.WorkloadCompatibility compatibility =
                    Objects.requireNonNull(backend.compatibility(batch), "compatibility result");
            GroupKey groupKey = new GroupKey(
                    compatibility,
                    request.objective(),
                    request.budget().warmupCount(),
                    request.budget().timedSampleCount());
            Group<C, D, K> group = grouped.get(groupKey);
            if (group == null) {
                group = new Group<>(compatibility, batch, cacheKey(compatibility, request.budget()));
                grouped.put(groupKey, group);
            }
            group.totalWeight = Math.addExact(group.totalWeight, occurrence.weight());
            group.occurrenceIndexes.add(index);
            group.occurrenceEvidence.add(new WorkloadTuningResult.OccurrenceEvidence(
                    occurrence.contextFingerprint(), occurrence.weight()));
        }
        return List.copyOf(grouped.values());
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> int identifyHitsAndMisses(
                    List<Group<C, D, K>> groups,
                    WorkloadTuningRequest.Budget budget,
                    BackendWorkloadTuning<C, D, K> backend,
                    Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> loaded) {
        int misses = 0;
        for (Group<C, D, K> group : groups) {
            WorkloadCacheFile.Entry entry = group.compatibility.reuseScope()
                    == WorkloadTuningRequest.ReuseScope.PERSISTENT
                    ? loaded.get(group.cacheKey) : null;
            if (entry == null) {
                misses = Math.incrementExact(misses);
                continue;
            }
            Optional<D> decision = Objects.requireNonNull(
                    backend.decodeCompatibleDecision(group.batch, entry.encodedDecision()),
                    "decodeCompatibleDecision result");
            if (decision.isEmpty()) {
                misses = Math.incrementExact(misses);
                continue;
            }
            group.decision = Objects.requireNonNull(decision.orElseThrow(), "decoded decision");
            group.source = WorkloadTuningResult.Source.CACHE_HIT;
            group.winnerIdentity = new WorkloadTuningRequest.CandidateIdentity(entry.winnerIdentity());
            group.winnerSummary = entry.summary();
            group.candidateEvidence = List.of();
        }
        return misses;
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> void validateAllMissCandidates(
                    List<Group<C, D, K>> groups,
                    WorkloadTuningRequest.Budget budget,
                    BackendWorkloadTuning<C, D, K> backend) {
        for (Group<C, D, K> group : groups) {
            if (group.decision != null) continue;
            List<K> supplied = Objects.requireNonNull(
                    backend.candidates(group.batch), "candidates result");
            List<K> candidates = List.copyOf(supplied);
            if (candidates.isEmpty()) {
                throw new IllegalArgumentException("candidate list must not be empty");
            }
            if (candidates.size() > budget.maximumCandidatesPerMiss()) {
                throw new IllegalArgumentException("candidate count exceeds the request budget");
            }
            Set<K> uniqueCandidates = new HashSet<>();
            Set<WorkloadTuningRequest.CandidateIdentity> uniqueIdentities = new HashSet<>();
            List<WorkloadTuningRequest.CandidateIdentity> identities =
                    new ArrayList<>(candidates.size());
            for (K candidate : candidates) {
                if (!uniqueCandidates.add(candidate)) {
                    throw new IllegalArgumentException("candidate list contains a duplicate");
                }
                WorkloadTuningRequest.CandidateIdentity first = Objects.requireNonNull(
                        backend.candidateIdentity(candidate), "candidateIdentity result");
                WorkloadTuningRequest.CandidateIdentity second = Objects.requireNonNull(
                        backend.candidateIdentity(candidate), "candidateIdentity stability result");
                if (!first.equals(second)) {
                    throw new IllegalArgumentException("candidate identity is unstable");
                }
                if (first.bytes().length > WorkloadCacheFile.MAX_OPAQUE_BYTES) {
                    throw new IllegalArgumentException("candidate identity exceeds its bound");
                }
                if (!uniqueIdentities.add(first)) {
                    throw new IllegalArgumentException("candidate identity is duplicated");
                }
                identities.add(first);
            }
            group.candidates = candidates;
            group.identities = List.copyOf(identities);
        }
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> void validateCompleteInvocationCount(
                    List<Group<C, D, K>> groups, WorkloadTuningRequest.Budget budget) {
        long invocationsPerCandidate = Math.addExact(
                (long) budget.warmupCount(), budget.timedSampleCount());
        long total = 0;
        for (Group<C, D, K> group : groups) {
            if (group.decision == null) {
                total = Math.addExact(total, Math.multiplyExact(
                        group.candidates.size(), invocationsPerCandidate));
            }
        }
        if (total < 0) {
            throw new ArithmeticException("candidate invocation count overflow");
        }
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> void measure(
                    Group<C, D, K> group,
                    WorkloadTuningRequest.Budget budget,
                    BackendWorkloadTuning<C, D, K> backend,
                    ColdCandidateMeasurement<C, K> measurement,
                    NanoClock clock) throws Exception {
        List<WorkloadTuningResult.CandidateEvidence> evidence =
                new ArrayList<>(group.candidates.size());
        long bestMedian = Long.MAX_VALUE;
        int winnerIndex = -1;
        for (int candidateIndex = 0; candidateIndex < group.candidates.size(); candidateIndex++) {
            K candidate = group.candidates.get(candidateIndex);
            for (int warmup = 0; warmup < budget.warmupCount(); warmup++) {
                measurement.execute(group.batch, candidate);
            }
            long[] samples = new long[budget.timedSampleCount()];
            for (int sample = 0; sample < samples.length; sample++) {
                long start = clock.nanoTime();
                measurement.execute(group.batch, candidate);
                long end = clock.nanoTime();
                if (end < start) {
                    throw new IllegalStateException("monotonic clock moved backwards");
                }
                samples[sample] = Math.subtractExact(end, start);
            }
            WorkloadTuningResult.SampleSummary summary = summarize(samples);
            List<Long> raw = Arrays.stream(samples).boxed().toList();
            evidence.add(new WorkloadTuningResult.CandidateEvidence(
                    group.identities.get(candidateIndex), raw, summary));
            if (winnerIndex < 0 || summary.medianNanos() < bestMedian) {
                winnerIndex = candidateIndex;
                bestMedian = summary.medianNanos();
            }
        }
        K winner = group.candidates.get(winnerIndex);
        group.decision = Objects.requireNonNull(
                backend.selectedDecision(group.batch, winner), "selectedDecision result");
        group.source = WorkloadTuningResult.Source.MEASURED;
        group.candidateEvidence = List.copyOf(evidence);
        group.winnerIdentity = group.identities.get(winnerIndex);
        group.winnerSummary = evidence.get(winnerIndex).summary();
    }

    private static WorkloadTuningResult.SampleSummary summarize(long[] samples) {
        long minimum = Long.MAX_VALUE;
        long maximum = Long.MIN_VALUE;
        for (long sample : samples) {
            if (sample < 0) {
                throw new IllegalArgumentException("elapsed sample must be non-negative");
            }
            minimum = Math.min(minimum, sample);
            maximum = Math.max(maximum, sample);
        }
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return new WorkloadTuningResult.SampleSummary(
                minimum, sorted[sorted.length / 2], maximum, samples.length);
    }

    private static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> WorkloadTuningResult<C, D> result(
                    WorkloadTuningRequest<C, D> request,
                    List<Group<C, D, K>> groups) {
        List<D> decisions = new ArrayList<>(request.occurrences().size());
        for (int index = 0; index < request.occurrences().size(); index++) decisions.add(null);
        List<WorkloadTuningResult.WorkloadEvidence> workloadEvidence =
                new ArrayList<>(groups.size());
        for (Group<C, D, K> group : groups) {
            for (int index : group.occurrenceIndexes) decisions.set(index, group.decision);
            workloadEvidence.add(new WorkloadTuningResult.WorkloadEvidence(
                    group.compatibility,
                    group.totalWeight,
                    group.occurrenceEvidence,
                    group.source,
                    group.candidateEvidence,
                    group.winnerIdentity,
                    group.winnerSummary));
        }
        List<BackendPartitionTuningHandoff<C, D>> handoffs =
                new ArrayList<>(request.occurrences().size());
        for (int index = 0; index < request.occurrences().size(); index++) {
            BackendPartitionTuningHandoff<C, D> input =
                    request.occurrences().get(index).handoff();
            handoffs.add(new BackendPartitionTuningHandoff<>(
                    input.partition(), input.candidateBatch(), Optional.of(decisions.get(index))));
        }
        return new WorkloadTuningResult<>(handoffs, new WorkloadTuningResult.Evidence(
                request.modelFingerprint(),
                request.profileFingerprint(),
                request.objective(),
                request.budget(),
                workloadEvidence));
    }

    private static WorkloadCacheFile.Key cacheKey(
            WorkloadTuningRequest.WorkloadCompatibility compatibility,
            WorkloadTuningRequest.Budget budget) {
        return new WorkloadCacheFile.Key(
                compatibility.schemaVersion(),
                compatibility.bytes(),
                OBJECTIVE_MIN_MEDIAN_NANOS,
                budget.warmupCount(),
                budget.timedSampleCount());
    }

    private static byte[] boundedDecisionEncoding(byte[] encoded) throws IOException {
        Objects.requireNonNull(encoded, "encodeDecision result");
        if (encoded.length == 0 || encoded.length > WorkloadCacheFile.MAX_OPAQUE_BYTES) {
            throw new IOException("encoded decision length is invalid");
        }
        return encoded.clone();
    }

    private record GroupKey(
            WorkloadTuningRequest.WorkloadCompatibility compatibility,
            WorkloadTuningRequest.Objective objective,
            int warmupCount,
            int timedSampleCount) { }

    private static final class Group<
            C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision,
            K> {
        final WorkloadTuningRequest.WorkloadCompatibility compatibility;
        final C batch;
        final WorkloadCacheFile.Key cacheKey;
        final List<Integer> occurrenceIndexes = new ArrayList<>();
        final List<WorkloadTuningResult.OccurrenceEvidence> occurrenceEvidence = new ArrayList<>();
        long totalWeight;
        D decision;
        List<K> candidates;
        List<WorkloadTuningRequest.CandidateIdentity> identities;
        WorkloadTuningResult.Source source;
        List<WorkloadTuningResult.CandidateEvidence> candidateEvidence;
        WorkloadTuningRequest.CandidateIdentity winnerIdentity;
        WorkloadTuningResult.SampleSummary winnerSummary;

        Group(
                WorkloadTuningRequest.WorkloadCompatibility compatibility,
                C batch,
                WorkloadCacheFile.Key cacheKey) {
            this.compatibility = compatibility;
            this.batch = batch;
            this.cacheKey = cacheKey;
        }
    }
}
