package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WorkloadTuningTest {
    @TempDir Path directory;

    @Test
    void measuresCompleteCandidatesSelectsLowestMedianAndRetainsRichEvidence() throws Exception {
        Batch batch = new Batch("alpha");
        Backend backend = new Backend(Map.of("alpha", List.of(new Candidate("first"), new Candidate("second"))));
        CountingMeasurement measurement = new CountingMeasurement();
        WorkloadTuning.NanoClock clock = clock(
                0, 9, 20, 25, 30, 37,
                40, 45, 50, 55, 60, 65);

        WorkloadTuningResult<Batch, Decision> result = WorkloadTuning.tune(
                request(directory.resolve("cache.bin"), batch, scopeSession(), 1, 1, 3),
                backend,
                measurement,
                clock);

        var workload = result.evidence().workloads().getFirst();
        assertAll(
                () -> assertEquals(8, measurement.executions),
                () -> assertEquals(1, backend.candidatesCalls),
                () -> assertEquals(1, backend.selectedDecisionCalls),
                () -> assertEquals(new Decision("second"),
                        result.selectedHandoffs().getFirst().selectedDecision().orElseThrow()),
                () -> assertSame(batch, result.selectedHandoffs().getFirst().candidateBatch()),
                () -> assertEquals(WorkloadTuningResult.Source.MEASURED, workload.source()),
                () -> assertEquals(List.of(9L, 5L, 7L),
                        workload.candidates().getFirst().elapsedSamplesNanos()),
                () -> assertEquals(new WorkloadTuningResult.SampleSummary(5, 7, 9, 3),
                        workload.candidates().getFirst().summary()),
                () -> assertEquals(List.of(5L, 5L, 5L),
                        workload.candidates().get(1).elapsedSamplesNanos()),
                () -> assertEquals(new WorkloadTuningResult.SampleSummary(5, 5, 5, 3),
                        workload.winnerSummary()));
    }

    @Test
    void deduplicatesExactOccurrencesSumsWeightAndBreaksTiesByEncounterOrder() throws Exception {
        Batch firstBatch = new Batch("same");
        Batch secondBatch = new Batch("same");
        Backend backend = new Backend(Map.of("same", List.of(new Candidate("first"), new Candidate("second"))));
        WorkloadTuningRequest<Batch, Decision> request = new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(1, bytes("model")),
                new WorkloadTuningRequest.ProfileFingerprint(1, bytes("profile")),
                List.of(
                        occurrence(firstBatch, 2, "left"),
                        occurrence(secondBatch, 3, "right")),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(1, 2, 0, 1),
                directory.resolve("session.bin"));

        WorkloadTuningResult<Batch, Decision> result = WorkloadTuning.tune(
                request, backend, new CountingMeasurement(), clock(0, 5, 10, 15));

        Decision first = result.selectedHandoffs().getFirst().selectedDecision().orElseThrow();
        Decision second = result.selectedHandoffs().get(1).selectedDecision().orElseThrow();
        assertAll(
                () -> assertSame(first, second),
                () -> assertEquals(new Decision("first"), first),
                () -> assertEquals(1, backend.candidatesCalls),
                () -> assertEquals(1, backend.selectedDecisionCalls),
                () -> assertEquals(0, backend.encodeCalls),
                () -> assertEquals(false, java.nio.file.Files.exists(directory.resolve("session.bin"))),
                () -> assertEquals(5, result.evidence().workloads().getFirst().totalWeight()),
                () -> assertEquals(2, result.evidence().workloads().getFirst().occurrences().size()));
    }

    @Test
    void propagatesMeasurementFailureWithoutPublishing() {
        Path cache = directory.resolve("failed.bin");
        Backend backend = new Backend(Map.of("alpha", List.of(new Candidate("only"))));
        IOException failure = new IOException("execution failed");

        IOException thrown = assertThrows(IOException.class, () -> WorkloadTuning.tune(
                request(cache, new Batch("alpha"), scopePersistent(), 0, 0, 1),
                backend,
                (batch, candidate) -> { throw failure; },
                clock(0, 1)));

        assertAll(
                () -> assertSame(failure, thrown),
                () -> assertEquals(false, java.nio.file.Files.exists(cache)));
    }

    static WorkloadTuningRequest<Batch, Decision> request(
            Path cache,
            Batch batch,
            WorkloadTuningRequest.WorkloadCompatibility compatibility,
            int ignored,
            int warmups,
            int samples) {
        Backend.compatibilities.put(batch.id(), compatibility);
        return new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(1, bytes("model")),
                new WorkloadTuningRequest.ProfileFingerprint(1, bytes("profile")),
                List.of(occurrence(batch, 1, "context")),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(4, 8, warmups, samples),
                cache);
    }

    static WorkloadTuningRequest.Occurrence<Batch, Decision> occurrence(
            Batch batch, long weight, String context) {
        return new WorkloadTuningRequest.Occurrence<>(
                new BackendPartitionTuningHandoff<>(
                        partition(), batch, Optional.empty()),
                weight,
                bytes(context));
    }

    static WorkloadTuningRequest.WorkloadCompatibility scopeSession() {
        return new WorkloadTuningRequest.WorkloadCompatibility(
                1, bytes("same-key"), WorkloadTuningRequest.ReuseScope.SESSION);
    }

    static WorkloadTuningRequest.WorkloadCompatibility scopePersistent() {
        return new WorkloadTuningRequest.WorkloadCompatibility(
                1, bytes("same-key"), WorkloadTuningRequest.ReuseScope.PERSISTENT);
    }

    static PlannedPartition partition() {
        return new PlannedPartition(new BackendId("cpu"), List.of(new NodeId(7)));
    }

    static byte[] bytes(String value) {
        return value.getBytes(StandardCharsets.UTF_8);
    }

    static WorkloadTuning.NanoClock clock(long... readings) {
        Deque<Long> values = new ArrayDeque<>();
        for (long reading : readings) values.add(reading);
        return () -> values.removeFirst();
    }

    record Batch(String id) implements BackendTuningCandidateBatch { }
    record Decision(String candidate) implements BackendTuningDecision { }
    record Candidate(String id) { }

    static class Backend implements BackendWorkloadTuning<Batch, Decision, Candidate> {
        static final Map<String, WorkloadTuningRequest.WorkloadCompatibility> compatibilities =
                new HashMap<>();
        final Map<String, List<Candidate>> candidates;
        int candidatesCalls;
        int identityCalls;
        int selectedDecisionCalls;
        int encodeCalls;
        int decodeCalls;

        Backend(Map<String, List<Candidate>> candidates) {
            this.candidates = candidates;
        }

        @Override
        public List<Candidate> candidates(Batch batch) {
            candidatesCalls++;
            return candidates.get(batch.id());
        }

        @Override
        public WorkloadTuningRequest.WorkloadCompatibility compatibility(Batch batch) {
            return compatibilities.getOrDefault(batch.id(), scopeSession());
        }

        @Override
        public WorkloadTuningRequest.CandidateIdentity candidateIdentity(Candidate candidate) {
            identityCalls++;
            return new WorkloadTuningRequest.CandidateIdentity(bytes(candidate.id()));
        }

        @Override
        public Decision selectedDecision(Batch batch, Candidate candidate) {
            selectedDecisionCalls++;
            return new Decision(candidate.id());
        }

        @Override
        public byte[] encodeDecision(Decision decision) {
            encodeCalls++;
            return bytes(decision.candidate());
        }

        @Override
        public Optional<Decision> decodeCompatibleDecision(Batch batch, byte[] encodedDecision) {
            decodeCalls++;
            String value = new String(encodedDecision, StandardCharsets.UTF_8);
            return candidates.getOrDefault(batch.id(), List.of()).stream()
                    .filter(candidate -> candidate.id().equals(value))
                    .findFirst()
                    .map(candidate -> new Decision(candidate.id()));
        }
    }

    static final class CountingMeasurement
            implements ColdCandidateMeasurement<Batch, Candidate> {
        int executions;

        @Override
        public void execute(Batch candidateBatch, Candidate candidate) {
            executions++;
        }
    }
}
