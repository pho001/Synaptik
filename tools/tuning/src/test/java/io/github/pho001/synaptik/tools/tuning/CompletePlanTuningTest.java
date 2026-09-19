package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CompletePlanTuningTest {
    @TempDir Path directory;

    @Test
    void sessionChecksAllCandidatesBeforeExactWarmupAndSamplesAndSelectsOnce()
            throws Exception {
        Backend backend = new Backend(scope(CompletePlanTuningRequest.ReuseScope.SESSION));
        List<String> events = new ArrayList<>();
        Correctness correctness = new Correctness(events);

        CompletePlanTuningResult<Batch, Decision> result =
                CompletePlanTuning.tune(
                        request(
                                directory.resolve("must-not-exist/cache.bin"),
                                2,
                                1,
                                3,
                                20),
                        backend,
                        correctness,
                        (batch, candidate) -> events.add("measure:" + candidate.id()),
                        clock(0, 9, 20, 25, 30, 37, 40, 45, 50, 55, 60, 65));

        assertAll(
                () -> assertFalse(Files.exists(directory.resolve("must-not-exist"))),
                () ->
                        assertEquals(
                                List.of(
                                        "capture:a",
                                        "compare:b",
                                        "measure:a",
                                        "measure:a",
                                        "measure:a",
                                        "measure:a",
                                        "measure:b",
                                        "measure:b",
                                        "measure:b",
                                        "measure:b"),
                                events),
                () -> assertEquals(List.of(1024L, 1024L), correctness.byteLimits),
                () ->
                        assertEquals(
                                new Decision("b"),
                                result.selectedHandoff().selectedDecision().orElseThrow()),
                () -> assertSame(backend.batchSeen, result.selectedHandoff().candidateBatch()),
                () ->
                        assertEquals(
                                List.of(9L, 5L, 7L),
                                result.evidence()
                                        .candidates()
                                        .getFirst()
                                        .elapsedSamplesNanos()),
                () ->
                        assertEquals(
                                new CompletePlanTuningResult.SampleSummary(5, 7, 9, 3),
                                result.evidence().candidates().getFirst().summary()),
                () ->
                        assertEquals(
                                CompletePlanTuningResult.CorrectnessAction.REFERENCE_CAPTURED,
                                result.evidence().candidates().getFirst().correctnessAction()),
                () ->
                        assertEquals(
                                CompletePlanTuningResult.CorrectnessAction.MATCH,
                                result.evidence().candidates().getLast().correctnessAction()),
                () -> assertEquals(1, backend.candidatesCalls),
                () -> assertEquals(1, backend.selectedCalls),
                () -> assertEquals(0, backend.encodeCalls),
                () -> assertEquals(0, backend.decodeCalls));

        assertThrows(
                UnsupportedOperationException.class,
                () -> result.evidence().candidates().add(result.evidence().candidates().getFirst()));
    }

    @Test
    void encounterOrderBreaksEqualMedianTie() throws Exception {
        Backend backend = new Backend(scope(CompletePlanTuningRequest.ReuseScope.SESSION));

        CompletePlanTuningResult<Batch, Decision> result =
                CompletePlanTuning.tune(
                        request(directory.resolve("unused"), 2, 0, 1, 4),
                        backend,
                        new Correctness(new ArrayList<>()),
                        (batch, candidate) -> {},
                        clock(0, 5, 10, 15));

        assertAll(
                () ->
                        assertEquals(
                                new Decision("a"),
                                result.selectedHandoff().selectedDecision().orElseThrow()),
                () -> assertEquals(1, backend.selectedCalls));
    }

    @Test
    void measurementFailurePropagatesWithoutConstructingDecision() {
        Backend backend = new Backend(scope(CompletePlanTuningRequest.ReuseScope.SESSION));
        IOException failure = new IOException("execution failed");

        IOException thrown =
                assertThrows(
                        IOException.class,
                        () ->
                                CompletePlanTuning.tune(
                                        request(directory.resolve("unused"), 2, 0, 1, 4),
                                        backend,
                                        new Correctness(new ArrayList<>()),
                                        (batch, candidate) -> {
                                            throw failure;
                                        },
                                        clock(0)));

        assertAll(
                () -> assertSame(failure, thrown),
                () -> assertEquals(0, backend.selectedCalls),
                () -> assertEquals(0, backend.encodeCalls));
    }

    static CompletePlanTuningRequest<Batch, Decision> request(
            Path path, int maximumCandidates, int warmups, int samples, long maximumExecutions) {
        return request(
                path,
                maximumCandidates,
                warmups,
                samples,
                maximumExecutions,
                1024,
                Optional.empty());
    }

    static CompletePlanTuningRequest<Batch, Decision> request(
            Path path,
            int maximumCandidates,
            int warmups,
            int samples,
            long maximumExecutions,
            long maximumCorrectnessBytes,
            Optional<Decision> selectedDecision) {
        return new CompletePlanTuningRequest<>(
                new CompletePlanTuningRequest.ModelFingerprint(1, bytes("model")),
                new CompletePlanTuningRequest.ProfileFingerprint(1, bytes("profile")),
                new CompletePlanTuningRequest.TargetFingerprint(1, bytes("target")),
                CompletePlanTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                CompletePlanTuningRequest.CorrectnessPolicy.EXACT_CANONICAL_BYTES,
                new CompletePlanTuningRequest.PolicyIdentity(1, bytes("policy")),
                maximumCandidates,
                warmups,
                samples,
                maximumExecutions,
                maximumCorrectnessBytes,
                path,
                new BackendPartitionTuningHandoff<>(
                        new PlannedPartition(new BackendId("cpu"), List.of(new NodeId(1))),
                        new Batch("batch"),
                        selectedDecision));
    }

    static CompletePlanTuningRequest.PlanCompatibility scope(
            CompletePlanTuningRequest.ReuseScope scope) {
        return new CompletePlanTuningRequest.PlanCompatibility(
                new CompletePlanTuningRequest.ProducerIdentity(1, bytes("producer")),
                new CompletePlanTuningRequest.DecisionCodecIdentity(1, bytes("codec")),
                1,
                bytes("compatibility"),
                scope);
    }

    static byte[] bytes(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }

    static CompletePlanTuning.NanoClock clock(long... values) {
        Deque<Long> readings = new ArrayDeque<>();
        for (long value : values) {
            readings.add(value);
        }
        return readings::removeFirst;
    }

    record Batch(String id) implements BackendTuningCandidateBatch {}

    record Decision(String id) implements BackendTuningDecision {}

    record Candidate(String id) {}

    static class Backend implements BackendCompletePlanTuning<Batch, Decision, Candidate> {
        final CompletePlanTuningRequest.PlanCompatibility compatibility;
        List<Candidate> suppliedCandidates = List.of(new Candidate("a"), new Candidate("b"));
        Batch batchSeen;
        int candidatesCalls;
        int selectedCalls;
        int encodeCalls;
        int decodeCalls;

        Backend(CompletePlanTuningRequest.PlanCompatibility compatibility) {
            this.compatibility = compatibility;
        }

        @Override
        public List<Candidate> candidates(Batch batch) {
            batchSeen = batch;
            candidatesCalls++;
            return suppliedCandidates;
        }

        @Override
        public CompletePlanTuningRequest.PlanCompatibility compatibility(Batch batch) {
            batchSeen = batch;
            return compatibility;
        }

        @Override
        public CompletePlanTuningRequest.CandidateIdentity candidateIdentity(
                Candidate candidate) {
            return new CompletePlanTuningRequest.CandidateIdentity(bytes(candidate.id()));
        }

        @Override
        public Decision selectedDecision(Batch batch, Candidate candidate) {
            selectedCalls++;
            return new Decision(candidate.id());
        }

        @Override
        public byte[] encodeDecision(Decision decision) {
            encodeCalls++;
            return bytes(decision.id());
        }

        @Override
        public Optional<Decision> decodeCompatibleDecision(Batch batch, byte[] encoded) {
            decodeCalls++;
            String id = new String(encoded, StandardCharsets.UTF_8);
            return suppliedCandidates.stream()
                    .filter(candidate -> candidate.id().equals(id))
                    .findFirst()
                    .map(candidate -> new Decision(candidate.id()));
        }
    }

    static class Correctness implements CompletePlanCorrectness<Batch, Candidate, String> {
        final List<String> events;
        final List<Long> byteLimits = new ArrayList<>();

        Correctness(List<String> events) {
            this.events = events;
        }

        @Override
        public String capture(Batch batch, Candidate candidate, long maximumBytes) {
            events.add("capture:" + candidate.id());
            byteLimits.add(maximumBytes);
            return "reference";
        }

        @Override
        public Outcome compare(
                String reference, Batch batch, Candidate candidate, long maximumBytes) {
            events.add("compare:" + candidate.id());
            byteLimits.add(maximumBytes);
            return Outcome.MATCH;
        }
    }
}
