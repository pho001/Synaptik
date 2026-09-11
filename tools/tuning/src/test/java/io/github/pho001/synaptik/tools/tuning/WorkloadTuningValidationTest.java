package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WorkloadTuningValidationTest {
    @TempDir Path directory;

    @Test
    void requestSnapshotsBytesAndRejectsInvalidInputsAndPresentDecision() {
        byte[] modelBytes = WorkloadTuningTest.bytes("model");
        byte[] contextBytes = WorkloadTuningTest.bytes("context");
        var batch = new WorkloadTuningTest.Batch("alpha");
        var occurrence = new WorkloadTuningRequest.Occurrence<>(
                new io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff<>(
                        WorkloadTuningTest.partition(), batch, Optional.empty()),
                1,
                contextBytes);
        var model = new WorkloadTuningRequest.ModelFingerprint(1, modelBytes);
        modelBytes[0] = 0;
        contextBytes[0] = 0;

        assertAll(
                () -> assertArrayEquals(WorkloadTuningTest.bytes("model"), model.bytes()),
                () -> assertArrayEquals(
                        WorkloadTuningTest.bytes("context"), occurrence.contextFingerprint()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.ModelFingerprint(0, new byte[] {1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.ProfileFingerprint(1, new byte[0])),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.CandidateIdentity(new byte[0])),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.Budget(0, 1, 0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.Budget(1, 0, 0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.Budget(1, 1, -1, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.Budget(1, 1, 0, 2)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningRequest.Occurrence<>(
                                new io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff<>(
                                        WorkloadTuningTest.partition(), batch,
                                        Optional.of(new WorkloadTuningTest.Decision("chosen"))),
                                1,
                                new byte[] {1})));
    }

    @Test
    void corruptCacheStopsBeforeCandidateEnumerationExecutionOrMutation() throws Exception {
        Path cache = directory.resolve("corrupt.bin");
        byte[] corrupt = new byte[64];
        java.util.Arrays.fill(corrupt, (byte) 7);
        Files.write(cache, corrupt);
        byte[] prior = Files.readAllBytes(cache);
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("only"))));
        var measurement = new WorkloadTuningTest.CountingMeasurement();

        assertThrows(IOException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        cache,
                        new WorkloadTuningTest.Batch("alpha"),
                        WorkloadTuningTest.scopePersistent(),
                        0,
                        0,
                        1),
                backend,
                measurement,
                WorkloadTuningTest.clock(0, 1)));

        assertAll(
                () -> assertEquals(0, backend.candidatesCalls),
                () -> assertEquals(0, backend.decodeCalls),
                () -> assertEquals(0, measurement.executions),
                () -> assertArrayEquals(prior, Files.readAllBytes(cache)));
    }

    @Test
    void validatesEveryMissCandidateListAndIdentityBeforeFirstExecution() {
        Path cache = directory.resolve("validation.bin");
        var first = new WorkloadTuningTest.Batch("first");
        var second = new WorkloadTuningTest.Batch("second");
        WorkloadTuningTest.Backend.compatibilities.put("first",
                new WorkloadTuningRequest.WorkloadCompatibility(
                        1, new byte[] {1}, WorkloadTuningRequest.ReuseScope.SESSION));
        WorkloadTuningTest.Backend.compatibilities.put("second",
                new WorkloadTuningRequest.WorkloadCompatibility(
                        1, new byte[] {2}, WorkloadTuningRequest.ReuseScope.SESSION));
        var request = new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(1, new byte[] {1}),
                new WorkloadTuningRequest.ProfileFingerprint(1, new byte[] {2}),
                List.of(
                        WorkloadTuningTest.occurrence(first, 1, "first"),
                        WorkloadTuningTest.occurrence(second, 1, "second")),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(2, 3, 0, 1),
                cache);
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "first", List.of(new WorkloadTuningTest.Candidate("valid")),
                "second", List.of(
                        new WorkloadTuningTest.Candidate("duplicate"),
                        new WorkloadTuningTest.Candidate("duplicate"))));
        var measurement = new WorkloadTuningTest.CountingMeasurement();

        assertThrows(IllegalArgumentException.class, () -> WorkloadTuning.tune(
                request, backend, measurement, WorkloadTuningTest.clock(0, 1)));
        assertEquals(0, measurement.executions);
    }

    @Test
    void missAndCandidateBudgetsFailBeforeEnumerationOrExecutionAsApplicable() {
        var first = new WorkloadTuningTest.Batch("budget-first");
        var second = new WorkloadTuningTest.Batch("budget-second");
        WorkloadTuningTest.Backend.compatibilities.put("budget-first",
                new WorkloadTuningRequest.WorkloadCompatibility(
                        1, new byte[] {11}, WorkloadTuningRequest.ReuseScope.SESSION));
        WorkloadTuningTest.Backend.compatibilities.put("budget-second",
                new WorkloadTuningRequest.WorkloadCompatibility(
                        1, new byte[] {12}, WorkloadTuningRequest.ReuseScope.SESSION));
        var occurrences = List.of(
                WorkloadTuningTest.occurrence(first, 1, "first"),
                WorkloadTuningTest.occurrence(second, 1, "second"));
        var missLimited = baseRequest(
                occurrences, new WorkloadTuningRequest.Budget(1, 3, 0, 1), "miss-limit.bin");
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "budget-first", List.of(new WorkloadTuningTest.Candidate("one")),
                "budget-second", List.of(new WorkloadTuningTest.Candidate("two"))));
        var measurement = new WorkloadTuningTest.CountingMeasurement();
        assertThrows(IllegalArgumentException.class, () -> WorkloadTuning.tune(
                missLimited, backend, measurement, WorkloadTuningTest.clock(0, 1, 2, 3)));
        assertEquals(0, backend.candidatesCalls);

        var candidateLimited = baseRequest(
                List.of(WorkloadTuningTest.occurrence(first, 1, "first")),
                new WorkloadTuningRequest.Budget(1, 1, 0, 1),
                "candidate-limit.bin");
        var tooMany = new WorkloadTuningTest.Backend(Map.of(
                "budget-first", List.of(
                        new WorkloadTuningTest.Candidate("one"),
                        new WorkloadTuningTest.Candidate("two"))));
        assertThrows(IllegalArgumentException.class, () -> WorkloadTuning.tune(
                candidateLimited, tooMany, measurement, WorkloadTuningTest.clock(0, 1, 2, 3)));
        assertEquals(0, measurement.executions);
    }

    @Test
    void nullEmptyAndNullContainingCandidateListsFailBeforeExecution() {
        var batch = new WorkloadTuningTest.Batch("malformed");
        var request = WorkloadTuningTest.request(
                directory.resolve("malformed.bin"), batch,
                WorkloadTuningTest.scopeSession(), 0, 0, 1);
        var measurement = new WorkloadTuningTest.CountingMeasurement();

        for (List<WorkloadTuningTest.Candidate> candidates : List.of(
                List.<WorkloadTuningTest.Candidate>of(),
                java.util.Arrays.asList((WorkloadTuningTest.Candidate) null))) {
            var backend = new WorkloadTuningTest.Backend(Map.of("malformed", candidates));
            assertThrows(RuntimeException.class, () -> WorkloadTuning.tune(
                    request, backend, measurement, WorkloadTuningTest.clock(0, 1)));
        }
        var nullList = new WorkloadTuningTest.Backend(Map.of()) {
            @Override
            public List<WorkloadTuningTest.Candidate> candidates(
                    WorkloadTuningTest.Batch ignored) {
                candidatesCalls++;
                return null;
            }
        };
        assertThrows(NullPointerException.class, () -> WorkloadTuning.tune(
                request, nullList, measurement, WorkloadTuningTest.clock(0, 1)));
        assertEquals(0, measurement.executions);
    }

    @Test
    void weightAndElapsedArithmeticOverflowBeforePublication() {
        var batch = new WorkloadTuningTest.Batch("overflow");
        WorkloadTuningTest.Backend.compatibilities.put("overflow",
                WorkloadTuningTest.scopeSession());
        var overflowWeights = baseRequest(
                List.of(
                        WorkloadTuningTest.occurrence(batch, Long.MAX_VALUE, "one"),
                        WorkloadTuningTest.occurrence(batch, 1, "two")),
                new WorkloadTuningRequest.Budget(1, 1, 0, 1),
                "weight-overflow.bin");
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "overflow", List.of(new WorkloadTuningTest.Candidate("only"))));
        var measurement = new WorkloadTuningTest.CountingMeasurement();
        assertThrows(ArithmeticException.class, () -> WorkloadTuning.tune(
                overflowWeights, backend, measurement, WorkloadTuningTest.clock(0, 1)));
        assertEquals(0, backend.candidatesCalls);

        Path elapsedCache = directory.resolve("elapsed-overflow.bin");
        assertThrows(ArithmeticException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        elapsedCache, batch, WorkloadTuningTest.scopePersistent(), 0, 0, 1),
                backend,
                measurement,
                WorkloadTuningTest.clock(Long.MIN_VALUE, Long.MAX_VALUE)));
        assertFalse(Files.exists(elapsedCache));
    }

    @Test
    void rejectsUnstableAndDuplicateIdentitiesBeforeExecution() {
        var batch = new WorkloadTuningTest.Batch("alpha");
        var measurement = new WorkloadTuningTest.CountingMeasurement();
        var unstable = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("only")))) {
            private int sequence;

            @Override
            public WorkloadTuningRequest.CandidateIdentity candidateIdentity(
                    WorkloadTuningTest.Candidate candidate) {
                return new WorkloadTuningRequest.CandidateIdentity(new byte[] {(byte) sequence++});
            }
        };
        assertThrows(IllegalArgumentException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        directory.resolve("unstable.bin"), batch,
                        WorkloadTuningTest.scopeSession(), 0, 0, 1),
                unstable,
                measurement,
                WorkloadTuningTest.clock(0, 1)));

        var duplicate = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(
                        new WorkloadTuningTest.Candidate("one"),
                        new WorkloadTuningTest.Candidate("two")))) {
            @Override
            public WorkloadTuningRequest.CandidateIdentity candidateIdentity(
                    WorkloadTuningTest.Candidate candidate) {
                return new WorkloadTuningRequest.CandidateIdentity(new byte[] {1});
            }
        };
        assertThrows(IllegalArgumentException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        directory.resolve("duplicate.bin"), batch,
                        WorkloadTuningTest.scopeSession(), 0, 0, 1),
                duplicate,
                measurement,
                WorkloadTuningTest.clock(0, 1, 2, 3)));
        assertEquals(0, measurement.executions);
    }

    @Test
    void rejectsBackwardClockAndIncompatibleCodecWithoutPublishing() {
        var batch = new WorkloadTuningTest.Batch("alpha");
        Path backwardCache = directory.resolve("backward.bin");
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("only"))));
        assertThrows(IllegalStateException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        backwardCache, batch, WorkloadTuningTest.scopePersistent(), 0, 0, 1),
                backend,
                new WorkloadTuningTest.CountingMeasurement(),
                WorkloadTuningTest.clock(9, 4)));
        assertFalse(Files.exists(backwardCache));

        Path codecCache = directory.resolve("codec.bin");
        var incompatibleCodec = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("only")))) {
            @Override
            public Optional<WorkloadTuningTest.Decision> decodeCompatibleDecision(
                    WorkloadTuningTest.Batch ignored, byte[] encodedDecision) {
                decodeCalls++;
                return Optional.empty();
            }
        };
        assertThrows(IllegalStateException.class, () -> WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        codecCache, batch, WorkloadTuningTest.scopePersistent(), 0, 0, 1),
                incompatibleCodec,
                new WorkloadTuningTest.CountingMeasurement(),
                WorkloadTuningTest.clock(0, 1)));
        assertFalse(Files.exists(codecCache));
    }

    @Test
    void requestAndResultCollectionsAndArraysDoNotLeakMutation() throws Exception {
        var batch = new WorkloadTuningTest.Batch("alpha");
        List<WorkloadTuningRequest.Occurrence<WorkloadTuningTest.Batch, WorkloadTuningTest.Decision>>
                mutable = new ArrayList<>();
        mutable.add(WorkloadTuningTest.occurrence(batch, 1, "context"));
        var request = new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(1, new byte[] {1}),
                new WorkloadTuningRequest.ProfileFingerprint(1, new byte[] {2}),
                mutable,
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(1, 1, 0, 1),
                directory.resolve("immutable.bin"));
        mutable.clear();
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("only"))));
        var result = WorkloadTuning.tune(
                request, backend, new WorkloadTuningTest.CountingMeasurement(),
                WorkloadTuningTest.clock(0, 1));
        byte[] exposed = request.modelFingerprint().bytes();
        exposed[0] = 99;

        assertAll(
                () -> assertEquals(1, request.occurrences().size()),
                () -> assertArrayEquals(new byte[] {1}, request.modelFingerprint().bytes()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.selectedHandoffs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.evidence().workloads().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.evidence().workloads().getFirst().candidates()
                                .getFirst().elapsedSamplesNanos().clear()));
    }

    @Test
    void candidateEvidenceRequiresRawSamplesToExactlyMatchItsSummary() {
        var identity = new WorkloadTuningRequest.CandidateIdentity(new byte[] {1});

        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningResult.CandidateEvidence(
                                identity,
                                List.of(),
                                new WorkloadTuningResult.SampleSummary(1, 1, 1, 1))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningResult.CandidateEvidence(
                                identity,
                                List.of(1L, 3L, 2L),
                                new WorkloadTuningResult.SampleSummary(1, 2, 3, 1))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningResult.CandidateEvidence(
                                identity,
                                List.of(1L, 3L, 2L),
                                new WorkloadTuningResult.SampleSummary(0, 2, 3, 3))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningResult.CandidateEvidence(
                                identity,
                                List.of(1L, 3L, 2L),
                                new WorkloadTuningResult.SampleSummary(1, 1, 3, 3))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new WorkloadTuningResult.CandidateEvidence(
                                identity,
                                List.of(1L, 3L, 2L),
                                new WorkloadTuningResult.SampleSummary(1, 2, 4, 3))));

        List<Long> unsorted = new ArrayList<>(List.of(5L, 1L, 3L));
        var valid = new WorkloadTuningResult.CandidateEvidence(
                identity,
                unsorted,
                new WorkloadTuningResult.SampleSummary(1, 3, 5, 3));
        unsorted.set(0, 99L);

        assertAll(
                () -> assertEquals(List.of(5L, 1L, 3L), valid.elapsedSamplesNanos()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> valid.elapsedSamplesNanos().clear()));
    }

    private WorkloadTuningRequest<WorkloadTuningTest.Batch, WorkloadTuningTest.Decision> baseRequest(
            List<WorkloadTuningRequest.Occurrence<
                    WorkloadTuningTest.Batch, WorkloadTuningTest.Decision>> occurrences,
            WorkloadTuningRequest.Budget budget,
            String cacheName) {
        return new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(1, new byte[] {1}),
                new WorkloadTuningRequest.ProfileFingerprint(1, new byte[] {2}),
                occurrences,
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                budget,
                directory.resolve(cacheName));
    }
}
