package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CompletePlanTuningValidationTest {
    @TempDir Path directory;

    @Test
    void requestSnapshotsEveryOpaqueIdentityRetainsHandoffAndPerformsNoIo() {
        byte[] model = CompletePlanTuningTest.bytes("model");
        byte[] profile = CompletePlanTuningTest.bytes("profile");
        byte[] target = CompletePlanTuningTest.bytes("target");
        byte[] policy = CompletePlanTuningTest.bytes("policy");
        CompletePlanTuningTest.Batch batch = new CompletePlanTuningTest.Batch("batch");
        BackendPartitionTuningHandoff<
                        CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                handoff =
                        new BackendPartitionTuningHandoff<>(
                                CompletePlanTuningTest.request(
                                                directory.resolve("seed"), 2, 0, 1, 4)
                                        .handoff()
                                        .partition(),
                                batch,
                                Optional.empty());
        Path cache = directory.resolve("absent-parent/cache.bin");

        CompletePlanTuningRequest<
                        CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                request =
                        new CompletePlanTuningRequest<>(
                                new CompletePlanTuningRequest.ModelFingerprint(1, model),
                                new CompletePlanTuningRequest.ProfileFingerprint(2, profile),
                                new CompletePlanTuningRequest.TargetFingerprint(3, target),
                                CompletePlanTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                                CompletePlanTuningRequest.CorrectnessPolicy.EXACT_CANONICAL_BYTES,
                                new CompletePlanTuningRequest.PolicyIdentity(4, policy),
                                2,
                                0,
                                1,
                                4,
                                0,
                                cache,
                                handoff);

        Arrays.fill(model, (byte) 0);
        Arrays.fill(profile, (byte) 0);
        Arrays.fill(target, (byte) 0);
        Arrays.fill(policy, (byte) 0);

        assertAll(
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("model"), request.modelFingerprint().bytes()),
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("profile"), request.profileFingerprint().bytes()),
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("target"), request.targetFingerprint().bytes()),
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("policy"), request.policyIdentity().bytes()),
                () -> assertSame(handoff, request.handoff()),
                () -> assertSame(batch, request.handoff().candidateBatch()),
                () -> assertSame(cache, request.modelPlanCache()),
                () -> assertFalse(Files.exists(directory.resolve("absent-parent"))));
    }

    @Test
    void compatibilitySnapshotsProducerCodecAndCompatibilityBytes() {
        byte[] producer = CompletePlanTuningTest.bytes("producer");
        byte[] codec = CompletePlanTuningTest.bytes("codec");
        byte[] compatibility = CompletePlanTuningTest.bytes("compatibility");

        CompletePlanTuningRequest.PlanCompatibility value =
                new CompletePlanTuningRequest.PlanCompatibility(
                        new CompletePlanTuningRequest.ProducerIdentity(1, producer),
                        new CompletePlanTuningRequest.DecisionCodecIdentity(2, codec),
                        3,
                        compatibility,
                        CompletePlanTuningRequest.ReuseScope.PERSISTENT);
        Arrays.fill(producer, (byte) 0);
        Arrays.fill(codec, (byte) 0);
        Arrays.fill(compatibility, (byte) 0);

        assertAll(
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("producer"), value.producerIdentity().bytes()),
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("codec"), value.decisionCodecIdentity().bytes()),
                () -> assertArrayEquals(CompletePlanTuningTest.bytes("compatibility"), value.bytes()),
                () -> assertEquals(3, value.schemaVersion()),
                () -> assertEquals(CompletePlanTuningRequest.ReuseScope.PERSISTENT, value.reuseScope()));
    }

    @Test
    void rejectsInvalidIdentityPolicyAndPathInputsWithoutFilesystemAccess() {
        byte[] oversized = new byte[ModelPlanCacheFile.MAX_OPAQUE_BYTES + 1];
        CompletePlanTuningRequest<
                        CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                base =
                        CompletePlanTuningTest.request(
                                directory.resolve("absent/cache.bin"), 2, 0, 1, 4);

        assertAll(
                () ->
                        assertThrows(
                                NullPointerException.class,
                                () -> new CompletePlanTuningRequest.ModelFingerprint(1, null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.ModelFingerprint(
                                                0, new byte[] {1})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.ProfileFingerprint(
                                                1, new byte[0])),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.TargetFingerprint(
                                                1, oversized)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.PolicyIdentity(
                                                -1, new byte[] {1})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.ProducerIdentity(
                                                1, oversized)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.DecisionCodecIdentity(
                                                1, new byte[0])),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new CompletePlanTuningRequest.CandidateIdentity(
                                                oversized)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        copyRequest(
                                                base,
                                                Path.of(""),
                                                2,
                                                0,
                                                1,
                                                4,
                                                0,
                                                Optional.empty())),
                () ->
                        assertThrows(
                                NullPointerException.class,
                                () ->
                                        copyRequest(
                                                base,
                                                null,
                                                2,
                                                0,
                                                1,
                                                4,
                                                0,
                                                Optional.empty())),
                () -> assertFalse(Files.exists(directory.resolve("absent"))));
    }

    @Test
    void rejectsEveryInvalidRequestBoundAndPresentDecision() {
        CompletePlanTuningRequest<
                        CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                base = CompletePlanTuningTest.request(directory.resolve("cache"), 2, 0, 1, 4);

        assertAll(
                () -> assertInvalidBounds(base, 0, 0, 1, 4, 0),
                () -> assertInvalidBounds(base, 2, -1, 1, 4, 0),
                () -> assertInvalidBounds(base, 2, 0, 0, 4, 0),
                () -> assertInvalidBounds(base, 2, 0, 2, 4, 0),
                () -> assertInvalidBounds(base, 2, 0, 1, 0, 0),
                () -> assertInvalidBounds(base, 2, 0, 1, 4, -1),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        copyRequest(
                                                base,
                                                base.modelPlanCache(),
                                                2,
                                                0,
                                                1,
                                                4,
                                                0,
                                                Optional.of(new CompletePlanTuningTest.Decision("a")))));
    }

    @Test
    void rejectsNullEmptyOverLimitAndNullContainingCandidateResultsBeforeCorrectness() {
        for (List<CompletePlanTuningTest.Candidate> supplied :
                Arrays.<List<CompletePlanTuningTest.Candidate>>asList(
                        null,
                        List.of(),
                        List.of(
                                new CompletePlanTuningTest.Candidate("a"),
                                new CompletePlanTuningTest.Candidate("b"),
                                new CompletePlanTuningTest.Candidate("c")),
                        Arrays.asList(
                                new CompletePlanTuningTest.Candidate("a"), null))) {
            CompletePlanTuningTest.Backend backend = sessionBackend();
            backend.suppliedCandidates = supplied;
            AtomicInteger actions = new AtomicInteger();

            assertThrows(
                    RuntimeException.class,
                    () ->
                            CompletePlanTuning.tune(
                                    CompletePlanTuningTest.request(
                                            directory.resolve("unused"), 2, 0, 1, 4),
                                    backend,
                                    countingCorrectness(actions),
                                    (batch, candidate) -> actions.incrementAndGet()));
            assertEquals(0, actions.get());
        }
    }

    @Test
    void rejectsDuplicateCandidateDuplicateIdentityAndUnstableIdentityBeforeCorrectness() {
        List<BackendCompletePlanTuning<
                        CompletePlanTuningTest.Batch,
                        CompletePlanTuningTest.Decision,
                        CompletePlanTuningTest.Candidate>>
                backends =
                        List.of(
                                backendWithCandidates(
                                        List.of(
                                                new CompletePlanTuningTest.Candidate("a"),
                                                new CompletePlanTuningTest.Candidate("a"))),
                                new CompletePlanTuningTest.Backend(
                                        CompletePlanTuningTest.scope(
                                                CompletePlanTuningRequest.ReuseScope.SESSION)) {
                                    @Override
                                    public CompletePlanTuningRequest.CandidateIdentity
                                            candidateIdentity(
                                                    CompletePlanTuningTest.Candidate candidate) {
                                        return new CompletePlanTuningRequest.CandidateIdentity(
                                                CompletePlanTuningTest.bytes("same"));
                                    }
                                },
                                new CompletePlanTuningTest.Backend(
                                        CompletePlanTuningTest.scope(
                                                CompletePlanTuningRequest.ReuseScope.SESSION)) {
                                    int identities;

                                    @Override
                                    public CompletePlanTuningRequest.CandidateIdentity
                                            candidateIdentity(
                                                    CompletePlanTuningTest.Candidate candidate) {
                                        identities++;
                                        return new CompletePlanTuningRequest.CandidateIdentity(
                                                CompletePlanTuningTest.bytes(
                                                        candidate.id() + identities));
                                    }
                                });

        for (var backend : backends) {
            AtomicInteger actions = new AtomicInteger();
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            CompletePlanTuning.tune(
                                    CompletePlanTuningTest.request(
                                            directory.resolve("unused"), 2, 0, 1, 4),
                                    backend,
                                    countingCorrectness(actions),
                                    (batch, candidate) -> actions.incrementAndGet()));
            assertEquals(0, actions.get());
        }
    }

    @Test
    void rejectsNullEmptyAndOversizedCandidateIdentityBeforeCorrectness() {
        for (int invalidIdentity : List.of(0, 1, 2)) {
            CompletePlanTuningTest.Backend backend =
                    new CompletePlanTuningTest.Backend(
                            CompletePlanTuningTest.scope(
                                    CompletePlanTuningRequest.ReuseScope.SESSION)) {
                        @Override
                        public CompletePlanTuningRequest.CandidateIdentity candidateIdentity(
                                CompletePlanTuningTest.Candidate candidate) {
                            return switch (invalidIdentity) {
                                case 0 -> null;
                                case 1 ->
                                        new CompletePlanTuningRequest.CandidateIdentity(
                                                new byte[0]);
                                default ->
                                        new CompletePlanTuningRequest.CandidateIdentity(
                                                new byte[ModelPlanCacheFile.MAX_OPAQUE_BYTES + 1]);
                            };
                        }
                    };
            AtomicInteger actions = new AtomicInteger();
            assertThrows(
                    RuntimeException.class,
                    () ->
                            CompletePlanTuning.tune(
                                    CompletePlanTuningTest.request(
                                            directory.resolve("unused"), 2, 0, 1, 4),
                                    backend,
                                    countingCorrectness(actions),
                                    (batch, candidate) -> actions.incrementAndGet()));
            assertEquals(0, actions.get());
        }
    }

    @Test
    void rejectsNullCompatibilityBeforeActionsAndNullSelectedDecisionAfterMeasurement() {
        AtomicInteger actions = new AtomicInteger();
        CompletePlanTuningTest.Backend nullCompatibility =
                new CompletePlanTuningTest.Backend(null);

        assertThrows(
                NullPointerException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                nullCompatibility,
                                countingCorrectness(actions),
                                (batch, candidate) -> actions.incrementAndGet()));
        assertEquals(0, actions.get());

        CompletePlanTuningTest.Backend nullDecision =
                new CompletePlanTuningTest.Backend(
                        CompletePlanTuningTest.scope(
                                CompletePlanTuningRequest.ReuseScope.SESSION)) {
                    @Override
                    public CompletePlanTuningTest.Decision selectedDecision(
                            CompletePlanTuningTest.Batch batch,
                            CompletePlanTuningTest.Candidate candidate) {
                        selectedCalls++;
                        return null;
                    }
                };

        assertThrows(
                NullPointerException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                nullDecision,
                                new CompletePlanTuningTest.Correctness(new ArrayList<>()),
                                (batch, candidate) -> {},
                                CompletePlanTuningTest.clock(0, 1, 2, 3)));
        assertEquals(1, nullDecision.selectedCalls);
    }

    @Test
    void rejectsBudgetAndCheckedArithmeticFailuresBeforeCorrectness() {
        AtomicInteger actions = new AtomicInteger();
        CompletePlanTuningTest.Backend backend = sessionBackend();

        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        CompletePlanTuning.tune(
                                                CompletePlanTuningTest.request(
                                                        directory.resolve("unused"),
                                                        2,
                                                        1,
                                                        3,
                                                        9),
                                                backend,
                                                countingCorrectness(actions),
                                                (batch, candidate) ->
                                                        actions.incrementAndGet())),
                () ->
                        assertThrows(
                                ArithmeticException.class,
                                () ->
                                        CompletePlanTuning.checkedTotalExecutions(
                                                1, Long.MAX_VALUE, 1, Long.MAX_VALUE)),
                () ->
                        assertThrows(
                                ArithmeticException.class,
                                () ->
                                        CompletePlanTuning.checkedTotalExecutions(
                                                Long.MAX_VALUE, 1, 1, Long.MAX_VALUE)),
                () -> assertEquals(0, actions.get()));
    }

    @Test
    void rejectsNullCorrectnessReferenceAndOutcomeBeforeTiming() {
        CompletePlanTuningTest.Backend backend = sessionBackend();
        AtomicInteger measurements = new AtomicInteger();

        assertThrows(
                NullPointerException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                backend,
                                new CompletePlanCorrectness<
                                        CompletePlanTuningTest.Batch,
                                        CompletePlanTuningTest.Candidate,
                                        String>() {
                                    @Override
                                    public String capture(
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return null;
                                    }

                                    @Override
                                    public Outcome compare(
                                            String reference,
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return Outcome.MATCH;
                                    }
                                },
                                (batch, candidate) -> measurements.incrementAndGet()));

        assertThrows(
                NullPointerException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                backend,
                                new CompletePlanCorrectness<
                                        CompletePlanTuningTest.Batch,
                                        CompletePlanTuningTest.Candidate,
                                        String>() {
                                    @Override
                                    public String capture(
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return "reference";
                                    }

                                    @Override
                                    public Outcome compare(
                                            String reference,
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return null;
                                    }
                                },
                                (batch, candidate) -> measurements.incrementAndGet()));
        assertEquals(0, measurements.get());
    }

    @Test
    void rejectsMonotonicClockReversalWithoutSelectingDecision() {
        CompletePlanTuningTest.Backend backend = sessionBackend();

        assertThrows(
                IllegalStateException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                backend,
                                new CompletePlanTuningTest.Correctness(new ArrayList<>()),
                                (batch, candidate) -> {},
                                CompletePlanTuningTest.clock(10, 9)));
        assertEquals(0, backend.selectedCalls);
    }

    @Test
    void mismatchHasExactTypedFailureAndStopsBeforeTiming() {
        CompletePlanTuningTest.Backend backend = sessionBackend();
        AtomicInteger measurements = new AtomicInteger();

        assertThrows(
                CompletePlanTuning.CorrectnessMismatchException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(
                                        directory.resolve("unused"), 2, 0, 1, 4),
                                backend,
                                new CompletePlanCorrectness<
                                        CompletePlanTuningTest.Batch,
                                        CompletePlanTuningTest.Candidate,
                                        String>() {
                                    @Override
                                    public String capture(
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return "reference";
                                    }

                                    @Override
                                    public Outcome compare(
                                            String reference,
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return Outcome.MISMATCH;
                                    }
                                },
                                (batch, candidate) -> measurements.incrementAndGet()));
        assertAll(
                () -> assertEquals(0, measurements.get()),
                () -> assertEquals(0, backend.selectedCalls),
                () -> assertEquals(0, backend.encodeCalls));
    }

    @Test
    void callerFailurePropagatesByIdentity() {
        CompletePlanTuningTest.Backend backend = sessionBackend();
        IOException failure = new IOException("cleanup failed");

        IOException thrown =
                assertThrows(
                        IOException.class,
                        () ->
                                CompletePlanTuning.tune(
                                        CompletePlanTuningTest.request(
                                                directory.resolve("unused"), 2, 0, 1, 4),
                                        backend,
                                        new CompletePlanCorrectness<
                                                CompletePlanTuningTest.Batch,
                                                CompletePlanTuningTest.Candidate,
                                                String>() {
                                            @Override
                                            public String capture(
                                                    CompletePlanTuningTest.Batch batch,
                                                    CompletePlanTuningTest.Candidate candidate,
                                                    long maximumBytes)
                                                    throws IOException {
                                                throw failure;
                                            }

                                            @Override
                                            public Outcome compare(
                                                    String reference,
                                                    CompletePlanTuningTest.Batch batch,
                                                    CompletePlanTuningTest.Candidate candidate,
                                                    long maximumBytes) {
                                                return Outcome.MATCH;
                                            }
                                        },
                                        (batch, candidate) -> {}));
        assertSame(failure, thrown);
    }

    private static CompletePlanTuningRequest<
                    CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
            copyRequest(
                    CompletePlanTuningRequest<
                                    CompletePlanTuningTest.Batch,
                                    CompletePlanTuningTest.Decision>
                            base,
                    Path path,
                    int maximumCandidates,
                    int warmups,
                    int samples,
                    long maximumExecutions,
                    long maximumCorrectnessBytes,
                    Optional<CompletePlanTuningTest.Decision> selectedDecision) {
        return new CompletePlanTuningRequest<>(
                base.modelFingerprint(),
                base.profileFingerprint(),
                base.targetFingerprint(),
                base.objective(),
                base.correctnessPolicy(),
                base.policyIdentity(),
                maximumCandidates,
                warmups,
                samples,
                maximumExecutions,
                maximumCorrectnessBytes,
                path,
                new BackendPartitionTuningHandoff<>(
                        base.handoff().partition(),
                        base.handoff().candidateBatch(),
                        selectedDecision));
    }

    private static void assertInvalidBounds(
            CompletePlanTuningRequest<
                            CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                    base,
            int maximumCandidates,
            int warmups,
            int samples,
            long maximumExecutions,
            long maximumCorrectnessBytes) {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        copyRequest(
                                base,
                                base.modelPlanCache(),
                                maximumCandidates,
                                warmups,
                                samples,
                                maximumExecutions,
                                maximumCorrectnessBytes,
                                Optional.empty()));
    }

    private static CompletePlanTuningTest.Backend sessionBackend() {
        return new CompletePlanTuningTest.Backend(
                CompletePlanTuningTest.scope(CompletePlanTuningRequest.ReuseScope.SESSION));
    }

    private static CompletePlanTuningTest.Backend backendWithCandidates(
            List<CompletePlanTuningTest.Candidate> candidates) {
        CompletePlanTuningTest.Backend backend = sessionBackend();
        backend.suppliedCandidates = candidates;
        return backend;
    }

    private static CompletePlanCorrectness<
                    CompletePlanTuningTest.Batch,
                    CompletePlanTuningTest.Candidate,
                    String>
            countingCorrectness(AtomicInteger actions) {
        return new CompletePlanCorrectness<>() {
            @Override
            public String capture(
                    CompletePlanTuningTest.Batch batch,
                    CompletePlanTuningTest.Candidate candidate,
                    long maximumBytes) {
                actions.incrementAndGet();
                return "reference";
            }

            @Override
            public Outcome compare(
                    String reference,
                    CompletePlanTuningTest.Batch batch,
                    CompletePlanTuningTest.Candidate candidate,
                    long maximumBytes) {
                actions.incrementAndGet();
                return Outcome.MATCH;
            }
        };
    }
}
