package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class TuningInspectionTest {
    @TempDir Path temporary;

    @Test
    void workloadPathAndBytesAreEquivalentRedactedAndDecoderLimited() throws Exception {
        WorkloadCacheFile.Key key = new WorkloadCacheFile.Key(7, bytes("compatibility"), 1, 2, 3);
        WorkloadCacheFile.Entry entry = new WorkloadCacheFile.Entry(key, bytes("secret-decision"),
                bytes("secret-winner"), new WorkloadTuningResult.SampleSummary(10, 20, 30, 3));
        byte[] encoded = new WorkloadCacheFile().encode(Map.of(key, entry));
        Path path = temporary.resolve("workload.cache");
        Files.write(path, encoded);
        var expectation = new TuningInspection.WorkloadExpectation(
                new WorkloadTuningRequest.WorkloadCompatibility(7, bytes("compatibility"),
                        WorkloadTuningRequest.ReuseScope.PERSISTENT),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS, 2, 3);

        var fromPath = TuningInspection.inspectWorkloadCache(path, expectation);
        var fromBytes = TuningInspection.inspectWorkloadCache(encoded, expectation);

        assertEquals(TuningInspection.ArtifactStatus.VALID, fromPath.status());
        assertEquals(TuningInspection.ArtifactSource.PATH, fromPath.source());
        assertEquals(TuningInspection.ArtifactSource.BYTE_ARRAY, fromBytes.source());
        assertEquals(fromPath.entries(), fromBytes.entries());
        assertEquals(TuningInspection.CompatibilityStatus.KEY_MATCH_REQUIRES_BACKEND_DECODER,
                fromPath.entries().getFirst().compatibilityStatus());
        assertTrue(fromPath.entries().getFirst().mismatches().isEmpty());
        assertFalse(fromPath.toString().contains("secret-decision"));
        assertFalse(fromPath.toString().contains("secret-winner"));
        assertEquals(64, fromPath.sha256().length());
        assertEquals(encoded.length, fromPath.observedByteCount());
        assertEquals(TuningInspection.CompatibilityStatus.NOT_EVALUATED,
                TuningInspection.inspectWorkloadCache(encoded).entries().getFirst()
                        .compatibilityStatus());
    }

    @Test
    void workloadComparisonOrdersIndependentMismatchesAndForbidsSessionReuse() throws Exception {
        WorkloadCacheFile.Key key = new WorkloadCacheFile.Key(2, bytes("actual"), 1, 0, 3);
        WorkloadCacheFile.Entry entry = new WorkloadCacheFile.Entry(key, bytes("decision"),
                bytes("winner"), new WorkloadTuningResult.SampleSummary(1, 2, 3, 3));
        byte[] encoded = new WorkloadCacheFile().encode(Map.of(key, entry));
        var session = new TuningInspection.WorkloadExpectation(
                new WorkloadTuningRequest.WorkloadCompatibility(3, bytes("expected"),
                        WorkloadTuningRequest.ReuseScope.SESSION),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS, 4, 5);

        var inspected = TuningInspection.inspectWorkloadCache(encoded, session).entries().getFirst();
        assertEquals(TuningInspection.CompatibilityStatus.MISMATCH,
                inspected.compatibilityStatus());
        assertEquals(List.of(
                TuningInspection.MismatchReason.COMPATIBILITY_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.COMPATIBILITY_VALUE_MISMATCH,
                TuningInspection.MismatchReason.WARMUP_COUNT_MISMATCH,
                TuningInspection.MismatchReason.TIMED_SAMPLE_COUNT_MISMATCH,
                TuningInspection.MismatchReason.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE),
                inspected.mismatches());

        var exactSession = new TuningInspection.WorkloadExpectation(
                new WorkloadTuningRequest.WorkloadCompatibility(2, bytes("actual"),
                        WorkloadTuningRequest.ReuseScope.SESSION),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS, 0, 3);
        var exact = TuningInspection.inspectWorkloadCache(encoded, exactSession).entries().getFirst();
        assertEquals(TuningInspection.CompatibilityStatus.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE,
                exact.compatibilityStatus());
        assertEquals(List.of(
                TuningInspection.MismatchReason.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE),
                exact.mismatches());
    }

    @Test
    void everyWorkloadMismatchReasonIsReportedIndependently() throws Exception {
        var expectation = new TuningInspection.WorkloadExpectation(
                new WorkloadTuningRequest.WorkloadCompatibility(1, bytes("compatibility"),
                        WorkloadTuningRequest.ReuseScope.PERSISTENT),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS, 2, 3);
        for (TuningInspection.MismatchReason reason : List.of(
                TuningInspection.MismatchReason.COMPATIBILITY_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.COMPATIBILITY_VALUE_MISMATCH,
                TuningInspection.MismatchReason.OBJECTIVE_MISMATCH,
                TuningInspection.MismatchReason.WARMUP_COUNT_MISMATCH,
                TuningInspection.MismatchReason.TIMED_SAMPLE_COUNT_MISMATCH)) {
            WorkloadCacheFile.Key key = workloadKeyWithMismatch(reason);
            var summary = summary(key.timedSampleCount());
            byte[] encoded = new WorkloadCacheFile().encode(Map.of(key,
                    new WorkloadCacheFile.Entry(
                            key, bytes("decision"), bytes("winner"), summary)));
            var inspected = TuningInspection.inspectWorkloadCache(encoded, expectation)
                    .entries().getFirst();
            assertEquals(TuningInspection.CompatibilityStatus.MISMATCH,
                    inspected.compatibilityStatus(), reason.name());
            assertEquals(List.of(reason), inspected.mismatches(), reason.name());
        }
    }

    @Test
    void modelPlanInspectionPreservesCanonicalOrderAndExactKeySemantics() throws Exception {
        ModelPlanCacheFile.Key first = modelKey(bytes("a"));
        ModelPlanCacheFile.Key second = modelKey(bytes("b"));
        var summary = new CompletePlanTuningResult.SampleSummary(4, 5, 6, 3);
        var entries = new LinkedHashMap<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry>();
        entries.put(second, new ModelPlanCacheFile.Entry(second, bytes("decision-b"),
                bytes("winner-b"), summary));
        entries.put(first, new ModelPlanCacheFile.Entry(first, bytes("decision-a"),
                bytes("winner-a"), summary));
        Path path = temporary.resolve("plans.cache");
        new ModelPlanCacheFile().publish(path, entries);
        byte[] encoded = Files.readAllBytes(path);
        var expectation = modelExpectation(bytes("a"), CompletePlanTuningRequest.ReuseScope.PERSISTENT);

        var inspected = TuningInspection.inspectModelPlanCache(encoded, expectation);
        assertEquals(2, inspected.entries().size());
        assertEquals(TuningInspection.CompatibilityStatus.KEY_MATCH_REQUIRES_BACKEND_DECODER,
                inspected.entries().getFirst().compatibilityStatus());
        assertEquals(TuningInspection.CompatibilityStatus.MISMATCH,
                inspected.entries().get(1).compatibilityStatus());
        assertEquals(List.of(TuningInspection.MismatchReason.MODEL_VALUE_MISMATCH),
                inspected.entries().get(1).mismatches());
        assertEquals(inspected.entries(),
                TuningInspection.inspectModelPlanCache(path, expectation).entries());
        assertEquals(TuningInspection.CompatibilityStatus.NOT_EVALUATED,
                TuningInspection.inspectModelPlanCache(encoded).entries().getFirst()
                        .compatibilityStatus());
    }

    @Test
    void everyModelPlanMismatchReasonAndExactSessionAreReportedIndependently() throws Exception {
        var expectation = modelExpectation(
                bytes("model"), CompletePlanTuningRequest.ReuseScope.PERSISTENT);
        for (TuningInspection.MismatchReason reason : List.of(
                TuningInspection.MismatchReason.PRODUCER_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.PRODUCER_VALUE_MISMATCH,
                TuningInspection.MismatchReason.DECISION_CODEC_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.DECISION_CODEC_VALUE_MISMATCH,
                TuningInspection.MismatchReason.MODEL_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.MODEL_VALUE_MISMATCH,
                TuningInspection.MismatchReason.PROFILE_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.PROFILE_VALUE_MISMATCH,
                TuningInspection.MismatchReason.TARGET_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.TARGET_VALUE_MISMATCH,
                TuningInspection.MismatchReason.COMPATIBILITY_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.COMPATIBILITY_VALUE_MISMATCH,
                TuningInspection.MismatchReason.OBJECTIVE_MISMATCH,
                TuningInspection.MismatchReason.CORRECTNESS_POLICY_MISMATCH,
                TuningInspection.MismatchReason.POLICY_SCHEMA_MISMATCH,
                TuningInspection.MismatchReason.POLICY_VALUE_MISMATCH,
                TuningInspection.MismatchReason.WARMUP_COUNT_MISMATCH,
                TuningInspection.MismatchReason.TIMED_SAMPLE_COUNT_MISMATCH)) {
            ModelPlanCacheFile.Key key = modelKeyWithMismatch(reason);
            Path path = temporary.resolve("model-mismatch-" + reason.name());
            int samples = key.timedSampleCount();
            new ModelPlanCacheFile().publish(path, Map.of(key,
                    new ModelPlanCacheFile.Entry(key, bytes("decision"), bytes("winner"),
                            completeSummary(samples))));
            var inspected = TuningInspection.inspectModelPlanCache(
                    Files.readAllBytes(path), expectation).entries().getFirst();
            assertEquals(TuningInspection.CompatibilityStatus.MISMATCH,
                    inspected.compatibilityStatus(), reason.name());
            assertEquals(List.of(reason), inspected.mismatches(), reason.name());
        }

        ModelPlanCacheFile.Key exactKey = modelKeyWithMismatch(null);
        Path path = temporary.resolve("model-exact-session");
        new ModelPlanCacheFile().publish(path, Map.of(exactKey,
                new ModelPlanCacheFile.Entry(exactKey, bytes("decision"), bytes("winner"),
                        completeSummary(3))));
        var exactSession = TuningInspection.inspectModelPlanCache(Files.readAllBytes(path),
                modelExpectation(bytes("model"), CompletePlanTuningRequest.ReuseScope.SESSION))
                .entries().getFirst();
        assertEquals(TuningInspection.CompatibilityStatus.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE,
                exactSession.compatibilityStatus());
        assertEquals(List.of(
                TuningInspection.MismatchReason.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE),
                exactSession.mismatches());
    }

    @Test
    void evidenceSummariesAreDetachedAndCacheHitsFabricateNoRows() {
        var workloadEvidence = new WorkloadTuningResult.Evidence(
                new WorkloadTuningRequest.ModelFingerprint(1, bytes("model")),
                new WorkloadTuningRequest.ProfileFingerprint(2, bytes("profile")),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(3, 4, 1, 3),
                List.of(new WorkloadTuningResult.WorkloadEvidence(
                        new WorkloadTuningRequest.WorkloadCompatibility(5, bytes("compat"),
                                WorkloadTuningRequest.ReuseScope.PERSISTENT),
                        9, List.of(new WorkloadTuningResult.OccurrenceEvidence(bytes("context"), 9)),
                        WorkloadTuningResult.Source.CACHE_HIT, List.of(),
                        new WorkloadTuningRequest.CandidateIdentity(bytes("winner")),
                        new WorkloadTuningResult.SampleSummary(1, 2, 3, 3))));
        var workload = TuningInspection.summarize(workloadEvidence);
        assertTrue(workload.workloads().getFirst().candidates().isEmpty());
        assertEquals(9, workload.workloads().getFirst().totalWeight());
        assertThrows(UnsupportedOperationException.class,
                () -> workload.workloads().add(workload.workloads().getFirst()));

        var completeEvidence = new CompletePlanTuningResult.Evidence(
                new CompletePlanTuningRequest.ModelFingerprint(1, bytes("model")),
                new CompletePlanTuningRequest.ProfileFingerprint(1, bytes("profile")),
                new CompletePlanTuningRequest.TargetFingerprint(1, bytes("target")),
                CompletePlanTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                CompletePlanTuningRequest.CorrectnessPolicy.EXACT_CANONICAL_BYTES,
                new CompletePlanTuningRequest.PolicyIdentity(1, bytes("policy")),
                CompletePlanTuningResult.Source.CACHE_HIT, List.of(),
                new CompletePlanTuningRequest.CandidateIdentity(bytes("winner")),
                new CompletePlanTuningResult.SampleSummary(7, 8, 9, 3));
        var complete = TuningInspection.summarize(completeEvidence);
        assertTrue(complete.candidates().isEmpty());
        assertNotEquals("winner", complete.selectedCandidate().sha256());
    }

    @Test
    void measuredEvidencePreservesRichRowsOrderAndCorrectnessWithoutIo() throws Exception {
        List<Long> workloadSamples = new java.util.ArrayList<>(List.of(9L, 1L, 5L));
        var workloadCandidate = new WorkloadTuningResult.CandidateEvidence(
                new WorkloadTuningRequest.CandidateIdentity(bytes("candidate-one")),
                workloadSamples, new WorkloadTuningResult.SampleSummary(1, 5, 9, 3));
        var workloadEvidence = new WorkloadTuningResult.Evidence(
                new WorkloadTuningRequest.ModelFingerprint(4, bytes("model-secret")),
                new WorkloadTuningRequest.ProfileFingerprint(5, bytes("profile-secret")),
                WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new WorkloadTuningRequest.Budget(6, 7, 2, 3),
                List.of(new WorkloadTuningResult.WorkloadEvidence(
                        new WorkloadTuningRequest.WorkloadCompatibility(8, bytes("compat-secret"),
                                WorkloadTuningRequest.ReuseScope.PERSISTENT),
                        5,
                        List.of(
                                new WorkloadTuningResult.OccurrenceEvidence(bytes("context-a"), 2),
                                new WorkloadTuningResult.OccurrenceEvidence(bytes("context-b"), 3)),
                        WorkloadTuningResult.Source.MEASURED, List.of(workloadCandidate),
                        new WorkloadTuningRequest.CandidateIdentity(bytes("candidate-one")),
                        new WorkloadTuningResult.SampleSummary(1, 5, 9, 3))));
        workloadSamples.set(0, 99L);
        long entriesBefore;
        try (var stream = Files.list(temporary)) {
            entriesBefore = stream.count();
        }
        var workload = TuningInspection.summarize(workloadEvidence);
        var workloadSummary = workload.workloads().getFirst();
        assertEquals(WorkloadTuningResult.Source.MEASURED, workloadSummary.source());
        assertEquals(5, workloadSummary.totalWeight());
        assertEquals(List.of(2L, 3L), workloadSummary.occurrences().stream()
                .map(TuningInspection.OccurrenceEvidenceSummary::weight).toList());
        assertEquals(List.of(9L, 1L, 5L),
                workloadSummary.candidates().getFirst().elapsedSamplesNanos());
        assertEquals(1, workloadSummary.minimumNanos());
        assertEquals(5, workloadSummary.medianNanos());
        assertEquals(9, workloadSummary.maximumNanos());
        assertEquals(3, workloadSummary.sampleCount());
        assertFalse(workload.toString().contains("model-secret"));
        assertFalse(workload.toString().contains("context-a"));
        assertThrows(UnsupportedOperationException.class,
                () -> workloadSummary.candidates().getFirst().elapsedSamplesNanos().add(11L));

        List<Long> referenceSamples = new java.util.ArrayList<>(List.of(8L, 2L, 4L));
        List<Long> matchSamples = new java.util.ArrayList<>(List.of(7L, 3L, 5L));
        var completeEvidence = new CompletePlanTuningResult.Evidence(
                new CompletePlanTuningRequest.ModelFingerprint(1, bytes("complete-model")),
                new CompletePlanTuningRequest.ProfileFingerprint(2, bytes("complete-profile")),
                new CompletePlanTuningRequest.TargetFingerprint(3, bytes("complete-target")),
                CompletePlanTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                CompletePlanTuningRequest.CorrectnessPolicy.EXACT_CANONICAL_BYTES,
                new CompletePlanTuningRequest.PolicyIdentity(4, bytes("complete-policy")),
                CompletePlanTuningResult.Source.MEASURED,
                List.of(
                        new CompletePlanTuningResult.CandidateEvidence(
                                new CompletePlanTuningRequest.CandidateIdentity(bytes("reference")),
                                CompletePlanTuningResult.CorrectnessAction.REFERENCE_CAPTURED,
                                referenceSamples,
                                new CompletePlanTuningResult.SampleSummary(2, 4, 8, 3)),
                        new CompletePlanTuningResult.CandidateEvidence(
                                new CompletePlanTuningRequest.CandidateIdentity(bytes("match")),
                                CompletePlanTuningResult.CorrectnessAction.MATCH, matchSamples,
                                new CompletePlanTuningResult.SampleSummary(3, 5, 7, 3))),
                new CompletePlanTuningRequest.CandidateIdentity(bytes("match")),
                new CompletePlanTuningResult.SampleSummary(3, 5, 7, 3));
        referenceSamples.set(0, 100L);
        matchSamples.set(0, 101L);
        var complete = TuningInspection.summarize(completeEvidence);
        assertEquals(CompletePlanTuningResult.Source.MEASURED, complete.source());
        assertEquals(List.of(
                        CompletePlanTuningResult.CorrectnessAction.REFERENCE_CAPTURED,
                        CompletePlanTuningResult.CorrectnessAction.MATCH),
                complete.candidates().stream()
                        .map(TuningInspection.CompleteCandidateMeasurementSummary::correctnessAction)
                        .toList());
        assertEquals(List.of(8L, 2L, 4L),
                complete.candidates().getFirst().elapsedSamplesNanos());
        assertEquals(List.of(7L, 3L, 5L),
                complete.candidates().get(1).elapsedSamplesNanos());
        assertEquals(3, complete.minimumNanos());
        assertEquals(5, complete.medianNanos());
        assertEquals(7, complete.maximumNanos());
        assertEquals(3, complete.sampleCount());
        assertFalse(complete.toString().contains("complete-model"));
        assertFalse(complete.toString().contains("reference"));
        assertThrows(UnsupportedOperationException.class,
                () -> complete.candidates().add(complete.candidates().getFirst()));
        try (var stream = Files.list(temporary)) {
            assertEquals(entriesBefore, stream.count());
        }
    }

    @Test
    void byteArrayIsSnapshottedAndNeverRetained() throws Exception {
        WorkloadCacheFile.Key key = new WorkloadCacheFile.Key(1, bytes("key"), 1, 0, 1);
        byte[] supplied = new WorkloadCacheFile().encode(Map.of(key,
                new WorkloadCacheFile.Entry(key, bytes("decision"), bytes("winner"),
                        new WorkloadTuningResult.SampleSummary(1, 1, 1, 1))));
        byte[] original = supplied.clone();
        var report = TuningInspection.inspectWorkloadCache(supplied);
        supplied[0] ^= 0x7f;
        assertEquals(TuningInspection.ArtifactStatus.VALID, report.status());
        assertArrayEquals(original, new WorkloadCacheFile().encode(Map.of(key,
                new WorkloadCacheFile.Entry(key, bytes("decision"), bytes("winner"),
                        new WorkloadTuningResult.SampleSummary(1, 1, 1, 1)))));
        assertFalse(report.toString().contains(Arrays.toString(original)));
    }

    private static ModelPlanCacheFile.Key modelKey(byte[] model) {
        return new ModelPlanCacheFile.Key(1, bytes("producer"), 1, bytes("codec"), 1, model,
                1, bytes("profile"), 1, bytes("target"), 1, bytes("compatibility"), 1, 1,
                1, bytes("policy"), 2, 3);
    }

    private static WorkloadCacheFile.Key workloadKeyWithMismatch(
            TuningInspection.MismatchReason reason) {
        int schema = 1;
        byte[] compatibility = bytes("compatibility");
        int objective = 1;
        int warmups = 2;
        int samples = 3;
        switch (reason) {
            case COMPATIBILITY_SCHEMA_MISMATCH -> schema = 2;
            case COMPATIBILITY_VALUE_MISMATCH -> compatibility = bytes("different");
            case OBJECTIVE_MISMATCH -> objective = 2;
            case WARMUP_COUNT_MISMATCH -> warmups = 4;
            case TIMED_SAMPLE_COUNT_MISMATCH -> samples = 5;
            default -> throw new IllegalArgumentException("not a workload mismatch: " + reason);
        }
        return new WorkloadCacheFile.Key(schema, compatibility, objective, warmups, samples);
    }

    private static ModelPlanCacheFile.Key modelKeyWithMismatch(
            TuningInspection.MismatchReason reason) {
        int producerSchema = 1;
        byte[] producer = bytes("producer");
        int codecSchema = 1;
        byte[] codec = bytes("codec");
        int modelSchema = 1;
        byte[] model = bytes("model");
        int profileSchema = 1;
        byte[] profile = bytes("profile");
        int targetSchema = 1;
        byte[] target = bytes("target");
        int compatibilitySchema = 1;
        byte[] compatibility = bytes("compatibility");
        int objective = 1;
        int correctness = 1;
        int policySchema = 1;
        byte[] policy = bytes("policy");
        int warmups = 2;
        int samples = 3;
        if (reason != null) {
            switch (reason) {
                case PRODUCER_SCHEMA_MISMATCH -> producerSchema = 2;
                case PRODUCER_VALUE_MISMATCH -> producer = bytes("different-producer");
                case DECISION_CODEC_SCHEMA_MISMATCH -> codecSchema = 2;
                case DECISION_CODEC_VALUE_MISMATCH -> codec = bytes("different-codec");
                case MODEL_SCHEMA_MISMATCH -> modelSchema = 2;
                case MODEL_VALUE_MISMATCH -> model = bytes("different-model");
                case PROFILE_SCHEMA_MISMATCH -> profileSchema = 2;
                case PROFILE_VALUE_MISMATCH -> profile = bytes("different-profile");
                case TARGET_SCHEMA_MISMATCH -> targetSchema = 2;
                case TARGET_VALUE_MISMATCH -> target = bytes("different-target");
                case COMPATIBILITY_SCHEMA_MISMATCH -> compatibilitySchema = 2;
                case COMPATIBILITY_VALUE_MISMATCH -> compatibility = bytes("different-compat");
                case OBJECTIVE_MISMATCH -> objective = 2;
                case CORRECTNESS_POLICY_MISMATCH -> correctness = 2;
                case POLICY_SCHEMA_MISMATCH -> policySchema = 2;
                case POLICY_VALUE_MISMATCH -> policy = bytes("different-policy");
                case WARMUP_COUNT_MISMATCH -> warmups = 4;
                case TIMED_SAMPLE_COUNT_MISMATCH -> samples = 5;
                default -> throw new IllegalArgumentException(
                        "not a model-plan mismatch: " + reason);
            }
        }
        return new ModelPlanCacheFile.Key(producerSchema, producer, codecSchema, codec,
                modelSchema, model, profileSchema, profile, targetSchema, target,
                compatibilitySchema, compatibility, objective, correctness, policySchema, policy,
                warmups, samples);
    }

    private static WorkloadTuningResult.SampleSummary summary(int sampleCount) {
        return sampleCount == 3
                ? new WorkloadTuningResult.SampleSummary(1, 2, 3, 3)
                : new WorkloadTuningResult.SampleSummary(1, 3, 5, 5);
    }

    private static CompletePlanTuningResult.SampleSummary completeSummary(int sampleCount) {
        return sampleCount == 3
                ? new CompletePlanTuningResult.SampleSummary(1, 2, 3, 3)
                : new CompletePlanTuningResult.SampleSummary(1, 3, 5, 5);
    }

    private static TuningInspection.ModelPlanExpectation modelExpectation(
            byte[] model, CompletePlanTuningRequest.ReuseScope scope) {
        var compatibility = new CompletePlanTuningRequest.PlanCompatibility(
                new CompletePlanTuningRequest.ProducerIdentity(1, bytes("producer")),
                new CompletePlanTuningRequest.DecisionCodecIdentity(1, bytes("codec")),
                1, bytes("compatibility"), scope);
        return new TuningInspection.ModelPlanExpectation(compatibility,
                new CompletePlanTuningRequest.ModelFingerprint(1, model),
                new CompletePlanTuningRequest.ProfileFingerprint(1, bytes("profile")),
                new CompletePlanTuningRequest.TargetFingerprint(1, bytes("target")),
                CompletePlanTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                CompletePlanTuningRequest.CorrectnessPolicy.EXACT_CANONICAL_BYTES,
                new CompletePlanTuningRequest.PolicyIdentity(1, bytes("policy")), 2, 3);
    }

    private static byte[] bytes(String value) {
        return value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }
}
