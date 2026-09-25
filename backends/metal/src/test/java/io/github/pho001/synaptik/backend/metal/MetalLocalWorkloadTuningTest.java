package io.github.pho001.synaptik.backend.metal;

import static io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGeneratorTest.analyze;
import static io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGeneratorTest.withInputs;
import static io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGeneratorTest.workload;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class MetalLocalWorkloadTuningTest {
    @Test
    void exactSingletonNegRoundTripsBothRoutesAndRetainsTraceProducer() {
        var api = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            MetalTraceProducer traceProducer = new MetalTraceProducer(ignored -> { });
            var tuning = new MetalLocalWorkloadTuning(device, traceProducer);
            var singleton = workload(device, 10_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);

            var handoff = tuning.candidateHandoff(singleton.context()).orElseThrow();
            var batch = handoff.candidateBatch();
            var candidates = tuning.candidates(batch);
            assertSame(singleton.context().partition(), handoff.partition());
            assertEquals(2, candidates.size());
            assertEquals(MetalLocalWorkloadTuning.ReuseScope.SESSION,
                    tuning.compatibility(batch).reuseScope());
            assertEquals(MetalNegTuningBatch.COMPATIBILITY_SCHEMA_VERSION,
                    tuning.compatibility(batch).schemaVersion());
            assertFalse(Arrays.equals(
                    tuning.candidateIdentity(candidates.get(0)).bytes(),
                    tuning.candidateIdentity(candidates.get(1)).bytes()));

            byte[] compatibility = tuning.compatibility(batch).bytes();
            byte[] mutatedCompatibility = tuning.compatibility(batch).bytes();
            mutatedCompatibility[0] ^= 1;
            assertArrayEquals(compatibility, tuning.compatibility(batch).bytes());

            var selected = tuning.selectedDecision(batch, candidates.get(1));
            byte[] encoded = tuning.encodeDecision(selected);
            assertEquals(Optional.of(selected),
                    tuning.decodeCompatibleDecision(batch, encoded));
            byte[] corrupt = encoded.clone();
            corrupt[corrupt.length - 1] ^= 1;
            assertTrue(tuning.decodeCompatibleDecision(batch, corrupt).isEmpty());

            var trialPreparation = tuning.trialPreparation(batch, candidates.get(1));
            var trialInputs = (MetalNegAnalysisInputs) trialPreparation.backendInputs();
            assertSame(traceProducer, trialInputs.traceProducer());
            assertEquals(MetalNegPreparationPlan.Route.MPSGRAPH,
                    analyze(withInputs(singleton, trialInputs).context()).plan().route());
        }
    }

    @Test
    void mpsGraphOnlyChangedForeignAndClosedAssociationsFailClosed() {
        var api = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        var otherApi = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            var tuning = new MetalLocalWorkloadTuning(device, null);
            var singleton = workload(device, 20_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var handoff = tuning.candidateHandoff(singleton.context()).orElseThrow();
            var batch = handoff.candidateBatch();
            var candidates = tuning.candidates(batch);
            byte[] encoded = tuning.encodeDecision(
                    tuning.selectedDecision(batch, candidates.getFirst()));

            var changed = workload(device, 30_000, Shape.of(5), false,
                    Optional.empty(), true, false, 1);
            var changedBatch = tuning.candidateHandoff(changed.context())
                    .orElseThrow().candidateBatch();
            assertTrue(tuning.decodeCompatibleDecision(changedBatch, encoded).isEmpty());
            var mpsGraphOnly = workload(device, 40_000, Shape.of(4), false,
                    Optional.empty(), true, false, 2);
            assertTrue(tuning.candidateHandoff(mpsGraphOnly.context()).isEmpty());

            var foreignTuning = new MetalLocalWorkloadTuning(otherDevice, null);
            var foreignWorkload = workload(otherDevice, 50_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var foreignHandoff = foreignTuning.candidateHandoff(
                    foreignWorkload.context()).orElseThrow();
            assertThrows(IllegalArgumentException.class, () -> tuning.selectedDecision(
                    batch,
                    foreignTuning.candidates(foreignHandoff.candidateBatch()).getFirst()));
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.candidateHandoff(foreignWorkload.context()));

            MetalBackendRuntime runtime = new MetalBackendRuntime(device);
            assertSame(runtime.localWorkloadTuning(), runtime.localWorkloadTuning());
            assertSame(runtime.completePlanTuning(), runtime.completePlanTuning());

            device.close();
            assertThrows(IllegalStateException.class, () -> tuning.candidates(batch));
        }
    }
}
