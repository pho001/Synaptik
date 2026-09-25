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

final class MetalCompletePlanTuningTest {
    @Test
    void authenticatedPhaseOneRouteProducesOneDefensiveFixedPlanCandidate() {
        var api = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            MetalTraceProducer traceProducer = new MetalTraceProducer(ignored -> { });
            var local = new MetalLocalWorkloadTuning(device, traceProducer);
            var complete = new MetalCompletePlanTuning(local);
            var singleton = workload(device, 60_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var localHandoff = local.candidateHandoff(singleton.context()).orElseThrow();
            var localBatch = localHandoff.candidateBatch();
            var localCandidates = local.candidates(localBatch);
            var mpsDecision = local.selectedDecision(localBatch, localCandidates.get(1));

            var handoff = complete.candidateHandoff(singleton.context(), mpsDecision);
            var batch = handoff.candidateBatch();
            var candidates = complete.candidates(batch);
            assertSame(singleton.context().partition(), handoff.partition());
            assertEquals(1, candidates.size());
            assertEquals(MetalCompletePlanTuning.ReuseScope.SESSION,
                    complete.compatibility(batch).reuseScope());
            byte[] compatibility = complete.compatibility(batch).bytes();
            byte[] mutatedCompatibility = complete.compatibility(batch).bytes();
            mutatedCompatibility[0] ^= 1;
            assertArrayEquals(compatibility, complete.compatibility(batch).bytes());

            var selected = complete.selectedDecision(batch, candidates.getFirst());
            byte[] encoded = complete.encodeDecision(selected);
            assertEquals(Optional.of(selected),
                    complete.decodeCompatibleDecision(batch, encoded));
            byte[] corrupt = encoded.clone();
            corrupt[corrupt.length - 1] ^= 1;
            assertTrue(complete.decodeCompatibleDecision(batch, corrupt).isEmpty());

            var preparation = complete.selectedPreparation(batch, selected);
            var inputs = (MetalNegAnalysisInputs) preparation.backendInputs();
            assertSame(traceProducer, inputs.traceProducer());
            assertEquals(MetalNegPreparationPlan.Route.MPSGRAPH,
                    analyze(withInputs(singleton, inputs).context()).plan().route());

            var customDecision = local.selectedDecision(localBatch, localCandidates.getFirst());
            var customHandoff = complete.candidateHandoff(
                    singleton.context(), customDecision);
            var customCandidate = complete.candidates(
                    customHandoff.candidateBatch()).getFirst();
            assertFalse(Arrays.equals(
                    complete.candidateIdentity(candidates.getFirst()).bytes(),
                    complete.candidateIdentity(customCandidate).bytes()));
            assertTrue(complete.decodeCompatibleDecision(
                    customHandoff.candidateBatch(), encoded).isEmpty());
        }
    }

    @Test
    void foreignPhaseOneAssociationAndClosedSessionAreRejected() {
        var api = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        var otherApi = new MetalNegRouteCandidateGeneratorTest.TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            var local = new MetalLocalWorkloadTuning(device, null);
            var complete = new MetalCompletePlanTuning(local);
            var singleton = workload(device, 70_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var localHandoff = local.candidateHandoff(singleton.context()).orElseThrow();
            var localBatch = localHandoff.candidateBatch();
            var localDecision = local.selectedDecision(
                    localBatch, local.candidates(localBatch).getFirst());
            var handoff = complete.candidateHandoff(singleton.context(), localDecision);
            var batch = handoff.candidateBatch();

            var foreignLocal = new MetalLocalWorkloadTuning(otherDevice, null);
            var foreignWorkload = workload(otherDevice, 80_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var foreignHandoff = foreignLocal.candidateHandoff(
                    foreignWorkload.context()).orElseThrow();
            var foreignDecision = foreignLocal.selectedDecision(
                    foreignHandoff.candidateBatch(),
                    foreignLocal.candidates(foreignHandoff.candidateBatch()).getFirst());
            assertThrows(IllegalArgumentException.class,
                    () -> complete.candidateHandoff(singleton.context(), foreignDecision));
            assertThrows(IllegalArgumentException.class,
                    () -> complete.candidateHandoff(foreignWorkload.context(), localDecision));

            device.close();
            assertThrows(IllegalStateException.class, () -> complete.candidates(batch));
        }
    }
}
