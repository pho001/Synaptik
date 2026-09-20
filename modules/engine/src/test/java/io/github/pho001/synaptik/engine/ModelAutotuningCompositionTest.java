package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Locks public request/evidence immutability and standalone validation. */
final class ModelAutotuningCompositionTest {
    @Test
    void requestSnapshotsOnlyContainerAndIdentitiesDefensivelyCopyBytes() {
        var config = config();
        var identityBytes = new byte[] {1, 2};
        var identity = new ModelAutotuningRequest.ModelIdentity(3, identityBytes);
        var tensor = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, Shape.of(1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(1))), false));
        var mutable = new ArrayList<>(List.of(tensor));

        var request = new ModelAutotuningRequest(config, identity, mutable);
        mutable.clear();
        identityBytes[0] = 9;

        assertSame(config, request.config());
        assertSame(identity, request.modelIdentity());
        assertSame(tensor, request.representativeInputs().getFirst());
        assertSame(request.representativeInputs(), request.representativeInputs());
        assertThrows(UnsupportedOperationException.class,
                () -> request.representativeInputs().clear());
        assertArrayEquals(new byte[] {1, 2}, identity.bytes());
        assertNotSame(identity.bytes(), identity.bytes());
        assertEquals(new ModelAutotuningRequest.ModelIdentity(3, new byte[] {1, 2}), identity);
        assertThrows(NullPointerException.class,
                () -> new ModelAutotuningRequest(config, identity,
                        java.util.Arrays.asList(tensor, null)));
    }

    @Test
    void evidenceSnapshotsCollectionsAndChecksSamplesAndSource() {
        var summary = new ModelAutotuningPreparation.SampleSummary(1, 3, 5, 3);
        var candidate = new ModelAutotuningPreparation.CandidateEvidence(
                new ModelAutotuningPreparation.CandidateIdentity(new byte[] {4}),
                new ArrayList<>(List.of(5L, 1L, 3L)), summary);
        var occurrence = new ModelAutotuningPreparation.OccurrenceEvidence(0, 0, 1,
                new ModelAutotuningPreparation.ContextIdentity(1, new byte[8]));
        var workload = new ModelAutotuningPreparation.WorkloadEvidence(
                new ModelAutotuningPreparation.CompatibilityIdentity(1, new byte[] {7},
                        ModelAutotuningPreparation.ReuseScope.SESSION),
                1, List.of(occurrence), ModelAutotuningPreparation.Source.MEASURED,
                List.of(candidate), candidate.identity(), summary);
        var evidence = new ModelAutotuningPreparation.Evidence(
                new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}),
                config().representativeProfile(), config().objective(), config().budget(),
                List.of(workload), new ModelAutotuningPreparation.CompletePlanEvidence(
                        workload.compatibility(), config().completePlanBudget(),
                        ModelAutotuningPreparation.Source.MEASURED,
                        List.of(new ModelAutotuningPreparation.CompletePlanCandidateEvidence(
                                candidate.identity(),
                                ModelAutotuningPreparation.CorrectnessAction.REFERENCE_CAPTURED,
                                candidate.elapsedSamplesNanos(), summary)),
                        candidate.identity(), summary));

        assertEquals(List.of(5L, 1L, 3L), candidate.elapsedSamplesNanos());
        assertEquals(1, evidence.workloads().size());
        assertThrows(IllegalArgumentException.class, () ->
                new ModelAutotuningPreparation.CandidateEvidence(candidate.identity(),
                        List.of(1L, 2L, 5L), summary));
        assertThrows(IllegalArgumentException.class, () ->
                new ModelAutotuningPreparation.WorkloadEvidence(workload.compatibility(), 1,
                        List.of(occurrence), ModelAutotuningPreparation.Source.CACHE_HIT,
                        List.of(candidate), candidate.identity(), summary));
    }

    private static ModelAutotuningConfig config() {
        return new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 2, 0, 1),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {2}),
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                Path.of("cache.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 0L),
                Path.of("unused-model-plan-cache.bin"));
    }
}
