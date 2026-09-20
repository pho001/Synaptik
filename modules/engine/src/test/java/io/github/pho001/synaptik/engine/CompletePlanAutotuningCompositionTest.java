package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks the public complete-plan evidence shape, ordering, and source-dependent invariants. */
final class CompletePlanAutotuningCompositionTest {
    @Test
    void measuredEvidencePreservesCorrectnessAndSampleEncounterOrder() {
        var budget = new ModelAutotuningConfig.CompletePlanBudget(2, 1, 3, 10, 64);
        var compatibility = new ModelAutotuningPreparation.CompatibilityIdentity(
                4, new byte[] {7, 8}, ModelAutotuningPreparation.ReuseScope.SESSION);
        var firstIdentity = new ModelAutotuningPreparation.CandidateIdentity(new byte[] {1});
        var secondIdentity = new ModelAutotuningPreparation.CandidateIdentity(new byte[] {2});
        var firstSummary = new ModelAutotuningPreparation.SampleSummary(1, 3, 5, 3);
        var secondSummary = new ModelAutotuningPreparation.SampleSummary(2, 4, 6, 3);
        var mutable = new ArrayList<>(List.of(
                new ModelAutotuningPreparation.CompletePlanCandidateEvidence(
                        firstIdentity,
                        ModelAutotuningPreparation.CorrectnessAction.REFERENCE_CAPTURED,
                        List.of(5L, 1L, 3L), firstSummary),
                new ModelAutotuningPreparation.CompletePlanCandidateEvidence(
                        secondIdentity,
                        ModelAutotuningPreparation.CorrectnessAction.MATCH,
                        List.of(4L, 6L, 2L), secondSummary)));

        var evidence = new ModelAutotuningPreparation.CompletePlanEvidence(
                compatibility, budget, ModelAutotuningPreparation.Source.MEASURED,
                mutable, firstIdentity, firstSummary);
        mutable.clear();

        assertSame(compatibility, evidence.compatibility());
        assertSame(budget, evidence.budget());
        assertEquals(List.of(
                ModelAutotuningPreparation.CorrectnessAction.REFERENCE_CAPTURED,
                ModelAutotuningPreparation.CorrectnessAction.MATCH),
                evidence.candidates().stream().map(
                        ModelAutotuningPreparation.CompletePlanCandidateEvidence
                                ::correctnessAction).toList());
        assertEquals(List.of(5L, 1L, 3L),
                evidence.candidates().getFirst().elapsedSamplesNanos());
        assertThrows(UnsupportedOperationException.class,
                () -> evidence.candidates().clear());
    }

    @Test
    void sourceControlsCandidateRowsAndOldEvidenceShapeIsAbsent() {
        var budget = new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1, 0);
        var compatibility = new ModelAutotuningPreparation.CompatibilityIdentity(
                1, new byte[] {1}, ModelAutotuningPreparation.ReuseScope.PERSISTENT);
        var winner = new ModelAutotuningPreparation.CandidateIdentity(new byte[] {2});
        var summary = new ModelAutotuningPreparation.SampleSummary(3, 3, 3, 1);

        var hit = new ModelAutotuningPreparation.CompletePlanEvidence(
                compatibility, budget, ModelAutotuningPreparation.Source.CACHE_HIT,
                List.of(), winner, summary);
        assertEquals(List.of(), hit.candidates());
        assertThrows(IllegalArgumentException.class, () ->
                new ModelAutotuningPreparation.CompletePlanEvidence(
                        compatibility, budget, ModelAutotuningPreparation.Source.MEASURED,
                        List.of(), winner, summary));

        List<List<Class<?>>> constructors = Arrays.stream(
                        ModelAutotuningPreparation.Evidence.class.getConstructors())
                .map(Constructor::getParameterTypes)
                .map(Arrays::asList)
                .toList();
        assertEquals(List.of(List.of(
                ModelAutotuningRequest.ModelIdentity.class,
                ModelAutotuningConfig.RepresentativeProfileIdentity.class,
                ModelAutotuningConfig.Objective.class,
                ModelAutotuningConfig.Budget.class,
                List.class,
                ModelAutotuningPreparation.CompletePlanEvidence.class)), constructors);
    }

    @Test
    void completePlanBudgetAndExplicitPathRemainIndependentFromPhaseOne() {
        var phaseOne = new ModelAutotuningConfig.Budget(1, 2, 3, 5);
        var phaseTwo = new ModelAutotuningConfig.CompletePlanBudget(7, 11, 13, 175, 19);
        Path modelPlan = Path.of("explicit-model-plan.bin");
        var config = new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                phaseOne,
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {1}),
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                Path.of("workload.bin"),
                phaseTwo,
                modelPlan);

        assertSame(phaseOne, config.budget());
        assertSame(phaseTwo, config.completePlanBudget());
        assertSame(modelPlan, config.modelPlanCache());
    }
}
