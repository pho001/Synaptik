import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneResult;
import benchmark.autotune.FinalistPreparation;
import benchmark.autotune.FinalistPreparationResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

public class FinalistPreparationTest {

    @Test
    void selectsFinalistsFromPhase1WithoutPostcheck() {
        AutoTuneResult a = result("A", 1.0, 4.0);
        AutoTuneResult b = result("B", 2.0, 1.0);
        AutoTuneResult c = result("C", 3.0, 2.0);

        FinalistPreparationResult prepared = FinalistPreparation.prepare(
                List.of(a, b, c),
                2,
                false,
                finalists -> finalists
        );

        assertEquals(FinalistPreparationResult.Status.OK, prepared.status());
        assertIterableEquals(List.of("A", "B", "C"), prepared.finalists().stream().map(OptimizerCandidate::name).toList());
    }

    @Test
    void reportsEmptyAfterPostcheckWhenCallbackDropsEverything() {
        AutoTuneResult a = result("A", 1.0, 1.0);

        FinalistPreparationResult prepared = FinalistPreparation.prepare(
                List.of(a),
                1,
                true,
                finalists -> List.of()
        );

        assertEquals(FinalistPreparationResult.Status.EMPTY_AFTER_POSTCHECK, prepared.status());
        assertEquals(0, prepared.finalists().size());
    }

    private static AutoTuneResult result(String name, double trainingScore, double inferenceScore) {
        OptimizerCandidate candidate = new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
        double forward = inferenceScore;
        double train = (trainingScore - (0.35 * forward) - (0.0005 * 20)) / 0.50;
        return new AutoTuneResult(candidate, 10, 20, forward, train, 0.1, trainingScore);
    }
}
