import Benchmark.measure.MeasurementObjective;
import Benchmark.measure.MeasurementScoring;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MeasurementScoringTest {

    @Test
    void inferenceScoreIgnoresTrainAndBroadcast() {
        double score = MeasurementScoring.score(1.25, 9.0, 7.0, 74, 141, MeasurementObjective.INFERENCE);
        assertEquals(1.25 + (0.0005 * 74), score, 1e-12);
    }

    @Test
    void trainingScoreMatchesExistingFormula() {
        double score = MeasurementScoring.score(1.25, 2.50, 7.0, 74, 141, MeasurementObjective.TRAINING);
        assertEquals((0.35 * 1.25) + (0.50 * 2.50) + (0.0005 * 141), score, 1e-12);
    }
}
