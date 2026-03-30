import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.Phase1FinalistSelector;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

public class Phase1FinalistSelectorTest {

    @Test
    void selectorReturnsUnionOfTopTrainingAndInferenceWithoutDuplicates() {
        Row a = row("A", 1.0, 4.0);
        Row b = row("B", 2.0, 1.0);
        Row c = row("C", 3.0, 2.0);
        Row d = row("D", 4.0, 3.0);

        List<Row> finalists = Phase1FinalistSelector.selectFinalists(
                List.of(a, b, c, d),
                Row::trainingScore,
                Row::inferenceScore,
                Row::candidate,
                2
        );

        assertEquals(3, finalists.size());
        assertIterableEquals(List.of(a, b, c), finalists);
    }

    @Test
    void selectorRespectsTopKBoundPerObjective() {
        Row a = row("A", 1.0, 1.0);
        Row b = row("B", 2.0, 2.0);
        Row c = row("C", 3.0, 3.0);

        List<Row> finalists = Phase1FinalistSelector.selectFinalists(
                List.of(a, b, c),
                Row::trainingScore,
                Row::inferenceScore,
                Row::candidate,
                1
        );

        assertIterableEquals(List.of(a), finalists);
    }

    private static Row row(String name, double trainingScore, double inferenceScore) {
        return new Row(new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults()), trainingScore, inferenceScore);
    }

    private record Row(OptimizerCandidate candidate, double trainingScore, double inferenceScore) {}
}
