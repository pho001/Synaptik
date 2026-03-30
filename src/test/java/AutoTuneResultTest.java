import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutoTuneResultTest {

    @Test
    void jsonContainsCandidateStageOrderAndMetrics() {
        OptimizerCandidate candidate = new OptimizerCandidate(
                "AUTO_JSON",
                List.of(OptimizationStage.CSE, OptimizationStage.AR, OptimizationStage.MEM),
                TuningKnobs.trainingDefaults()
        );
        AutoTuneResult result = new AutoTuneResult(candidate, 74, 141, 1.25, 2.50, 3.75, 1.99);

        String json = result.toJson(59, 0);

        assertTrue(json.contains("\"candidateName\": \"AUTO_JSON\""));
        assertTrue(json.contains("\"stageOrder\": [\"CSE\", \"AR\", \"MEM\"]"));
        assertTrue(json.contains("\"graphInfSize\": 74"));
        assertTrue(json.contains("\"graphTrnSize\": 141"));
        assertTrue(json.contains("\"validCandidates\": 59"));
        assertTrue(json.contains("\"mismatchedCandidates\": 0"));
    }
}
