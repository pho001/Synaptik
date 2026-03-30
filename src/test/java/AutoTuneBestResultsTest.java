import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneBestResults;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AutoTuneBestResultsTest {

    @Test
    void updateTracksBestTrainingAndInferenceIndependently() {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        OptimizerCandidate c = candidate("C");

        AutoTuneBestResults best = new AutoTuneBestResults(null, null);
        best = best.update(perf(a, 1.0, 5.0, 0.1));
        best = best.update(perf(b, 2.0, 1.0, 0.1));
        best = best.update(perf(c, 3.0, 3.0, 0.1));

        assertEquals("B", best.training().candidate().name());
        assertEquals("A", best.inference().candidate().name());
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, double forward, double train, double broadcast) {
        return new CandidatePerf(
                candidate,
                "CSE",
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forward,
                train,
                broadcast
        );
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }
}
