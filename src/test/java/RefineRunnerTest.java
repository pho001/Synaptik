import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CandidatePerfSource;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.RefineConfig;
import benchmark.autotune.RefinedCandidate;
import benchmark.autotune.RefineRunner;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

public class RefineRunnerTest {

    @Test
    void refineRunnerAveragesRepeatedMeasurementsAndKeepsOrder() {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        List<OptimizerCandidate> finalists = List.of(a, b);

        AtomicInteger calls = new AtomicInteger();
        List<String> tiers = new ArrayList<>();
        CandidatePerfSource source = (candidate, warmup, measure, tier, cache) -> {
            tiers.add(tier + ":" + warmup + ":" + measure);
            int call = calls.getAndIncrement();
            if (candidate.name().equals("A")) {
                return perf(candidate, 10, 20, 1.0 + call, 2.0 + call, 3.0 + call);
            }
            return perf(candidate, 30, 40, 10.0 + call, 20.0 + call, 30.0 + call);
        };

        List<RefinedCandidate> refined = RefineRunner.refine(finalists, new RefineConfig(3, 5, 7), source, () -> 0L);

        assertIterableEquals(List.of(a, b), refined.stream().map(r -> r.perf().candidate()).toList());
        CandidatePerf first = refined.get(0).perf();
        CandidatePerf second = refined.get(1).perf();

        assertEquals(10, first.graphInfSize());
        assertEquals(20, first.graphTrnSize());
        assertEquals((1.0 + 2.0 + 3.0) / 3.0, first.forwardMs(), 1e-12);
        assertEquals((2.0 + 3.0 + 4.0) / 3.0, first.trainMs(), 1e-12);
        assertEquals((3.0 + 4.0 + 5.0) / 3.0, first.broadcastMs(), 1e-12);

        assertEquals(30, second.graphInfSize());
        assertEquals(40, second.graphTrnSize());
        assertEquals((13.0 + 14.0 + 15.0) / 3.0, second.forwardMs(), 1e-12);
        assertEquals((23.0 + 24.0 + 25.0) / 3.0, second.trainMs(), 1e-12);
        assertEquals((33.0 + 34.0 + 35.0) / 3.0, second.broadcastMs(), 1e-12);

        assertEquals(
                List.of("REFINE:5:7", "REFINE:5:7", "REFINE:5:7", "REFINE:5:7", "REFINE:5:7", "REFINE:5:7"),
                tiers
        );
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }

    private static CandidatePerf perf(
            OptimizerCandidate candidate,
            int graphInf,
            int graphTrn,
            double forward,
            double train,
            double broadcast
    ) {
        return new CandidatePerf(
                candidate,
                "CSE",
                CoarseKnobSignature.of(candidate),
                graphInf,
                graphTrn,
                forward,
                train,
                broadcast
        );
    }
}
