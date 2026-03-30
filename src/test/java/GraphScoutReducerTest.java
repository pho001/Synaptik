import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.BeamSearchConfig;
import benchmark.autotune.CandidateEvalCache;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.GraphScoutConfig;
import benchmark.autotune.GraphScoutReducer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GraphScoutReducerTest {

    @Test
    void reducerKeepsBestFamiliesAndAppliesPerStageCap() {
        List<OptimizerCandidate> candidates = new ArrayList<>();
        candidates.add(candidate("A1", List.of(OptimizationStage.CSE)));
        candidates.add(candidate("A2", List.of(OptimizationStage.CSE)));
        candidates.add(candidate("A3", List.of(OptimizationStage.CSE)));
        candidates.add(candidate("B1", List.of(OptimizationStage.AR)));
        candidates.add(candidate("B2", List.of(OptimizationStage.AR)));
        candidates.add(candidate("B3", List.of(OptimizationStage.AR)));
        candidates.add(candidate("C1", List.of(OptimizationStage.FUSE)));
        candidates.add(candidate("C2", List.of(OptimizationStage.FUSE)));
        candidates.add(candidate("C3", List.of(OptimizationStage.FUSE)));

        GraphScoutConfig config = new GraphScoutConfig(
                2, 2, 2, 2,
                1, 3,
                1, 1,
                10, 10,
                0, 2,
                1, 1,
                1000, 0L,
                2.0,
                new BeamSearchConfig(1, 1, 1, 2, 2, 2, 2, 2)
        );
        CandidateEvalCache cache = new CandidateEvalCache("ctx", c -> c.name());

        List<OptimizerCandidate> reduced = GraphScoutReducer.reduceCandidates(
                candidates,
                config,
                cache,
                (candidate, warmup, measure, tier, evalCache) -> perf(candidate),
                OptimizerCandidate::name,
                msg -> {},
                System::nanoTime
        );

        assertEquals(4, reduced.size());
        assertEquals(2, reduced.stream().filter(c -> "CSE".equals(stageOrder(c))).count());
        assertEquals(2, reduced.stream().filter(c -> "AR".equals(stageOrder(c))).count());
        assertTrue(reduced.stream().noneMatch(c -> "FUSE".equals(stageOrder(c))));
    }

    @Test
    void reducerUsesPrescreenTierWhenFilteredSetExceedsKeepLimit() {
        List<OptimizerCandidate> candidates = List.of(
                candidate("A1", List.of(OptimizationStage.CSE)),
                candidate("A2", List.of(OptimizationStage.CSE)),
                candidate("B1", List.of(OptimizationStage.AR)),
                candidate("B2", List.of(OptimizationStage.AR))
        );
        GraphScoutConfig config = new GraphScoutConfig(
                1, 1, 1, 2,
                1, 3,
                2, 2,
                1, 1,
                0, 4,
                1, 1,
                1000, 0L,
                2.0,
                new BeamSearchConfig(1, 1, 1, 1, 1, 1, 1, 1)
        );
        CandidateEvalCache cache = new CandidateEvalCache("ctx", c -> c.name());
        AtomicInteger prescreenCalls = new AtomicInteger();

        GraphScoutReducer.reduceCandidates(
                candidates,
                config,
                cache,
                (candidate, warmup, measure, tier, evalCache) -> {
                    if ("PRESCREEN".equals(tier)) {
                        prescreenCalls.incrementAndGet();
                    }
                    return perf(candidate);
                },
                OptimizerCandidate::name,
                msg -> {},
                System::nanoTime
        );

        assertTrue(prescreenCalls.get() > 0);
    }

    private static CandidatePerf perf(OptimizerCandidate candidate) {
        String stage = stageOrder(candidate);
        double score;
        switch (stage) {
            case "CSE" -> score = 1.0;
            case "AR" -> score = 2.0;
            case "FUSE" -> score = 5.0;
            default -> score = 10.0;
        }
        return new CandidatePerf(
                candidate,
                stage,
                CoarseKnobSignature.of(candidate),
                10,
                20,
                score,
                score,
                0.1
        );
    }

    private static String stageOrder(OptimizerCandidate candidate) {
        if (candidate.stageOrder().isEmpty()) {
            return "NONE";
        }
        return candidate.stageOrder().get(0).name();
    }

    private static OptimizerCandidate candidate(String name, List<OptimizationStage> stages) {
        return new OptimizerCandidate(name, stages, TuningKnobs.trainingDefaults());
    }
}
