import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutotuneSearchSupport;
import benchmark.autotune.BeamSearchConfig;
import benchmark.autotune.CandidateGraphIndex;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.FamilyScoutStats;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutotuneSearchSupportTest {

    @Test
    void stageOrderKeyAndNeighborsRespectMemLast() {
        String key = AutotuneSearchSupport.stageOrderKey(candidate("X", List.of(OptimizationStage.CSE, OptimizationStage.MEM)));
        assertEquals("CSE,MEM", key);

        List<String> neighbors = AutotuneSearchSupport.stageOrderNeighbors("CSE,MEM");
        assertTrue(neighbors.contains("MEM"));
        assertTrue(neighbors.contains("AR,CSE,MEM"));
        assertTrue(neighbors.contains("CSE,FUSE,MEM"));
        assertTrue(neighbors.stream().noneMatch(x -> x.contains(",MEM,AR")));
    }

    @Test
    void sampleCandidatesEvenlySelectsStableSpread() {
        List<OptimizerCandidate> candidates = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            candidates.add(candidate("C" + i, List.of()));
        }

        List<OptimizerCandidate> sampled = AutotuneSearchSupport.sampleCandidatesEvenly(candidates, 5);
        assertIterableEquals(
                List.of(candidates.get(0), candidates.get(2), candidates.get(5), candidates.get(7), candidates.get(9)),
                sampled
        );
    }

    @Test
    void capCandidatesPerStageOrderLimitsEachFamily() {
        List<OptimizerCandidate> candidates = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            candidates.add(candidate("A" + i, List.of(OptimizationStage.CSE)));
            candidates.add(candidate("B" + i, List.of(OptimizationStage.AR)));
        }

        List<OptimizerCandidate> capped = AutotuneSearchSupport.capCandidatesPerStageOrder(candidates, 2);
        long cse = capped.stream().filter(c -> "CSE".equals(AutotuneSearchSupport.stageOrderKey(c))).count();
        long ar = capped.stream().filter(c -> "AR".equals(AutotuneSearchSupport.stageOrderKey(c))).count();

        assertEquals(2, cse);
        assertEquals(2, ar);
    }

    @Test
    void pruneFamilyScoutRoundKeepsBestTrainingAndInferenceFamilies() {
        FamilyScoutStats a = family("A", 1.0, 10.0);
        FamilyScoutStats b = family("B", 10.0, 1.0);
        FamilyScoutStats c = family("C", 5.0, 5.0);

        List<FamilyScoutStats> survivors = AutotuneSearchSupport.pruneFamilyScoutRound(
                List.of(a, b, c),
                2,
                1,
                1
        );

        assertEquals(2, survivors.size());
        assertTrue(survivors.contains(a));
        assertTrue(survivors.contains(b));
    }

    @Test
    void selectCandidatesViaBeamExploresNeighboringCandidates() {
        OptimizerCandidate baseCandidate = candidate("BASE", List.of(OptimizationStage.CSE));
        OptimizerCandidate stageNeighborCandidate = candidate("STAGE", List.of(OptimizationStage.AR, OptimizationStage.CSE));
        OptimizerCandidate knobNeighborCandidate = candidate("KNOB", List.of(OptimizationStage.CSE), toggledStrictKnobs());

        CandidatePerf base = perf(baseCandidate, 0.5, 0.5);
        CandidatePerf stageNeighbor = perf(stageNeighborCandidate, 0.6, 0.6);
        CandidatePerf knobNeighbor = new CandidatePerf(
                knobNeighborCandidate,
                "CSE",
                CoarseKnobSignature.of(knobNeighborCandidate),
                10,
                20,
                0.7,
                0.7,
                0.1
        );

        List<CandidatePerf> prescreen = List.of(base, stageNeighbor, knobNeighbor);
        Set<OptimizerCandidate> seeds = new LinkedHashSet<>(List.of(baseCandidate));
        BeamSearchConfig config = new BeamSearchConfig(1, 1, 1, 2, 2, 2, 2, 2);

        List<OptimizerCandidate> out = AutotuneSearchSupport.selectCandidatesViaBeam(
                prescreen,
                seeds,
                config,
                AutotuneSearchSupport::stageOrderNeighbors,
                c -> c.name(),
                msg -> {}
        );

        assertTrue(out.contains(baseCandidate));
        assertTrue(out.contains(stageNeighborCandidate));
    }

    private static FamilyScoutStats family(String stageOrder, double trainingScore, double inferenceScore) {
        OptimizerCandidate c = candidate(stageOrder, List.of());
        FamilyScoutStats stats = new FamilyScoutStats(stageOrder, List.of(c), List.of(c), 2.0);
        stats.record(new CandidatePerf(c, stageOrder, CoarseKnobSignature.of(c), 10, 20, trainingScore, trainingScore, 0.1));
        stats.record(new CandidatePerf(c, stageOrder, CoarseKnobSignature.of(c), 10, 20, inferenceScore, inferenceScore, 0.1));
        return stats;
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, double forwardMs, double trainMs) {
        return new CandidatePerf(
                candidate,
                AutotuneSearchSupport.stageOrderKey(candidate),
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forwardMs,
                trainMs,
                0.1
        );
    }

    private static OptimizerCandidate candidate(String name, List<OptimizationStage> stages) {
        return candidate(name, stages, TuningKnobs.trainingDefaults());
    }

    private static OptimizerCandidate candidate(String name, List<OptimizationStage> stages, TuningKnobs knobs) {
        return new OptimizerCandidate(name, stages, knobs);
    }

    private static TuningKnobs toggledStrictKnobs() {
        TuningKnobs base = TuningKnobs.trainingDefaults();
        return new TuningKnobs(
                false,
                base.fuseConfig(),
                base.kernelConfig(),
                base.blasProvider(),
                base.blasMatMulMinWork(),
                base.blasF32RequireMgeK(),
                base.blasF32MaxNOverK()
        );
    }
}
