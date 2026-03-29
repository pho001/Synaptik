import Benchmark.OptimizationStage;
import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;
import Benchmark.autotune.CandidateEvalCache;
import Benchmark.autotune.CandidateGraphIndex;
import Benchmark.autotune.CandidatePerf;
import Benchmark.autotune.CoarseKnobSignature;
import Benchmark.autotune.FamilyScoutStats;
import Benchmark.autotune.RunningEstimate;
import Benchmark.measure.CandidateMeasurementResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutotuneSupportModelsTest {

    @Test
    void coarseKnobSignatureDistanceDetectsSingleGroupDifference() {
        TuningKnobs baseKnobs = TuningKnobs.trainingDefaults();
        TuningKnobs strictFlipKnobs = new TuningKnobs(
                false,
                baseKnobs.fuseConfig(),
                baseKnobs.kernelConfig(),
                baseKnobs.blasProvider(),
                baseKnobs.blasMatMulMinWork(),
                baseKnobs.blasF32RequireMgeK(),
                baseKnobs.blasF32MaxNOverK()
        );
        OptimizerCandidate a = new OptimizerCandidate("A", List.of(OptimizationStage.AR), baseKnobs);
        OptimizerCandidate b = new OptimizerCandidate("B", List.of(OptimizationStage.AR), strictFlipKnobs);

        CoarseKnobSignature sa = CoarseKnobSignature.of(a);
        CoarseKnobSignature sb = CoarseKnobSignature.of(b);

        assertEquals(1, sa.distance(sb));
    }

    @Test
    void runningEstimateComputesFiniteBandAfterTwoSamples() {
        RunningEstimate estimate = new RunningEstimate(2.0);
        estimate.add(1.0);
        estimate.add(3.0);

        assertEquals(2, estimate.count());
        assertEquals(2.0, estimate.mean(), 1e-12);
        assertTrue(Double.isFinite(estimate.optimistic()));
        assertTrue(Double.isFinite(estimate.conservative()));
        assertTrue(estimate.optimistic() <= estimate.mean());
        assertTrue(estimate.conservative() >= estimate.mean());
    }

    @Test
    void familyScoutStatsConsumesSamplesAndRecordsScores() {
        OptimizerCandidate c1 = candidate("C1", List.of(OptimizationStage.CSE));
        OptimizerCandidate c2 = candidate("C2", List.of(OptimizationStage.CSE));
        FamilyScoutStats stats = new FamilyScoutStats(
                "CSE",
                List.of(c1, c2),
                List.of(c1, c2),
                2.0
        );

        assertTrue(stats.hasRemainingSamples());
        assertEquals(c1, stats.nextSample());
        stats.record(perf(c1, "CSE", 1.0, 2.0));
        assertEquals(1, stats.samples());
        assertTrue(stats.hasRemainingSamples());
        assertEquals(c2, stats.nextSample());
        stats.record(perf(c2, "CSE", 2.0, 3.0));
        assertEquals(2, stats.samples());
        assertTrue(stats.trainingMean() > 0.0);
        assertTrue(stats.inferenceMean() > 0.0);
    }

    @Test
    void candidateEvalCacheRoundTripsMeasurementResult() {
        CandidateEvalCache cache = new CandidateEvalCache(
                "dtype=FLOAT32|graphBlocks=6",
                c -> "fp:" + c.name()
        );
        OptimizerCandidate candidate = candidate("CACHE", List.of(OptimizationStage.AR));
        CandidateMeasurementResult expected = new CandidateMeasurementResult(candidate, 7, 11, 1.5, 2.5, 3.5);

        cache.put(candidate, "SCOUT", 1, 3, expected);
        CandidateMeasurementResult actual = cache.get(candidate, "SCOUT", 1, 3);

        assertNotNull(actual);
        assertEquals(expected.graphInfSize(), actual.graphInfSize());
        assertEquals(expected.graphTrnSize(), actual.graphTrnSize());
        assertEquals(expected.forwardMs(), actual.forwardMs(), 1e-12);
    }

    @Test
    void candidateGraphIndexFindsStageAndKnobNeighbors() {
        TuningKnobs baseKnobs = TuningKnobs.trainingDefaults();
        TuningKnobs strictFlipKnobs = new TuningKnobs(
                false,
                baseKnobs.fuseConfig(),
                baseKnobs.kernelConfig(),
                baseKnobs.blasProvider(),
                baseKnobs.blasMatMulMinWork(),
                baseKnobs.blasF32RequireMgeK(),
                baseKnobs.blasF32MaxNOverK()
        );
        OptimizerCandidate baseCandidate = candidate("BASE", List.of(OptimizationStage.CSE), baseKnobs);
        OptimizerCandidate stageNeighborCandidate = candidate("STAGE", List.of(OptimizationStage.AR, OptimizationStage.CSE));
        OptimizerCandidate knobNeighborCandidate = candidate("KNOB", List.of(OptimizationStage.CSE), strictFlipKnobs);

        CandidatePerf base = perf(baseCandidate, "CSE", 1.0, 2.0);
        CandidatePerf stageNeighbor = perf(stageNeighborCandidate, "AR,CSE", 1.1, 2.1);
        CandidatePerf knobNeighbor = new CandidatePerf(
                knobNeighborCandidate,
                "CSE",
                CoarseKnobSignature.of(knobNeighborCandidate),
                10,
                20,
                1.2,
                2.2,
                3.2
        );

        CandidateGraphIndex index = new CandidateGraphIndex(
                List.of(base, stageNeighbor, knobNeighbor),
                stageOrder -> "CSE".equals(stageOrder) ? List.of("AR,CSE") : List.of()
        );

        List<CandidatePerf> neighbors = index.neighbors(base);
        assertEquals(2, neighbors.size());
    }

    private static OptimizerCandidate candidate(String name, List<OptimizationStage> stageOrder) {
        return candidate(name, stageOrder, TuningKnobs.trainingDefaults());
    }

    private static OptimizerCandidate candidate(String name, List<OptimizationStage> stageOrder, TuningKnobs knobs) {
        return new OptimizerCandidate(name, stageOrder, knobs);
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, String stageOrderKey, double forwardMs, double trainMs) {
        return new CandidatePerf(
                candidate,
                stageOrderKey,
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forwardMs,
                trainMs,
                3.0
        );
    }
}
