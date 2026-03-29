package Benchmark.measure;

import Benchmark.OptimizationStage;
import Benchmark.OptimizerCandidate;

public final class CandidateMeasurementHarness {
    private final BenchmarkScenarioSource benchmarkScenarioSource;
    private final BroadcastScenarioSource broadcastScenarioSource;
    private final int graphBlocks;
    private final int fusedScoutExtraPrewarmIters;
    private final NanoClock clock;

    public CandidateMeasurementHarness(
            BenchmarkScenarioSource benchmarkScenarioSource,
            BroadcastScenarioSource broadcastScenarioSource,
            int graphBlocks,
            int fusedScoutExtraPrewarmIters,
            NanoClock clock
    ) {
        this.benchmarkScenarioSource = benchmarkScenarioSource;
        this.broadcastScenarioSource = broadcastScenarioSource;
        this.graphBlocks = graphBlocks;
        this.fusedScoutExtraPrewarmIters = fusedScoutExtraPrewarmIters;
        this.clock = clock == null ? NanoClock.SYSTEM : clock;
    }

    public CandidateMeasurementResult measure(
            OptimizerCandidate candidate,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC,
            int warmupIters,
            int measureIters,
            String tier,
            CandidateMeasurementCachePort cache
    ) {
        if (cache != null) {
            CandidateMeasurementResult cached = cache.get(candidate, tier, warmupIters, measureIters);
            if (cached != null) {
                return cached;
            }
        }

        MeasurementPolicy policy = TieredMeasurementPolicy.forTier(
                MeasurementTier.fromName(tier),
                candidate != null && candidate.stageOrder().contains(OptimizationStage.FUSE),
                warmupIters,
                measureIters,
                fusedScoutExtraPrewarmIters
        );

        MeasuredBenchmarkScenario forward = benchmarkScenarioSource.create(baseA, baseB, baseC, candidate, false, graphBlocks);
        int graphInfSize = forward.graphSize();
        forward.setTrainingMode(false);
        double forwardMs = MeasurementExecutor.measureAverageMs(policy, forward::compute, clock);

        MeasuredBenchmarkScenario training = benchmarkScenarioSource.create(baseA, baseB, baseC, candidate, true, graphBlocks);
        int graphTrnSize = training.graphSize();
        training.setTrainingMode(true);
        double trainMs = MeasurementExecutor.measureAverageMs(policy, training::compute, clock);

        MeasuredBroadcastScenario broadcast = broadcastScenarioSource.create(baseBroadcastA, baseBroadcastB, baseBroadcastC, candidate);
        double broadcastMs = MeasurementExecutor.measureAverageMs(policy, broadcast::compute, clock);

        CandidateMeasurementResult result = new CandidateMeasurementResult(
                candidate,
                graphInfSize,
                graphTrnSize,
                forwardMs,
                trainMs,
                broadcastMs
        );
        if (cache != null) {
            cache.put(candidate, tier, warmupIters, measureIters, result);
        }
        return result;
    }
}
