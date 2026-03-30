package benchmark.autotune;

public record GraphScoutConfig(
        int stageScoutSamplePerRound,
        int stageScoutMaxSamplesPerStage,
        int stageScoutMaxRounds,
        int stageScoutMinActiveFamilies,
        int stageScoutWarmupIters,
        int stageScoutMeasureIters,
        int stageScoutTopTrain,
        int stageScoutTopInference,
        int prescreenKeepTrain,
        int prescreenKeepInference,
        int prescreenDiversitySeedsPerFamily,
        int prescreenMaxPerStageOrder,
        int prescreenWarmupIters,
        int prescreenMeasureIters,
        int progressLogEvery,
        long progressMinIntervalMs,
        double confidenceZ,
        BeamSearchConfig beamSearchConfig
) {
    public GraphScoutConfig {
        if (stageScoutSamplePerRound <= 0 || stageScoutMaxSamplesPerStage <= 0 || stageScoutMaxRounds <= 0
                || stageScoutMinActiveFamilies <= 0 || stageScoutMeasureIters <= 0
                || stageScoutTopTrain <= 0 || stageScoutTopInference <= 0
                || prescreenKeepTrain <= 0 || prescreenKeepInference <= 0 || prescreenMaxPerStageOrder <= 0
                || prescreenMeasureIters <= 0 || progressLogEvery <= 0 || confidenceZ < 0.0 || beamSearchConfig == null) {
            throw new IllegalArgumentException("Invalid GraphScoutConfig");
        }
        if (stageScoutWarmupIters < 0 || prescreenWarmupIters < 0 || prescreenDiversitySeedsPerFamily < 0 || progressMinIntervalMs < 0L) {
            throw new IllegalArgumentException("Invalid GraphScoutConfig");
        }
    }
}
