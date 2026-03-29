package Benchmark.autotune;

public record BeamSearchConfig(
        int rounds,
        int seedTrain,
        int seedInference,
        int beamWidthTrain,
        int beamWidthInference,
        int keepTrain,
        int keepInference,
        int maxPerStage
) {
    public BeamSearchConfig {
        if (rounds <= 0 || seedTrain <= 0 || seedInference <= 0
                || beamWidthTrain <= 0 || beamWidthInference <= 0
                || keepTrain <= 0 || keepInference <= 0 || maxPerStage <= 0) {
            throw new IllegalArgumentException("All BeamSearchConfig values must be > 0");
        }
    }
}
