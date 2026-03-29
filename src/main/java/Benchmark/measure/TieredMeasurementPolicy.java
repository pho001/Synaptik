package Benchmark.measure;

public final class TieredMeasurementPolicy {
    private TieredMeasurementPolicy() {}

    public static MeasurementPolicy forTier(
            MeasurementTier tier,
            boolean hasFuseStage,
            int warmupIters,
            int measureIters,
            int fusedScoutExtraPrewarmIters
    ) {
        int extraPrewarm = 0;
        if (tier == MeasurementTier.SCOUT && hasFuseStage) {
            extraPrewarm = Math.max(0, fusedScoutExtraPrewarmIters);
        }
        return new MeasurementPolicy(extraPrewarm, warmupIters, measureIters);
    }
}
