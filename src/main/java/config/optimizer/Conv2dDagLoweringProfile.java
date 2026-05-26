package config.optimizer;

/**
 * Tunable thresholds for heuristic conv2d DAG lowering.
 *
 * <p>The defaults intentionally preserve the conservative built-in behavior. Calibration can later
 * replace this profile without changing the lowering rule itself.</p>
 *
 * @param pointwiseMinInChannels minimum input channels for 1x1 pointwise lowering
 * @param pointwiseMinOutChannels minimum output channels for 1x1 pointwise lowering
 * @param pointwiseMaxOutOverIn maximum {@code outChannels / inChannels} ratio for 1x1 lowering
 * @param pointwiseMinSpatial minimum {@code batch * outH * outW} for 1x1 lowering
 * @param standard3x3MinInChannels minimum input channels for 3x3 stride-1 pad-1 lowering
 * @param standard3x3MinOutChannels minimum output channels for 3x3 stride-1 pad-1 lowering
 * @param standard3x3MinSpatial minimum {@code batch * outH * outW} for 3x3 stride-1 pad-1 lowering
 */
public record Conv2dDagLoweringProfile(
        int pointwiseMinInChannels,
        int pointwiseMinOutChannels,
        double pointwiseMaxOutOverIn,
        long pointwiseMinSpatial,
        int standard3x3MinInChannels,
        int standard3x3MinOutChannels,
        long standard3x3MinSpatial
) {
    public Conv2dDagLoweringProfile {
        pointwiseMinInChannels = Math.max(1, pointwiseMinInChannels);
        pointwiseMinOutChannels = Math.max(1, pointwiseMinOutChannels);
        pointwiseMaxOutOverIn = positiveFinite(pointwiseMaxOutOverIn, 2.0d);
        pointwiseMinSpatial = Math.max(0L, pointwiseMinSpatial);
        standard3x3MinInChannels = Math.max(1, standard3x3MinInChannels);
        standard3x3MinOutChannels = Math.max(1, standard3x3MinOutChannels);
        standard3x3MinSpatial = Math.max(0L, standard3x3MinSpatial);
    }

    public static Conv2dDagLoweringProfile defaults() {
        return new Conv2dDagLoweringProfile(
                128,
                64,
                2.0d,
                256L,
                64,
                64,
                512L
        );
    }

    private static double positiveFinite(double value, double fallback) {
        return Double.isFinite(value) && value > 0.0d ? value : fallback;
    }
}
