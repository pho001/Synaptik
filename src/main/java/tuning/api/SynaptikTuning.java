package tuning.api;

import tensor.DataType;

/**
 * Entry point for ergonomic calibration, benchmark, profile-building, and future tuning workflows.
 *
 * <p>Instances are lightweight and stateless. Each workflow method returns a fresh mutable builder;
 * builders are intended for single-threaded request assembly and one execution.</p>
 */
public final class SynaptikTuning {
    private SynaptikTuning() {
    }

    /**
     * Creates a tuning facade.
     *
     * @return tuning facade
     */
    public static SynaptikTuning create() {
        return new SynaptikTuning();
    }

    /**
     * Starts a platform calibration workflow.
     *
     * @return calibration builder
     */
    public CalibrationDsl calibration() {
        return new CalibrationDsl();
    }

    /**
     * Starts a benchmark workflow.
     *
     * @return benchmark builder
     */
    public BenchmarkDsl benchmark() {
        return new BenchmarkDsl();
    }

    /**
     * Starts an execution-profile builder.
     *
     * <p>The returned builder creates immutable {@link config.profile.ExecutionProfile} records used
     * by benchmark, autotune, compile, prepare, and explicit tensor compute flows. It is intended for
     * readable Java configuration such as {@code Synaptik.tuning().profile().dtype(...).runtime()...}
     * instead of repeated long record constructors.</p>
     *
     * @return execution profile builder
     */
    public ExecutionProfileDsl profile() {
        return new ExecutionProfileDsl();
    }

    /**
     * Starts calibration with one dtype already selected.
     *
     * @param dtype dtype to calibrate; must be supported by calibration
     * @return calibration builder
     */
    public CalibrationDsl calibrate(DataType dtype) {
        return calibration().dtype(dtype);
    }
}
