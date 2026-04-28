package tuning.api;

import tensor.DataType;

/**
 * Entry point for ergonomic calibration, benchmark, and future tuning workflows.
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
     * Starts calibration with one dtype already selected.
     *
     * @param dtype dtype to calibrate; must be supported by calibration
     * @return calibration builder
     */
    public CalibrationDsl calibrate(DataType dtype) {
        return calibration().dtype(dtype);
    }
}
