package config.backend;

/**
 * Numerical accumulation policy for sum/reduction kernels.
 *
 * <p>This setting can affect both performance and numerical error. It is therefore a semantic runtime
 * policy, not a pure hardware calibration knob.</p>
 */
public enum SumAccuracyMode {
    /**
     * Fast accumulation with minimal compensation.
     */
    FAST,
    /**
     * Kahan compensated summation.
     */
    KAHAN,
    /**
     * Neumaier compensated summation.
     */
    NEUMAIER
}
