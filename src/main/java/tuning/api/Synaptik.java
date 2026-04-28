package tuning.api;

/**
 * Public facade for ergonomic Synaptik framework workflows.
 *
 * <p>The facade intentionally exposes workflow builders rather than owning tuning algorithms. Each
 * builder delegates to the existing low-level request and session types in {@code tuning.*}, so
 * fluent calls and direct request construction use the same execution, validation, reporting, and
 * persistence paths.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * Synaptik.tuning()
 *         .calibration()
 *         .dtype(DataType.FLOAT64)
 *         .allFamilies()
 *         .quick()
 *         .run();
 * }</pre>
 */
public final class Synaptik {
    private Synaptik() {
    }

    /**
     * Starts the ergonomic tuning API.
     *
     * @return a new tuning facade instance
     */
    public static SynaptikTuning tuning() {
        return SynaptikTuning.create();
    }
}
