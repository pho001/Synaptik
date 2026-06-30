package tuning.api;

import runtime.contract.ExecutionMode;
import tensor.DataType;
import tuning.calibration.PlatformCalibrationResult;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationRunner;
import tuning.calibration.run.CalibrationScope;
import tuning.measure.MeasurementPolicy;
import tuning.preset.TuningPreset;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Fluent builder for platform runtime calibration.
 *
 * <p>The builder is a convenience layer over {@link CalibrationCommand} and {@link CalibrationRunner}.
 * It does not contain calibration algorithms. Calling {@link #toCommand()} returns the exact command
 * that {@link #run()} will pass to {@code CalibrationRunner.create().run(command)}.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * List<PlatformCalibrationResult> results = Synaptik.tuning()
 *         .calibration()
 *         .dtype(DataType.FLOAT64)
 *         .allFamilies()
 *         .quick()
 *         .training()
 *         .measurement(1, 3, 1)
 *         .progress().lines()
 *         .colorAuto()
 *         .outputRoot(Path.of("profiles"))
 *         .run();
 * }</pre>
 *
 * <p>This builder is mutable and not thread-safe. Build a new instance for each calibration run.</p>
 */
public final class CalibrationDsl {
    private static final Path DEFAULT_OUTPUT_ROOT = Path.of("profiles");
    private static final List<DataType> SUPPORTED_DTYPES = List.of(
            DataType.FLOAT64,
            DataType.FLOAT32,
            DataType.BFLOAT16
    );

    private List<DataType> dataTypes = List.of(DataType.FLOAT64);
    private CalibrationFamilyId family;
    private CalibrationScope scope = CalibrationScope.ALL_FAMILIES;
    private TuningPreset preset = TuningPreset.BALANCED;
    private ExecutionMode mode = ExecutionMode.FORWARD_BACKWARD;
    private MeasurementPolicy measurement;
    private String colorMode = "auto";
    private String progressMode = "live";
    private Path outputRoot = DEFAULT_OUTPUT_ROOT;
    private boolean includeAccelerators;
    private CalibrationRunner runner;

    /**
     * Opens the grouped dtype selector for dot-style calls such as
     * {@code calibration().dtypes().single(DataType.FLOAT64)}.
     *
     * @return dtype selector
     */
    public DTypes dtypes() {
        return new DTypes(this);
    }

    /**
     * Selects one dtype for calibration.
     *
     * @param dtype dtype to calibrate; valid values are {@code FLOAT64}, {@code FLOAT32}, and
     *              {@code BFLOAT16}
     * @return this builder
     * @throws IllegalArgumentException if dtype is unsupported
     */
    public CalibrationDsl dtype(DataType dtype) {
        validateDType(dtype);
        dataTypes = List.of(dtype);
        return this;
    }

    /**
     * Selects one dtype using CLI-compatible aliases.
     *
     * @param alias one of {@code f64}, {@code f32}, or {@code bf16}
     * @return this builder
     * @throws IllegalArgumentException if alias is unknown
     */
    public CalibrationDsl dtype(String alias) {
        return dtype(parseDType(alias));
    }

    /**
     * Selects all calibration-supported floating point dtypes.
     *
     * @return this builder
     */
    public CalibrationDsl allDTypes() {
        dataTypes = SUPPORTED_DTYPES;
        return this;
    }

    /**
     * Opens the grouped family selector for dot-style calls such as
     * {@code calibration().families().all()}.
     *
     * @return family selector
     */
    public Families families() {
        return new Families(this);
    }

    /**
     * Selects one calibration family.
     *
     * @param family family id from {@link CalibrationFamilyRegistry}
     * @return this builder
     */
    public CalibrationDsl family(CalibrationFamilyId family) {
        this.family = Objects.requireNonNull(family, "family cannot be null");
        scope = CalibrationScope.SINGLE_FAMILY;
        return this;
    }

    /**
     * Selects one calibration family by CLI name.
     *
     * @param familyName family CLI name such as {@code matmul} or {@code scheduler}
     * @return this builder
     * @throws IllegalArgumentException if family name is unknown
     */
    public CalibrationDsl family(String familyName) {
        return family(CalibrationFamilyRegistry.parse(familyName));
    }

    /**
     * Selects the full non-accelerator family suite unless {@link #includeAccelerators()} is also set.
     *
     * @return this builder
     */
    public CalibrationDsl allFamilies() {
        family = null;
        scope = CalibrationScope.ALL_FAMILIES;
        return this;
    }

    /**
     * Sets the preset used for calibration planning and default measurement policy.
     *
     * @param preset preset; {@code null} falls back to {@code BALANCED}
     * @return this builder
     */
    public CalibrationDsl preset(TuningPreset preset) {
        this.preset = preset == null ? TuningPreset.BALANCED : preset;
        return this;
    }

    /**
     * Uses the quick preset.
     *
     * @return this builder
     */
    public CalibrationDsl quick() {
        return preset(TuningPreset.QUICK);
    }

    /**
     * Uses the balanced preset.
     *
     * @return this builder
     */
    public CalibrationDsl balanced() {
        return preset(TuningPreset.BALANCED);
    }

    /**
     * Uses the thorough preset.
     *
     * @return this builder
     */
    public CalibrationDsl thorough() {
        return preset(TuningPreset.THOROUGH);
    }

    /**
     * Opens the grouped execution-mode selector.
     *
     * @return mode selector
     */
    public Modes mode() {
        return new Modes(this);
    }

    /**
     * Sets the execution mode.
     *
     * @param mode execution mode; {@code null} falls back to {@code FORWARD_BACKWARD}
     * @return this builder
     */
    public CalibrationDsl mode(ExecutionMode mode) {
        this.mode = mode == null ? ExecutionMode.FORWARD_BACKWARD : mode;
        return this;
    }

    /**
     * Uses forward-only calibration mode.
     *
     * @return this builder
     */
    public CalibrationDsl forward() {
        return mode(ExecutionMode.FORWARD);
    }

    /**
     * Uses forward/backward calibration mode.
     *
     * @return this builder
     */
    public CalibrationDsl forwardBackward() {
        return mode(ExecutionMode.FORWARD_BACKWARD);
    }

    /**
     * Alias for {@link #forwardBackward()}.
     *
     * @return this builder
     */
    public CalibrationDsl training() {
        return forwardBackward();
    }

    /**
     * Opens the grouped measurement selector.
     *
     * @return measurement selector
     */
    public Measurements measurement() {
        return new Measurements(this);
    }

    /**
     * Overrides measurement loop counts while preserving preset measurement flags.
     *
     * @param warmupIters unreported warmup iterations; must be non-negative
     * @param measureIters measured iterations per repeat; must be at least one
     * @param repeats number of timing repeats; must be at least one
     * @return this builder
     */
    public CalibrationDsl measurement(int warmupIters, int measureIters, int repeats) {
        MeasurementPolicy base = preset.benchmarkMeasurement();
        measurement = new MeasurementPolicy(
                warmupIters,
                measureIters,
                repeats,
                base.measureCompile(),
                base.measurePrepare(),
                base.measureColdRun(),
                base.measureSteadyState(),
                base.captureStepTrace()
        );
        return this;
    }

    /**
     * Uses an explicit measurement policy.
     *
     * @param policy measurement policy; {@code null} means the runner will use preset defaults
     * @return this builder
     */
    public CalibrationDsl measurement(MeasurementPolicy policy) {
        measurement = policy;
        return this;
    }

    /**
     * Opens the grouped progress selector.
     *
     * @return progress selector
     */
    public Progress progress() {
        return new Progress(this);
    }

    /**
     * Uses the live terminal progress renderer.
     *
     * @return this builder
     */
    public CalibrationDsl progressLive() {
        progressMode = "live";
        return this;
    }

    /**
     * Uses line-oriented progress output.
     *
     * @return this builder
     */
    public CalibrationDsl progressLines() {
        progressMode = "lines";
        return this;
    }

    /**
     * Disables calibration progress output.
     *
     * @return this builder
     */
    public CalibrationDsl progressQuiet() {
        progressMode = "quiet";
        return this;
    }

    /**
     * Opens the grouped color selector.
     *
     * @return color selector
     */
    public Colors color() {
        return new Colors(this);
    }

    /**
     * Uses automatic color detection.
     *
     * @return this builder
     */
    public CalibrationDsl colorAuto() {
        colorMode = "auto";
        return this;
    }

    /**
     * Forces ANSI colors.
     *
     * @return this builder
     */
    public CalibrationDsl colorAlways() {
        colorMode = "always";
        return this;
    }

    /**
     * Disables ANSI colors.
     *
     * @return this builder
     */
    public CalibrationDsl colorNever() {
        colorMode = "never";
        return this;
    }

    /**
     * Sets the profile artifact root.
     *
     * @param outputRoot root directory for calibration artifacts; {@code null} uses {@code profiles}
     * @return this builder
     */
    public CalibrationDsl outputRoot(Path outputRoot) {
        this.outputRoot = outputRoot == null ? DEFAULT_OUTPUT_ROOT : outputRoot;
        return this;
    }

    /**
     * Includes opt-in accelerator calibration families in all-family runs.
     *
     * @return this builder
     */
    public CalibrationDsl includeAccelerators() {
        includeAccelerators = true;
        return this;
    }

    /**
     * Uses a custom runner. This is primarily useful for tests.
     *
     * @param runner runner to execute with; {@code null} restores the default runner
     * @return this builder
     */
    public CalibrationDsl runner(CalibrationRunner runner) {
        this.runner = runner;
        return this;
    }

    /**
     * Builds the low-level command represented by this fluent configuration.
     *
     * @return calibration command
     */
    public CalibrationCommand toCommand() {
        return new CalibrationCommand(
                dataTypes,
                family,
                scope,
                preset,
                mode,
                measurement,
                colorMode,
                progressMode,
                outputRoot,
                includeAccelerators
        );
    }

    /**
     * Runs calibration using the existing {@link CalibrationRunner}.
     *
     * @return calibration results, one result per executed family/pass segment
     */
    public List<PlatformCalibrationResult> run() {
        CalibrationRunner resolvedRunner = runner == null ? CalibrationRunner.create() : runner;
        return resolvedRunner.run(toCommand());
    }

    private static void validateDType(DataType dtype) {
        if (!SUPPORTED_DTYPES.contains(dtype)) {
            throw new IllegalArgumentException("Calibration supports only FLOAT64, FLOAT32, and BFLOAT16: " + dtype);
        }
    }

    private static DataType parseDType(String alias) {
        if (alias == null) {
            throw new IllegalArgumentException("dtype alias cannot be null");
        }
        return switch (alias.trim().toLowerCase(Locale.ROOT)) {
            case "f64" -> DataType.FLOAT64;
            case "f32" -> DataType.FLOAT32;
            case "bf16" -> DataType.BFLOAT16;
            default -> throw new IllegalArgumentException("Unknown calibration dtype: " + alias);
        };
    }

    /**
     * Grouped dtype selector used by {@link CalibrationDsl#dtypes()}.
     */
    public static final class DTypes {
        private final CalibrationDsl parent;

        private DTypes(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Selects one dtype.
         *
         * @param dtype dtype to calibrate
         * @return parent builder
         */
        public CalibrationDsl single(DataType dtype) {
            return parent.dtype(dtype);
        }

        /**
         * Selects one dtype by CLI-compatible alias.
         *
         * @param alias one of {@code f64}, {@code f32}, or {@code bf16}
         * @return parent builder
         */
        public CalibrationDsl single(String alias) {
            return parent.dtype(alias);
        }

        /**
         * Selects all calibration-supported dtypes.
         *
         * @return parent builder
         */
        public CalibrationDsl all() {
            return parent.allDTypes();
        }
    }

    /**
     * Grouped family selector used by {@link CalibrationDsl#families()}.
     */
    public static final class Families {
        private final CalibrationDsl parent;

        private Families(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Selects one family by id.
         *
         * @param family family id
         * @return parent builder
         */
        public CalibrationDsl single(CalibrationFamilyId family) {
            return parent.family(family);
        }

        /**
         * Selects one family by CLI name.
         *
         * @param familyName family CLI name
         * @return parent builder
         */
        public CalibrationDsl single(String familyName) {
            return parent.family(familyName);
        }

        /**
         * Selects all standard families.
         *
         * @return parent builder
         */
        public CalibrationDsl all() {
            return parent.allFamilies();
        }
    }

    /**
     * Grouped mode selector used by {@link CalibrationDsl#mode()}.
     */
    public static final class Modes {
        private final CalibrationDsl parent;

        private Modes(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Selects forward-only mode.
         *
         * @return parent builder
         */
        public CalibrationDsl forward() {
            return parent.forward();
        }

        /**
         * Selects forward/backward mode.
         *
         * @return parent builder
         */
        public CalibrationDsl forwardBackward() {
            return parent.forwardBackward();
        }

        /**
         * Alias for forward/backward mode.
         *
         * @return parent builder
         */
        public CalibrationDsl training() {
            return parent.training();
        }
    }

    /**
     * Grouped measurement selector used by {@link CalibrationDsl#measurement()}.
     */
    public static final class Measurements {
        private final CalibrationDsl parent;

        private Measurements(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Overrides loop counts while preserving preset measurement flags.
         *
         * @param warmupIters warmup iterations
         * @param measureIters measured iterations
         * @param repeats repeats
         * @return parent builder
         */
        public CalibrationDsl iterations(int warmupIters, int measureIters, int repeats) {
            return parent.measurement(warmupIters, measureIters, repeats);
        }

        /**
         * Uses an explicit measurement policy.
         *
         * @param policy policy to use
         * @return parent builder
         */
        public CalibrationDsl policy(MeasurementPolicy policy) {
            return parent.measurement(policy);
        }
    }

    /**
     * Grouped progress selector used by {@link CalibrationDsl#progress()}.
     */
    public static final class Progress {
        private final CalibrationDsl parent;

        private Progress(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Uses live progress rendering.
         *
         * @return parent builder
         */
        public CalibrationDsl live() {
            return parent.progressLive();
        }

        /**
         * Uses line-oriented progress rendering.
         *
         * @return parent builder
         */
        public CalibrationDsl lines() {
            return parent.progressLines();
        }

        /**
         * Disables progress rendering.
         *
         * @return parent builder
         */
        public CalibrationDsl quiet() {
            return parent.progressQuiet();
        }
    }

    /**
     * Grouped color selector used by {@link CalibrationDsl#color()}.
     */
    public static final class Colors {
        private final CalibrationDsl parent;

        private Colors(CalibrationDsl parent) {
            this.parent = parent;
        }

        /**
         * Uses automatic color behavior.
         *
         * @return parent builder
         */
        public CalibrationDsl auto() {
            return parent.colorAuto();
        }

        /**
         * Forces ANSI colors.
         *
         * @return parent builder
         */
        public CalibrationDsl always() {
            return parent.colorAlways();
        }

        /**
         * Disables ANSI colors.
         *
         * @return parent builder
         */
        public CalibrationDsl never() {
            return parent.colorNever();
        }
    }
}
