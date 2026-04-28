package tuning.api;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.util.Objects;

/**
 * Fluent builder for {@link ExecutionProfile}.
 *
 * <p>The builder is only a readability layer over the immutable {@code ExecutionProfile} record. It
 * does not introduce a second profile model and does not mutate runtime or optimizer configuration
 * objects. Call {@link #build()} to create the record consumed by compile, prepare, benchmark, and
 * autotune flows.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * ExecutionProfile profile = Synaptik.tuning()
 *         .profile()
 *         .name("main-calibrated-runtime-f64")
 *         .candidate("calibrated-runtime")
 *         .dtype(DataType.FLOAT64)
 *         .mode().training()
 *         .optimizer().trainingDefaults()
 *         .runtime().fromPlatformProfile(calibratedRuntime)
 *         .build();
 * }</pre>
 *
 * <p>This builder is mutable and not thread-safe. Build a new instance for each profile.</p>
 */
public final class ExecutionProfileDsl {
    private String profileName = "default";
    private String candidateName;
    private DataType dataType;
    private ExecutionMode mode = ExecutionMode.FORWARD_BACKWARD;
    private OptimizerConfig optimizer;
    private RuntimeConfig runtime;
    private WorkloadProfile workload = WorkloadProfile.none();

    /**
     * Sets the profile namespace.
     *
     * @param profileName profile namespace; blank values are normalized by {@link ExecutionProfile}
     * @return this builder
     */
    public ExecutionProfileDsl name(String profileName) {
        this.profileName = profileName;
        return this;
    }

    /**
     * Sets the benchmark/autotune candidate name.
     *
     * @param candidateName candidate display name; blank values fall back to profile name
     * @return this builder
     */
    public ExecutionProfileDsl candidate(String candidateName) {
        this.candidateName = candidateName;
        return this;
    }

    /**
     * Sets the tensor dtype for the profile.
     *
     * @param dataType dtype used by workloads measured with this profile
     * @return this builder
     */
    public ExecutionProfileDsl dtype(DataType dataType) {
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        return this;
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
    public ExecutionProfileDsl mode(ExecutionMode mode) {
        this.mode = mode == null ? ExecutionMode.FORWARD_BACKWARD : mode;
        return this;
    }

    /**
     * Uses forward-only mode.
     *
     * @return this builder
     */
    public ExecutionProfileDsl forward() {
        return mode(ExecutionMode.FORWARD);
    }

    /**
     * Uses forward/backward mode.
     *
     * @return this builder
     */
    public ExecutionProfileDsl forwardBackward() {
        return mode(ExecutionMode.FORWARD_BACKWARD);
    }

    /**
     * Alias for {@link #forwardBackward()}.
     *
     * @return this builder
     */
    public ExecutionProfileDsl training() {
        return forwardBackward();
    }

    /**
     * Opens the grouped optimizer selector.
     *
     * @return optimizer selector
     */
    public Optimizers optimizer() {
        return new Optimizers(this);
    }

    /**
     * Uses an explicit optimizer config.
     *
     * @param optimizer optimizer config
     * @return this builder
     */
    public ExecutionProfileDsl optimizer(OptimizerConfig optimizer) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        return this;
    }

    /**
     * Opens the grouped runtime selector.
     *
     * @return runtime selector
     */
    public Runtimes runtime() {
        return new Runtimes(this);
    }

    /**
     * Uses an explicit runtime config.
     *
     * @param runtime runtime config
     * @return this builder
     */
    public ExecutionProfileDsl runtime(RuntimeConfig runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime cannot be null");
        return this;
    }

    /**
     * Adds an optional workload descriptor.
     *
     * @param workload workload profile; {@code null} becomes {@link WorkloadProfile#none()}
     * @return this builder
     */
    public ExecutionProfileDsl workload(WorkloadProfile workload) {
        this.workload = workload == null ? WorkloadProfile.none() : workload;
        return this;
    }

    /**
     * Builds an immutable execution profile.
     *
     * @return execution profile
     * @throws IllegalStateException if dtype, optimizer, or runtime has not been selected
     */
    public ExecutionProfile build() {
        if (dataType == null) {
            throw new IllegalStateException("ExecutionProfile dtype must be selected before build().");
        }
        if (optimizer == null) {
            throw new IllegalStateException("ExecutionProfile optimizer must be selected before build().");
        }
        if (runtime == null) {
            throw new IllegalStateException("ExecutionProfile runtime must be selected before build().");
        }
        return new ExecutionProfile(
                profileName,
                candidateName,
                dataType,
                mode,
                optimizer,
                runtime,
                workload
        );
    }

    /**
     * Alias for {@link #build()}.
     *
     * @return execution profile
     */
    public ExecutionProfile toExecutionProfile() {
        return build();
    }

    /**
     * Grouped mode selector used by {@link ExecutionProfileDsl#mode()}.
     */
    public static final class Modes {
        private final ExecutionProfileDsl parent;

        private Modes(ExecutionProfileDsl parent) {
            this.parent = parent;
        }

        /**
         * Selects forward-only mode.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl forward() {
            return parent.forward();
        }

        /**
         * Selects forward/backward mode.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl forwardBackward() {
            return parent.forwardBackward();
        }

        /**
         * Alias for forward/backward mode.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl training() {
            return parent.training();
        }
    }

    /**
     * Grouped optimizer selector used by {@link ExecutionProfileDsl#optimizer()}.
     */
    public static final class Optimizers {
        private final ExecutionProfileDsl parent;

        private Optimizers(ExecutionProfileDsl parent) {
            this.parent = parent;
        }

        /**
         * Uses no graph optimization.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl noOptimization() {
            return parent.optimizer(OptimizerConfig.noOptimization());
        }

        /**
         * Uses inference optimizer defaults.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl inferenceDefaults() {
            return parent.optimizer(OptimizerConfig.inferenceDefaults());
        }

        /**
         * Uses training optimizer defaults.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl trainingDefaults() {
            return parent.optimizer(OptimizerConfig.trainingDefaults());
        }

        /**
         * Uses an explicit optimizer config.
         *
         * @param optimizer optimizer config
         * @return parent builder
         */
        public ExecutionProfileDsl config(OptimizerConfig optimizer) {
            return parent.optimizer(optimizer);
        }
    }

    /**
     * Grouped runtime selector used by {@link ExecutionProfileDsl#runtime()}.
     */
    public static final class Runtimes {
        private final ExecutionProfileDsl parent;

        private Runtimes(ExecutionProfileDsl parent) {
            this.parent = parent;
        }

        /**
         * Uses no-optimization, no-vectorization, no-parallelism runtime defaults.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl noOptNoVecNoPar() {
            return parent.runtime(RuntimeConfig.noOptNoVecNoPar());
        }

        /**
         * Uses inference runtime defaults.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl inferenceDefaults() {
            return parent.runtime(RuntimeConfig.inferenceDefaults());
        }

        /**
         * Uses training runtime defaults.
         *
         * @return parent builder
         */
        public ExecutionProfileDsl trainingDefaults() {
            return parent.runtime(RuntimeConfig.trainingDefaults());
        }

        /**
         * Uses runtime settings converted from a calibrated platform profile.
         *
         * @param profile platform runtime profile
         * @return parent builder
         */
        public ExecutionProfileDsl fromPlatformProfile(PlatformRuntimeProfile profile) {
            Objects.requireNonNull(profile, "profile cannot be null");
            return parent.runtime(profile.toRuntimeConfig());
        }

        /**
         * Uses an explicit runtime config.
         *
         * @param runtime runtime config
         * @return parent builder
         */
        public ExecutionProfileDsl config(RuntimeConfig runtime) {
            return parent.runtime(runtime);
        }
    }
}
