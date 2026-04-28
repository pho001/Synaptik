package config.profile;

import config.optimizer.OptimizerConfig;

import java.util.Objects;

/**
 * Graph-side policy extracted from an executable profile.
 *
 * <p>Graph autotune is intentionally scoped to this policy: it may choose optimizer settings for a
 * graph, but it must not change calibrated hardware/runtime parameters. This value object is the
 * boundary between graph decisions such as rewrite/CSE/fusion/memory planning and
 * {@link PlatformRuntimeProfile}, which owns runtime thresholds and backend dispatch settings.</p>
 *
 * @param optimizer optimizer configuration used when compiling a tensor graph
 */
public record GraphExecutionPolicy(
        OptimizerConfig optimizer
) {
    public GraphExecutionPolicy {
        optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
    }

    /**
     * Wraps an explicit optimizer configuration as a graph execution policy.
     *
     * @param optimizer optimizer configuration; must not be {@code null}
     * @return policy containing the supplied optimizer configuration
     */
    public static GraphExecutionPolicy of(OptimizerConfig optimizer) {
        return new GraphExecutionPolicy(optimizer);
    }

    /**
     * Extracts only the graph policy portion of a complete execution profile.
     *
     * @param profile source profile; must not be {@code null}
     * @return policy containing {@code profile.optimizer()}
     */
    public static GraphExecutionPolicy fromExecutionProfile(ExecutionProfile profile) {
        Objects.requireNonNull(profile, "profile cannot be null");
        return new GraphExecutionPolicy(profile.optimizer());
    }

    /**
     * Returns the default graph policy for training-capable forward/backward execution.
     *
     * @return policy backed by {@link OptimizerConfig#trainingDefaults()}
     */
    public static GraphExecutionPolicy trainingDefaults() {
        return new GraphExecutionPolicy(OptimizerConfig.trainingDefaults());
    }

    /**
     * Returns the default graph policy for forward-only inference execution.
     *
     * @return policy backed by {@link OptimizerConfig#inferenceDefaults()}
     */
    public static GraphExecutionPolicy inferenceDefaults() {
        return new GraphExecutionPolicy(OptimizerConfig.inferenceDefaults());
    }

    /**
     * Returns a graph policy that disables optimizer stages.
     *
     * <p>This is primarily useful as a benchmark baseline. It does not disable runtime vectorization,
     * parallelism, BLAS, or accelerators; use an appropriate {@link config.runtime.RuntimeConfig} for
     * those runtime decisions.</p>
     *
     * @return policy backed by {@link OptimizerConfig#noOptimization()}
     */
    public static GraphExecutionPolicy noOptimization() {
        return new GraphExecutionPolicy(OptimizerConfig.noOptimization());
    }
}
