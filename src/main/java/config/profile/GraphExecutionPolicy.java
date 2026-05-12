package config.profile;

import config.compile.CompileConfig;

import java.util.Objects;

/**
 * Graph-side policy extracted from an executable profile.
 *
 * <p>Graph autotune is intentionally scoped to this policy: it may choose compile settings for a
 * graph, but it must not change calibrated hardware/runtime parameters. This value object is the
 * boundary between graph decisions such as rewrite/CSE/fusion/memory planning and
 * {@link PlatformRuntimeProfile}, which owns runtime thresholds and backend dispatch settings.</p>
 *
 * @param compile compile configuration used when compiling a tensor graph
 */
public record GraphExecutionPolicy(
        CompileConfig compile
) {
    public GraphExecutionPolicy {
        compile = Objects.requireNonNull(compile, "compile cannot be null");
    }

    /**
     * Wraps an explicit compile configuration as a graph execution policy.
     *
     * @param compile compile configuration; must not be {@code null}
     * @return policy containing the supplied compile configuration
     */
    public static GraphExecutionPolicy of(CompileConfig compile) {
        return new GraphExecutionPolicy(compile);
    }

    /**
     * Extracts only the graph policy portion of a complete execution profile.
     *
     * @param profile source profile; must not be {@code null}
     * @return policy containing {@code profile.compile()}
     */
    public static GraphExecutionPolicy fromExecutionProfile(ExecutionProfile profile) {
        Objects.requireNonNull(profile, "profile cannot be null");
        return new GraphExecutionPolicy(profile.compile());
    }

    /**
     * Returns the default graph policy for training-capable forward/backward execution.
     *
     * @return policy backed by {@link CompileConfig#training()}
     */
    public static GraphExecutionPolicy trainingDefaults() {
        return new GraphExecutionPolicy(CompileConfig.training());
    }

    /**
     * Returns the default graph policy for forward-only inference execution.
     *
     * @return policy backed by {@link CompileConfig#inference()}
     */
    public static GraphExecutionPolicy inferenceDefaults() {
        return new GraphExecutionPolicy(CompileConfig.inference());
    }

    /**
     * Returns a graph policy that disables graph optimization.
     *
     * <p>This is primarily useful as a benchmark baseline. It does not disable runtime vectorization,
     * parallelism, BLAS, or accelerators; use an appropriate {@link config.runtime.RuntimeConfig} for
     * those runtime decisions.</p>
     *
     * @return policy backed by {@link CompileConfig#noGraphOptimizationBaseline()}
     */
    public static GraphExecutionPolicy noGraphOptimization() {
        return new GraphExecutionPolicy(CompileConfig.noGraphOptimizationBaseline());
    }
}
