package tensor;

import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;

/**
 * Mutable convenience options for {@link Tensor#compute(ComputeOptions)}.
 *
 * <p>The fluent setters mutate and return this instance. Instances are intended
 * to be configured on one thread and then passed to a compute call; concurrent
 * mutation is not synchronized. Null compile/autotune values reset to the
 * documented defaults, while null optimizer/runtime values mean "use the
 * execution support defaults".</p>
 */
public final class ComputeOptions {
    private CompileMode compileMode = CompileMode.INFERENCE_ONLY;
    private AutotunePolicy autotunePolicy = AutotunePolicy.NEVER;
    private OptimizerConfig optimizer;
    private RuntimeConfig runtime;

    /**
     * Returns the compile behavior requested for the compute call.
     *
     * @return non-null compile mode; defaults to {@link CompileMode#INFERENCE_ONLY}
     */
    public CompileMode compileMode() {
        return compileMode;
    }

    /**
     * Chooses whether the compute call should compile only the forward graph or
     * include training metadata.
     *
     * <p>This method mutates the current options object and is intended for
     * fluent setup immediately before calling {@link Tensor#compute(ComputeOptions)}.
     * Passing null restores the default inference-only mode.</p>
     *
     * @param compileMode compile mode to use, or null to reset to
     *                    {@link CompileMode#INFERENCE_ONLY}
     * @return this mutable options instance
     */
    public ComputeOptions compileMode(CompileMode compileMode) {
        this.compileMode = compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode;
        return this;
    }

    /**
     * Returns the autotuning policy used before execution.
     *
     * @return non-null autotune policy; defaults to {@link AutotunePolicy#NEVER}
     */
    public AutotunePolicy autotunePolicy() {
        return autotunePolicy;
    }

    /**
     * Chooses whether graph autotuning may run before the compute call executes.
     *
     * <p>The policy affects preparation cost and may consult tuning history, but
     * it does not change the tensor graph being compiled. Passing null restores
     * the default policy, which skips autotuning.</p>
     *
     * @param autotunePolicy policy to use, or null to reset to
     *                       {@link AutotunePolicy#NEVER}
     * @return this mutable options instance
     */
    public ComputeOptions autotune(AutotunePolicy autotunePolicy) {
        this.autotunePolicy = autotunePolicy == null ? AutotunePolicy.NEVER : autotunePolicy;
        return this;
    }

    /**
     * Returns the optimizer configuration override.
     *
     * @return optimizer config, or null to use the default optimizer profile
     */
    public OptimizerConfig optimizer() {
        return optimizer;
    }

    /**
     * Provides an optimizer configuration override for this compute call.
     *
     * <p>When null, the compute path selects the standard optimizer profile for
     * the requested compile mode. Supplying a config lets callers enable, disable,
     * or tune graph rewrite, CSE, partition, fusion, and memory-planning stages
     * for this one invocation.</p>
     *
     * @param optimizer optimizer config, or null to use the default optimizer
     * @return this mutable options instance
     */
    public ComputeOptions optimizer(OptimizerConfig optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    /**
     * Returns the runtime configuration override.
     *
     * @return runtime config, or null to use the default runtime profile
     */
    public RuntimeConfig runtime() {
        return runtime;
    }

    /**
     * Provides a runtime configuration override for backend execution.
     *
     * <p>The runtime config controls backend availability and execution policies
     * such as fused execution, accelerator use, BLAS use, approximation policy,
     * and convolution dispatch. Passing null delegates those choices to the
     * framework default runtime profile.</p>
     *
     * @param runtime runtime config, or null to use the default runtime
     * @return this mutable options instance
     */
    public ComputeOptions runtime(RuntimeConfig runtime) {
        this.runtime = runtime;
        return this;
    }
}
