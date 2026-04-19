package tensor;

import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;

/**
 * Mutable convenience options for {@link Tensor#compute(ComputeOptions)}.
 */
public final class ComputeOptions {
    private CompileMode compileMode = CompileMode.INFERENCE_ONLY;
    private AutotunePolicy autotunePolicy = AutotunePolicy.NEVER;
    private OptimizerConfig optimizer;
    private RuntimeConfig runtime;

    public CompileMode compileMode() {
        return compileMode;
    }

    public ComputeOptions compileMode(CompileMode compileMode) {
        this.compileMode = compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode;
        return this;
    }

    public AutotunePolicy autotunePolicy() {
        return autotunePolicy;
    }

    public ComputeOptions autotune(AutotunePolicy autotunePolicy) {
        this.autotunePolicy = autotunePolicy == null ? AutotunePolicy.NEVER : autotunePolicy;
        return this;
    }

    public OptimizerConfig optimizer() {
        return optimizer;
    }

    public ComputeOptions optimizer(OptimizerConfig optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public RuntimeConfig runtime() {
        return runtime;
    }

    public ComputeOptions runtime(RuntimeConfig runtime) {
        this.runtime = runtime;
        return this;
    }
}
