package tensor;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tuning.tensor.TensorComputeProfileResolver;

import java.util.Objects;

final class TensorExecutionSupport {
    private TensorExecutionSupport() {
    }

    static ComputeBackend resolveBackend(ComputeBackend forcedBackend) {
        return forcedBackend != null ? forcedBackend : ComputeBackend.CPU;
    }

    static PreparedExecution prepare(Tensor tensor, ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return CompiledGraph.compile(tensor, profile.compile(), compileModeForProfile(profile)).prepare(profile.runtime());
    }

    static void compute(Tensor tensor, ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        compute(prepare(tensor, profile), profile.mode());
    }

    static CompiledGraph compile(Tensor tensor, CompileMode compileMode) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        CompileMode effectiveMode = compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode;
        return CompiledGraph.compile(tensor, TensorComputeProfileResolver.defaultCompile(tensor, effectiveMode), effectiveMode);
    }

    static Tensor compute(Tensor tensor) {
        return compute(tensor, CompileMode.INFERENCE_ONLY);
    }

    static Tensor compute(Tensor tensor, CompileMode compileMode) {
        return compute(tensor, new ComputeOptions().compileMode(compileMode));
    }

    static Tensor compute(Tensor tensor, ComputeOptions options) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        ComputeOptions effectiveOptions = options == null ? new ComputeOptions() : options;
        ExecutionProfile profile = TensorComputeProfileResolver.resolve(tensor, effectiveOptions);
        PreparedExecution prepared = prepare(tensor, profile);
        compute(prepared, profile.mode());
        return tensor;
    }

    static void compute(PreparedExecution execution, ExecutionMode mode) {
        if (execution == null) {
            throw new IllegalArgumentException("execution cannot be null");
        }
        execution.execute(mode);
    }

    private static CompileMode compileModeForProfile(ExecutionProfile profile) {
        return profile != null && profile.mode() == ExecutionMode.FORWARD_BACKWARD
                ? CompileMode.TRAINING
                : CompileMode.INFERENCE_ONLY;
    }
}
