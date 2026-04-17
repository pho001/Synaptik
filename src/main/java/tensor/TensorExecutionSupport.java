package tensor;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;

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
        return CompiledGraph.compile(tensor, profile.optimizer()).prepare(profile.runtime());
    }

    static void compute(Tensor tensor, ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        compute(prepare(tensor, profile), profile.mode());
    }

    static void compute(PreparedExecution execution, ExecutionMode mode) {
        if (execution == null) {
            throw new IllegalArgumentException("execution cannot be null");
        }
        execution.execute(mode);
    }
}
