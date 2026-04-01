import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.optimizer.GraphOptimizer;
import tensor.Tensor;

public final class TestGraphSupport {
    private TestGraphSupport() {
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer) {
        return rootTensor.compile(optimizer);
    }

    public static CompiledGraph execute(Tensor rootTensor, GraphOptimizer optimizer, RuntimeConfig runtimeConfig, ExecutionMode mode) {
        PreparedExecution execution = prepare(rootTensor, optimizer, runtimeConfig);
        execution.execute(mode);
        return rootTensor.getCompiledGraph();
    }

    public static CompiledGraph execute(Tensor rootTensor, GraphOptimizer optimizer) {
        PreparedExecution execution = prepare(rootTensor, optimizer, null);
        ExecutionMode mode = execution.supportsBackward() ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        execution.execute(mode);
        return rootTensor.getCompiledGraph();
    }

    public static PreparedExecution prepare(Tensor rootTensor, GraphOptimizer optimizer, RuntimeConfig runtimeConfig) {
        return rootTensor.prepare(optimizer, runtimeConfig);
    }
}
