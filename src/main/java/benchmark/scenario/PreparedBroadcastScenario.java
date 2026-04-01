package benchmark.scenario;

import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.optimizer.GraphOptimizer;
import tensor.Tensor;

public final class PreparedBroadcastScenario {
    public final Tensor out;
    public final GraphOptimizer optimizer;
    private final RuntimeConfig runtimeConfig;

    public PreparedBroadcastScenario(Tensor output, GraphOptimizer optimizer, RuntimeConfig runtimeConfig) {
        this.out = output;
        this.optimizer = optimizer;
        this.runtimeConfig = runtimeConfig;
    }

    public Tensor output() {
        return out;
    }

    public GraphOptimizer optimizer() {
        return optimizer;
    }

    public void compute() {
        out.compute(optimizer, runtimeConfig, ExecutionMode.FORWARD);
    }
}
