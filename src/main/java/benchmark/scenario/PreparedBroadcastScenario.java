package benchmark.scenario;

import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import tensor.Tensor;

public final class PreparedBroadcastScenario {
    public final Tensor out;
    private final CompiledGraph compiledGraph;
    public final GraphOptimizer optimizer;
    private final RuntimeConfig runtimeConfig;

    public PreparedBroadcastScenario(Tensor output, CompiledGraph compiledGraph, GraphOptimizer optimizer, RuntimeConfig runtimeConfig) {
        this.out = output;
        this.compiledGraph = compiledGraph;
        this.optimizer = optimizer;
        this.runtimeConfig = runtimeConfig;
    }

    public Tensor output() {
        return out;
    }

    public GraphOptimizer optimizer() {
        return optimizer;
    }

    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    public void compute() {
        compiledGraph.execute(runtimeConfig, ExecutionMode.FORWARD);
    }
}
