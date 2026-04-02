package benchmark.scenario;

import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import tensor.Tensor;

public final class PreparedBenchmarkScenario {
    public final Tensor A;
    public final Tensor B;
    public final Tensor C;
    public final Tensor ta7;
    private final CompiledGraph compiledGraph;
    public final GraphOptimizer optimizer;
    private final RuntimeConfig runtimeConfig;
    private ExecutionMode executionMode;

    public PreparedBenchmarkScenario(
            Tensor a,
            Tensor b,
            Tensor c,
            Tensor output,
            CompiledGraph compiledGraph,
            GraphOptimizer optimizer,
            RuntimeConfig runtimeConfig,
            boolean trainingMode
    ) {
        this.A = a;
        this.B = b;
        this.C = c;
        this.ta7 = output;
        this.compiledGraph = compiledGraph;
        this.optimizer = optimizer;
        this.runtimeConfig = runtimeConfig;
        this.executionMode = trainingMode ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
    }

    public Tensor a() {
        return A;
    }

    public Tensor b() {
        return B;
    }

    public Tensor c() {
        return C;
    }

    public Tensor output() {
        return ta7;
    }

    public GraphOptimizer optimizer() {
        return optimizer;
    }

    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    public void setTrainingMode(boolean trainingMode) {
        this.executionMode = trainingMode ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
    }

    public void compute() {
        compiledGraph.execute(runtimeConfig, executionMode);
    }
}
