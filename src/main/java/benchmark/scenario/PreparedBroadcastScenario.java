package benchmark.scenario;

import graph.optimizer.GraphOptimizer;
import tensor.Tensor;

public final class PreparedBroadcastScenario {
    public final Tensor out;
    public final GraphOptimizer optimizer;

    public PreparedBroadcastScenario(Tensor output, GraphOptimizer optimizer) {
        this.out = output;
        this.optimizer = optimizer;
    }

    public Tensor output() {
        return out;
    }

    public GraphOptimizer optimizer() {
        return optimizer;
    }

    public void compute() {
        out.compute(optimizer);
    }
}
