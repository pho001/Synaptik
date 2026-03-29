package Benchmark.scenario;

import Graph.optimizer.GraphOptimizer;
import Tensor.Tensor;

public final class PreparedBenchmarkScenario {
    public final Tensor A;
    public final Tensor B;
    public final Tensor C;
    public final Tensor ta7;
    public final GraphOptimizer optimizer;

    public PreparedBenchmarkScenario(Tensor a, Tensor b, Tensor c, Tensor output, GraphOptimizer optimizer) {
        this.A = a;
        this.B = b;
        this.C = c;
        this.ta7 = output;
        this.optimizer = optimizer;
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

    public void compute() {
        ta7.compute(optimizer);
    }
}
