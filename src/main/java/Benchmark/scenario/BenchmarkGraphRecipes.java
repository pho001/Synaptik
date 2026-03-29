package Benchmark.scenario;

import Tensor.Tensor;

public final class BenchmarkGraphRecipes {
    private BenchmarkGraphRecipes() {}

    public static Tensor buildOptimizerBenchmarkGraph(
            Tensor A,
            Tensor B,
            Tensor C,
            Tensor linearIn,
            Tensor w1,
            Tensor b1,
            Tensor w2,
            Tensor b2,
            Tensor w3,
            Tensor b3,
            int graphBlocks
    ) {
        int blocks = Math.max(1, graphBlocks);
        Tensor x = A.mul(0.50).add(B.mul(0.30)).sub(C.mul(0.20));
        for (int i = 0; i < blocks; i++) {
            x = x.mul(0.70).add(B.mul(0.20));
            x = x.sub(C.mul(0.10));
            x = x.add(A.mul(0.05));
            x = x.mul(0.95).add(B.mul(0.03)).sub(C.mul(0.02));
        }
        Tensor linear1 = linearIn.matmul(w1).add(b1);
        Tensor linear2 = linear1.matmul(w2).add(b2);
        Tensor linear3 = linear2.matmul(w3).add(b3);
        Tensor linearScalar = linear3.sum();
        return x.mul(x).add(B.mul(0.01)).add(linearScalar);
    }

    public static Tensor buildBroadcastGraph(Tensor A, Tensor B, Tensor C) {
        return A.add(B).mul(C).add(A).sigmoid();
    }
}
