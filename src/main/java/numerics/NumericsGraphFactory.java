package numerics;

import tensor.Tensor;

final class NumericsGraphFactory {
    private NumericsGraphFactory() {
    }

    static Tensor buildOptimizerLikeGraph(
            Tensor a,
            Tensor b,
            Tensor c,
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
        Tensor x = a.mul(0.50).add(b.mul(0.30)).sub(c.mul(0.20));
        for (int i = 0; i < blocks; i++) {
            x = x.mul(0.70).add(b.mul(0.20));
            x = x.sub(c.mul(0.10));
            x = x.add(a.mul(0.05));
            x = x.mul(0.95).add(b.mul(0.03)).sub(c.mul(0.02));
        }
        Tensor linear1 = linearIn.linear(w1, b1);
        Tensor linear2 = linear1.linear(w2, b2);
        Tensor linear3 = linear2.linear(w3, b3);
        Tensor linearScalar = linear3.sum();
        return x.mul(x).add(b.mul(0.01)).add(linearScalar);
    }

    static Tensor buildBroadcastGraph(Tensor a, Tensor b, Tensor c) {
        return a.add(b).mul(c).add(a).sigmoid();
    }
}
