package graph.compile.planning.memory;

import tensor.Tensor;

import java.util.Map;

record TensorLifetimePlan(
        Map<Tensor, NodeLifetime> lifetimes,
        int forwardBoundaryIndex
) {
}
