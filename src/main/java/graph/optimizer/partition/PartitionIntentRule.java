package graph.optimizer.partition;

import backend.ComputeBackend;
import graph.optimizer.OptimizationRule;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

public final class PartitionIntentRule implements OptimizationRule {
    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return List.of();
        }
        for (int i = sortedGraph.size() - 1; i >= 0; i--) {
            Tensor tensor = sortedGraph.get(i);
            ComputeBackend backend = tensor.resolveBackend();
            if (backend == null || backend == ComputeBackend.CPU) {
                continue;
            }
            propagateBackwardIntent(tensor, backend, java.util.Collections.newSetFromMap(new IdentityHashMap<>()));
        }
        return sortedGraph;
    }

    private void propagateBackwardIntent(Tensor tensor, ComputeBackend backend, Set<Tensor> seen) {
        if (tensor == null || backend == null || backend == ComputeBackend.CPU || !seen.add(tensor)) {
            return;
        }
        if (tensor.resolveBackend() == ComputeBackend.CPU) {
            TensorInternalAccess.setBackend(tensor, backend);
        }

        if (tensor.getOperation() == null || tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return;
        }

        switch (tensor.getOperation().opType()) {
            case RELU, TANH, FAST_TANH, SIGMOID, ABS, EXP, FAST_EXP, LOG,
                    SUM, MEAN, RESHAPE, CONTIGUOUS, NOOP -> {
                if (tensor.getPrevTensors().size() == 1) {
                    propagateBackwardIntent(tensor.getPrevTensors().getFirst(), backend, seen);
                }
            }
            case ADD -> {
                for (Tensor input : tensor.getPrevTensors()) {
                    propagateBackwardIntent(input, backend, seen);
                }
            }
            case MATMUL, LINEAR -> {
                // Compute roots are the terminal accelerator claims for the current prototype.
            }
            default -> {
            }
        }
    }
}
