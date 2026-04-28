package graph.optimizer.intent;

import backend.ComputeBackend;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * Utilities for preserving accelerator backend intent across rewrites and backward closures.
 *
 * <p>The methods mutate tensor backend metadata. They are intended for compile-time optimizer use, not concurrent graph
 * mutation.
 */
public final class BackendIntentPropagator {
    private BackendIntentPropagator() {
    }

    /**
     * Copies accelerator backend intent from a source tensor to a target tensor.
     *
     * @param target tensor to update
     * @param source tensor whose backend intent should be preserved
     */
    public static void preserve(Tensor target, Tensor source) {
        if (target == null || source == null) {
            return;
        }
        preserve(target, source.resolveBackend());
    }

    /**
     * Preserves an accelerator backend on a target tensor.
     *
     * @param target tensor to update
     * @param backend backend to preserve
     */
    public static void preserve(Tensor target, ComputeBackend backend) {
        if (target == null || !isAcceleratorBackend(backend)) {
            return;
        }
        TensorInternalAccess.setBackend(target, backend);
    }

    /**
     * Propagates accelerator backend intent backward through supported producer closures.
     *
     * @param graph graph in topological order
     */
    public static void propagateBackwardClosure(List<Tensor> graph) {
        if (graph == null || graph.isEmpty()) {
            return;
        }
        for (int i = graph.size() - 1; i >= 0; i--) {
            Tensor tensor = graph.get(i);
            ComputeBackend backend = tensor == null ? null : tensor.resolveBackend();
            if (isAcceleratorBackend(backend)) {
                propagateBackwardIntent(tensor, backend, java.util.Collections.newSetFromMap(new IdentityHashMap<>()));
            }
        }
    }

    private static void propagateBackwardIntent(Tensor tensor, ComputeBackend backend, Set<Tensor> seen) {
        if (tensor == null || !isAcceleratorBackend(backend) || !seen.add(tensor)) {
            return;
        }
        if (tensor.resolveBackend() == ComputeBackend.CPU) {
            TensorInternalAccess.setBackend(tensor, backend);
        }

        Operation op = tensor.getOperation();
        List<Tensor> inputs = tensor.getPrevTensors();
        if (op == null || inputs == null || inputs.isEmpty()) {
            return;
        }

        switch (op.opType()) {
            case RELU, TANH, FAST_TANH, SIGMOID, ABS, EXP, FAST_EXP, LOG,
                    SUM, MEAN, RESHAPE, CONTIGUOUS, NOOP -> {
                if (inputs.size() == 1) {
                    propagateBackwardIntent(inputs.getFirst(), backend, seen);
                }
            }
            case ADD -> {
                for (Tensor input : inputs) {
                    propagateBackwardIntent(input, backend, seen);
                }
            }
            case MATMUL, LINEAR -> {
                // Compute roots are terminal claims for the current prototype.
            }
            default -> {
            }
        }
    }

    private static boolean isAcceleratorBackend(ComputeBackend backend) {
        return backend != null && backend != ComputeBackend.CPU;
    }
}
