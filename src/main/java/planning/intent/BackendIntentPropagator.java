package planning.intent;

import backend.contract.ComputeBackend;
import operations.Operation;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utilities for preserving accelerator backend intent across rewrites and backward closures.
 */
public final class BackendIntentPropagator {
    private BackendIntentPropagator() {
    }

    /**
     * Copies accelerator backend intent from an origin tensor to a target tensor.
     *
     * @param target tensor to update
     * @param source origin tensor whose backend intent should be preserved
     * @param plan backend intent plan
     * @return updated backend intent plan
     */
    public static BackendIntentPlan preserve(Tensor target, Tensor source, BackendIntentPlan plan) {
        BackendIntentPlan current = plan == null ? BackendIntentPlan.empty() : plan;
        if (target == null || source == null) {
            return current;
        }
        return preserve(target, current.backend(source), current);
    }

    /**
     * Preserves an accelerator backend on a target tensor.
     *
     * @param target tensor to update
     * @param backend backend to preserve
     * @param plan backend intent plan
     * @return updated backend intent plan
     */
    public static BackendIntentPlan preserve(Tensor target, ComputeBackend backend, BackendIntentPlan plan) {
        BackendIntentPlan current = plan == null ? BackendIntentPlan.empty() : plan;
        if (target == null || !isAcceleratorBackend(backend)) {
            return current;
        }
        return current.withBackend(target, backend);
    }

    /**
     * Propagates accelerator backend intent backward through supported producer closures.
     *
     * @param graph graph in topological order
     * @param plan backend intent plan
     * @return updated backend intent plan
     */
    public static BackendIntentPlan propagateBackwardClosure(List<Tensor> graph, BackendIntentPlan plan) {
        BackendIntentPlan current = plan == null ? BackendIntentPlan.empty() : plan;
        if (graph == null || graph.isEmpty()) {
            return current;
        }
        IdentityHashMap<Tensor, ComputeBackend> intents = current.mutableCopy();
        for (Tensor tensor : graph) {
            ComputeBackend backend = current.backend(tensor);
            if (isAcceleratorBackend(backend)) {
                intents.put(tensor, backend);
            }
        }
        for (int i = graph.size() - 1; i >= 0; i--) {
            Tensor tensor = graph.get(i);
            ComputeBackend backend = tensor == null ? null : resolve(intents, tensor);
            if (isAcceleratorBackend(backend)) {
                Set<Tensor> seen = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
                propagateBackwardIntent(tensor, backend, intents, seen);
                propagateBackwardIntent(tensor.getGradient(), backend, intents, seen);
            }
        }
        return BackendIntentPlan.fromMutable(intents);
    }

    private static void propagateBackwardIntent(
            Tensor tensor,
            ComputeBackend backend,
            Map<Tensor, ComputeBackend> intents,
            Set<Tensor> seen
    ) {
        if (tensor == null || !isAcceleratorBackend(backend) || !seen.add(tensor)) {
            return;
        }
        if (resolve(intents, tensor) == ComputeBackend.CPU) {
            intents.put(tensor, backend);
        }

        Operation op = tensor.getOperation();
        List<Tensor> inputs = tensor.getPrevTensors();
        if (op == null || inputs == null || inputs.isEmpty()) {
            return;
        }

        switch (op.opType()) {
            case RELU, TANH, FAST_TANH, SIGMOID, ABS, EXP, FAST_EXP, ERF, LOG, NEG, INV, POW, SQRT, FLOOR, CEIL, SIGN,
                    CLAMP_MIN, CLAMP_MAX, SUM, MEAN, REDUCE_MIN, REDUCE_MAX,
                    RESHAPE, CONTIGUOUS, PERMUTE, EXPAND, EXPAND_DIMS, SQUEEZE, SELECT, SLICE_BACKWARD, NOOP -> {
                if (inputs.size() == 1) {
                    propagateBackwardIntent(inputs.getFirst(), backend, intents, seen);
                }
            }
            case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR, GT, GE, LT, LE, EQ, NE,
                    LOGICAL_AND, LOGICAL_OR, WHERE, MUL_SCALAR,
                    GATHER, GATHER_AXIS, GATHER_ND, GATHER_ND_GRAD, TAKE_ALONG_AXIS,
                    SCATTER_ADD, SCATTER_AXIS_ADD, SCATTER_ELEMENTS, SCATTER_ND,
                    MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD -> {
                for (Tensor input : inputs) {
                    propagateBackwardIntent(input, backend, intents, seen);
                }
            }
            case MATMUL, LINEAR -> {
                for (Tensor input : inputs) {
                    propagateLayoutProducerIntent(input, backend, intents, seen);
                }
            }
            default -> {
            }
        }
    }

    private static void propagateLayoutProducerIntent(
            Tensor tensor,
            ComputeBackend backend,
            Map<Tensor, ComputeBackend> intents,
            Set<Tensor> seen
    ) {
        if (tensor == null || !isAcceleratorBackend(backend) || !seen.add(tensor)) {
            return;
        }
        Operation op = tensor.getOperation();
        if (op == null || !isLayoutSupportProducer(op.opType())) {
            return;
        }
        if (resolve(intents, tensor) == ComputeBackend.CPU) {
            intents.put(tensor, backend);
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null) {
            return;
        }
        for (Tensor input : inputs) {
            propagateLayoutProducerIntent(input, backend, intents, seen);
        }
    }

    private static boolean isLayoutSupportProducer(Operation.OpType opType) {
        return switch (opType) {
            case RESHAPE, CONTIGUOUS, PERMUTE, EXPAND, EXPAND_DIMS, SQUEEZE, SLICE -> true;
            default -> false;
        };
    }

    private static boolean isAcceleratorBackend(ComputeBackend backend) {
        return BackendIntentPlan.isAcceleratorBackend(backend);
    }

    private static ComputeBackend resolve(Map<Tensor, ComputeBackend> intents, Tensor tensor) {
        return tensor == null ? ComputeBackend.CPU : intents.getOrDefault(tensor, ComputeBackend.CPU);
    }
}
