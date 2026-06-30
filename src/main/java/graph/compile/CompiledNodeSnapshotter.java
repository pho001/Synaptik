package graph.compile;

import graph.compile.intent.BackendIntentPlan;
import graph.model.AliasViewPolicy;
import graph.model.CompiledNode;
import graph.model.CompiledTensorDataSnapshot;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * Compile-time boundary that snapshots mutable Tensor graph nodes into immutable compiled model values.
 */
public final class CompiledNodeSnapshotter {
    private CompiledNodeSnapshotter() {
    }

    /**
     * Captures compiled node snapshots in topological order.
     *
     * @param orderedGraph tensors in topological order
     * @param backendIntentPlan compile-local backend intent, or {@code null} for CPU-default intent
     * @return immutable compiled node snapshots
     */
    public static List<CompiledNode> snapshot(
            List<Tensor> orderedGraph,
            BackendIntentPlan backendIntentPlan
    ) {
        if (orderedGraph == null || orderedGraph.isEmpty()) {
            return List.of();
        }
        BackendIntentPlan intents = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < orderedGraph.size(); i++) {
            ids.put(orderedGraph.get(i), i);
        }
        Set<Tensor> gradientTargets = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor tensor : orderedGraph) {
            Tensor gradient = tensor.getGradient();
            if (gradient != null && ids.containsKey(gradient)) {
                gradientTargets.add(gradient);
            }
        }
        IdentityHashMap<Tensor, Integer> storageOwnerIds = new IdentityHashMap<>();
        List<CompiledNode> out = new ArrayList<>(orderedGraph.size());
        for (int i = 0; i < orderedGraph.size(); i++) {
            Tensor tensor = orderedGraph.get(i);
            List<Tensor> inputs = tensor.getPrevTensors();
            List<Integer> inputIds = new ArrayList<>(inputs == null ? 0 : inputs.size());
            if (inputs != null) {
                for (Tensor input : inputs) {
                    Integer inputId = ids.get(input);
                    if (inputId == null) {
                        throw new IllegalStateException(
                                "Compiled node input is missing from ordered graph: " + tensor.getLabel()
                        );
                    }
                    inputIds.add(inputId);
                }
            }
            int storageOwnerId = resolveStorageOwnerId(i, tensor, inputs, storageOwnerIds);
            storageOwnerIds.put(tensor, storageOwnerId);
            out.add(CompiledNode.compiledSnapshot(
                    i,
                    tensor.getOperation(),
                    intents.backend(tensor),
                    inputIds,
                    storageOwnerId,
                    tensor.getShapeUnsafe(),
                    tensor.getStridesUnsafe(),
                    tensor.getStorageOffsetUnsafe(),
                    tensor.getDataType(),
                    tensor.isBackward(),
                    tensor.getOperation() == null,
                    tensor.getRequiresGrad(),
                    tensor.isTrainableParameter(),
                    tensor.isContiguous(),
                    tensor.hasStorageOffset(),
                    gradientTargets.contains(tensor),
                    tensor.getFlatDataSize(),
                    tensor.getLabel(),
                    CompiledTensorDataSnapshot.captureStaticLeaf(tensor)
            ));
        }
        return List.copyOf(out);
    }

    private static int resolveStorageOwnerId(
            int nodeId,
            Tensor tensor,
            List<Tensor> inputs,
            IdentityHashMap<Tensor, Integer> storageOwnerIds
    ) {
        if (!AliasViewPolicy.aliasesInput0AtRuntime(tensor) || inputs == null || inputs.isEmpty()) {
            return nodeId;
        }
        Tensor input0 = inputs.getFirst();
        return storageOwnerIds.getOrDefault(input0, nodeId);
    }
}
