package graph.compile.session;

import graph.compile.intent.BackendIntentPlan;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Compile-time graph snapshot used by optimizer passes.
 *
 * <p>The snapshot clones the graph topology, backward markers, gradients, and leaf values so optimizer
 * rules can freely mutate the cloned graph without touching the original semantic Tensor graph.
 */
final class OptimizerGraphSnapshot {
    private final List<Tensor> graph;
    private final Tensor forwardOutput;
    private final Map<Tensor, Tensor> originalBySnapshot;
    private final BackendIntentPlan backendIntentPlan;

    private OptimizerGraphSnapshot(
            List<Tensor> graph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> originalBySnapshot,
            BackendIntentPlan backendIntentPlan
    ) {
        this.graph = List.copyOf(graph);
        this.forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        this.originalBySnapshot = Map.copyOf(originalBySnapshot);
        this.backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
    }

    /**
     * Captures a detached optimizer snapshot.
     *
     * @param graph source graph in topological order
     * @param forwardOutput semantic forward output in {@code graph}
     * @return snapshot with cloned graph and clone-to-original mapping
     */
    static OptimizerGraphSnapshot capture(List<Tensor> graph, Tensor forwardOutput, BackendIntentPlan backendIntentPlan) {
        Objects.requireNonNull(graph, "graph cannot be null");
        Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        IdentityHashMap<Tensor, Tensor> clones = new IdentityHashMap<>();
        IdentityHashMap<Tensor, Tensor> originals = new IdentityHashMap<>();
        for (Tensor tensor : graph) {
            cloneRecursive(tensor, clones, originals);
        }

        List<Tensor> clonedGraph = new ArrayList<>(graph.size());
        for (Tensor tensor : graph) {
            Tensor clone = clones.get(tensor);
            if (clone == null) {
                throw new IllegalStateException("Missing cloned tensor for compile snapshot node " + tensor.getLabel());
            }
            clonedGraph.add(clone);
        }
        Tensor clonedForwardOutput = clones.get(forwardOutput);
        if (clonedForwardOutput == null) {
            throw new IllegalStateException("Missing cloned forward output node");
        }
        BackendIntentPlan snapshotPlan = (backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan)
                .remapFromOriginalBySnapshot(originals);
        return new OptimizerGraphSnapshot(clonedGraph, clonedForwardOutput, originals, snapshotPlan);
    }

    /**
     * Returns the cloned optimizer graph.
     *
     * @return cloned graph in topological order
     */
    List<Tensor> graph() {
        return graph;
    }

    /**
     * Returns the cloned forward output tensor.
     *
     * @return cloned forward output
     */
    Tensor forwardOutput() {
        return forwardOutput;
    }

    /**
     * Returns the mapping from snapshot tensors back to original tensors.
     *
     * @return immutable clone-to-original map
     */
    Map<Tensor, Tensor> originalBySnapshot() {
        return originalBySnapshot;
    }

    /**
     * Returns backend intent remapped to cloned optimizer tensors.
     *
     * @return snapshot-local backend intent plan
     */
    BackendIntentPlan backendIntentPlan() {
        return backendIntentPlan;
    }

    private static Tensor cloneRecursive(
            Tensor original,
            IdentityHashMap<Tensor, Tensor> clones,
            IdentityHashMap<Tensor, Tensor> originals
    ) {
        if (original == null) {
            return null;
        }
        Tensor existing = clones.get(original);
        if (existing != null) {
            return existing;
        }

        Tensor clone = new Tensor(
                original.getShapeUnsafe().clone(),
                original.getStridesUnsafe().clone(),
                original.getStorageOffsetUnsafe(),
                null,
                original.getOperation(),
                original.getLabel(),
                original.getDataType()
        );
        clone.setRequiresGrad(original.getRequiresGrad());
        clone.setTrainableParameter(original.isTrainableParameter());
        TensorInternalAccess.setBackward(clone, original.isBackward());
        TensorInternalAccess.setGradientRule(clone, TensorInternalAccess.gradientRule(original));
        clones.put(original, clone);
        originals.put(clone, original);

        List<Tensor> prev = original.getPrevTensors();
        if (prev == null) {
            TensorInternalAccess.setPrevTensors(clone, null);
        } else {
            List<Tensor> mappedPrev = new ArrayList<>(prev.size());
            for (Tensor parent : prev) {
                mappedPrev.add(cloneRecursive(parent, clones, originals));
            }
            TensorInternalAccess.setPrevTensors(clone, mappedPrev);
        }

        Tensor gradient = original.getGradient();
        if (gradient != null) {
            TensorInternalAccess.setGradient(clone, cloneRecursive(gradient, clones, originals));
        }

        if (original.getOperation() == null) {
            clone.copyDataFrom(original);
        }
        return clone;
    }
}
