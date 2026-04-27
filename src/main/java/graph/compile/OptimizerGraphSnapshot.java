package graph.compile;

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
 * The snapshot clones the graph topology, backward markers, gradients, and leaf values so optimizer
 * rules can freely mutate the cloned graph without touching the original semantic Tensor graph.
 */
public final class OptimizerGraphSnapshot {
    private final List<Tensor> graph;
    private final Tensor forwardOutput;
    private final Map<Tensor, Tensor> originalBySnapshot;

    private OptimizerGraphSnapshot(
            List<Tensor> graph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> originalBySnapshot
    ) {
        this.graph = List.copyOf(graph);
        this.forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        this.originalBySnapshot = Map.copyOf(originalBySnapshot);
    }

    public static OptimizerGraphSnapshot capture(List<Tensor> graph, Tensor forwardOutput) {
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
        return new OptimizerGraphSnapshot(clonedGraph, clonedForwardOutput, originals);
    }

    public List<Tensor> graph() {
        return graph;
    }

    public Tensor forwardOutput() {
        return forwardOutput;
    }

    public Map<Tensor, Tensor> originalBySnapshot() {
        return originalBySnapshot;
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
        TensorInternalAccess.setBackward(clone, original.isBackward());
        TensorInternalAccess.setBackend(clone, original.resolveBackend());
        TensorInternalAccess.setBackwardFunction(clone, TensorInternalAccess.backwardFunction(original));
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
