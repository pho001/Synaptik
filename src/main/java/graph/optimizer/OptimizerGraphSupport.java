package graph.optimizer;

import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Shared graph mutation and closure helpers for optimizer rules.
 *
 * <p>These utilities operate on mutable tensor graph edges. They are intended for single-threaded optimizer passes over
 * detached compile snapshots or working compile graphs.
 */
public final class OptimizerGraphSupport {
    private OptimizerGraphSupport() {
    }

    /**
     * Rewrites a tensor's input list through a replacement map.
     *
     * @param tensor tensor whose inputs should be rewritten
     * @param replacements replacement map from old tensors to new tensors
     */
    public static void rewriteInputs(Tensor tensor, Map<Tensor, Tensor> replacements) {
        if (tensor.getPrevTensors() == null || replacements.isEmpty()) {
            return;
        }
        List<Tensor> inputs = TensorInternalAccess.prevTensors(tensor);
        for (int i = 0; i < inputs.size(); i++) {
            Tensor resolved = resolveReplacement(inputs.get(i), replacements);
            if (resolved != null) {
                inputs.set(i, resolved);
            }
        }
    }

    /**
     * Resolves a transitive replacement for a tensor.
     *
     * @param tensor original tensor
     * @param replacements replacement map
     * @return final replacement, or {@code null} when no replacement exists
     */
    public static Tensor resolveReplacement(Tensor tensor, Map<Tensor, Tensor> replacements) {
        Tensor current = replacements.get(tensor);
        if (current == null) {
            return null;
        }
        while (replacements.containsKey(current)) {
            current = replacements.get(current);
        }
        return current;
    }

    /**
     * Rebuilds a topological closure from observable roots of a graph.
     *
     * @param graph source graph
     * @return rebuilt graph in post-order topological order
     */
    public static List<Tensor> rebuildTopologicalClosure(List<Tensor> graph) {
        if (graph.isEmpty()) {
            return graph;
        }
        return rebuildTopologicalClosureFromRoots(observableRoots(graph));
    }

    /**
     * Rebuilds a topological closure from explicit roots.
     *
     * @param roots observable roots
     * @return rebuilt graph in post-order topological order
     */
    public static List<Tensor> rebuildTopologicalClosureFromRoots(List<Tensor> roots) {
        if (roots == null || roots.isEmpty()) {
            return List.of();
        }

        Set<Tensor> visited = new HashSet<>();
        List<Tensor> rebuilt = new ArrayList<>();
        for (Tensor root : roots) {
            dfsPostOrder(root, visited, rebuilt);
        }
        return rebuilt;
    }

    /**
     * Finds graph nodes with no consumers.
     *
     * @param graph graph to inspect
     * @return consumer-free sink tensors
     */
    public static List<Tensor> consumerFreeSinks(List<Tensor> graph) {
        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (Tensor tensor : graph) {
            if (tensor.getPrevTensors() == null) {
                continue;
            }
            for (Tensor input : tensor.getPrevTensors()) {
                consumerCounts.put(input, consumerCounts.getOrDefault(input, 0) + 1);
            }
        }

        List<Tensor> sinks = new ArrayList<>();
        for (Tensor tensor : graph) {
            if (consumerCounts.getOrDefault(tensor, 0) == 0) {
                sinks.add(tensor);
            }
        }
        return sinks;
    }

    /**
     * Returns graph roots that must remain observable after rewrites.
     *
     * <p>This includes consumer-free sinks and operation-backed gradient tensors.
     *
     * @param graph graph to inspect
     * @return observable roots in stable order
     */
    public static List<Tensor> observableRoots(List<Tensor> graph) {
        if (graph == null || graph.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<Tensor> roots = new LinkedHashSet<>(consumerFreeSinks(graph));
        for (Tensor tensor : graph) {
            Tensor gradient = tensor.getGradient();
            if (gradient != null && gradient.getOperation() != null) {
                roots.add(gradient);
            }
        }
        return List.copyOf(roots);
    }

    /**
     * Resolves root tensors through a replacement map.
     *
     * @param roots roots to resolve
     * @param replacements replacement map
     * @return resolved roots in stable order
     */
    public static List<Tensor> resolveRoots(List<Tensor> roots, Map<Tensor, Tensor> replacements) {
        if (roots == null || roots.isEmpty()) {
            return List.of();
        }
        if (replacements == null || replacements.isEmpty()) {
            return List.copyOf(roots);
        }
        LinkedHashSet<Tensor> resolvedRoots = new LinkedHashSet<>();
        for (Tensor root : roots) {
            Tensor resolved = resolveReplacement(root, replacements);
            resolvedRoots.add(resolved == null ? root : resolved);
        }
        return List.copyOf(resolvedRoots);
    }

    private static void dfsPostOrder(Tensor node, Set<Tensor> visited, List<Tensor> out) {
        if (node == null || !visited.add(node)) {
            return;
        }
        if (node.getPrevTensors() != null) {
            for (Tensor input : node.getPrevTensors()) {
                dfsPostOrder(input, visited, out);
            }
        }
        out.add(node);
    }
}
