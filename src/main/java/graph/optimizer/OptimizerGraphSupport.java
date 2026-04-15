package graph.optimizer;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class OptimizerGraphSupport {
    private OptimizerGraphSupport() {
    }

    public static void rewriteInputs(Tensor tensor, Map<Tensor, Tensor> replacements) {
        if (tensor.getPrevTensors() == null || replacements.isEmpty()) {
            return;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        for (int i = 0; i < inputs.size(); i++) {
            Tensor resolved = resolveReplacement(inputs.get(i), replacements);
            if (resolved != null) {
                inputs.set(i, resolved);
            }
        }
    }

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

    public static List<Tensor> rebuildTopologicalClosure(List<Tensor> graph) {
        if (graph.isEmpty()) {
            return graph;
        }
        return rebuildTopologicalClosureFromRoots(consumerFreeSinks(graph));
    }

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
