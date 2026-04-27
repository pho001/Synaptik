package graph.compile;

import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class BackwardGraphBuilder {
    private BackwardGraphBuilder() {
    }

    public record Result(List<Tensor> backwardTargets) {
        public Result {
            backwardTargets = List.copyOf(backwardTargets == null ? List.of() : backwardTargets);
        }
    }

    public static Result build(List<Tensor> forwardGraph, Tensor forwardRoot) {
        List<Tensor> graph = List.copyOf(forwardGraph == null ? List.of() : forwardGraph);
        TensorInternalAccess.setGradient(forwardRoot, Tensor.onesLike(forwardRoot));
        for (int i = graph.size() - 1; i >= 0; i--) {
            TensorInternalAccess.buildBackwardGraph(graph.get(i));
        }

        List<Tensor> backwardTargets = collectBackwardTargets(graph);
        if (backwardTargets.isEmpty()) {
            backwardTargets = collectAvailableGradients(graph);
        }
        markBackwardNodes(graph);
        return new Result(backwardTargets);
    }

    private static List<Tensor> collectAvailableGradients(List<Tensor> graph) {
        List<Tensor> out = new ArrayList<>();
        for (Tensor tensor : graph) {
            if (tensor.getGradient() != null) {
                out.add(tensor.getGradient());
            }
        }
        return out;
    }

    private static void markBackwardNodes(List<Tensor> graph) {
        Set<Tensor> visited = new HashSet<>();
        Set<Tensor> forwardSet = new HashSet<>(graph);

        for (int i = graph.size() - 1; i >= 0; i--) {
            Tensor gradTensor = graph.get(i).getGradient();
            if (gradTensor != null) {
                markBackwardDfs(gradTensor, visited, forwardSet);
            }
        }
    }

    private static void markBackwardDfs(Tensor tensor, Set<Tensor> visited, Set<Tensor> forwardSet) {
        if (tensor == null || visited.contains(tensor)) {
            return;
        }

        visited.add(tensor);
        if (tensor.getPrevTensors() != null) {
            for (Tensor parent : tensor.getPrevTensors()) {
                markBackwardDfs(parent, visited, forwardSet);
            }
        }

        if (tensor.getOperation() != null && !forwardSet.contains(tensor)) {
            TensorInternalAccess.setBackward(tensor, true);
        }
    }

    private static List<Tensor> collectBackwardTargets(List<Tensor> graph) {
        Set<Tensor> unique = new LinkedHashSet<>();
        for (Tensor tensor : graph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad() && tensor.getGradient() != null) {
                unique.add(tensor.getGradient());
            }
        }
        return List.copyOf(unique);
    }
}
