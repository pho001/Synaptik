package tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

final class TensorGraphTraversal {
    private TensorGraphTraversal() {
    }

    static List<Tensor> topologicalSort(Tensor root) {
        Deque<Tensor> sorted = new ArrayDeque<>();
        Set<Tensor> visited = new LinkedHashSet<>();
        visit(root, visited, sorted);
        return new ArrayList<>(sorted);
    }

    private static void visit(Tensor tensor, Set<Tensor> visited, Deque<Tensor> sorted) {
        if (visited.contains(tensor)) {
            return;
        }
        visited.add(tensor);
        List<Tensor> previous = tensor.getPrevTensors();
        if (previous != null) {
            for (Tensor prev : previous) {
                visit(prev, visited, sorted);
            }
        }
        if (previous == null) {
            sorted.addFirst(tensor);
        } else {
            sorted.addLast(tensor);
        }
    }
}
