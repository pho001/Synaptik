package graph.optimizer.fusion;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class FusedCostModel {
    public static boolean resolveLowCostHint(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return false;
        }
        for (Tensor t : cluster) {
            if (t == null || t.getOperation() == null) {
                continue;
            }
            Operation.OpType type = t.getOperation().opType();
            if (type == null) {
                return false;
            }
            switch (type) {
                case ADD, SUB, MUL, MIN, MAX, NEG, MUL_SCALAR, RELU, NOOP -> {
                    // keep scanning
                }
                default -> {
                    return false;
                }
            }
        }
        return true;
    }
    public static int estimateDispatchComplexity(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return 1;
        }
        int total = 0;
        for (Tensor t : cluster) {
            if (t == null || t.getOperation() == null) {
                continue;
            }
            total += t.getOperation().isCheap() ? 1 : 4;
        }
        return Math.max(1, total);

    }
    public static int resolveDispatchScale(int dispatchComplexity) {
        int normalized = (Math.max(1, dispatchComplexity) + 7) / 8;
        return Math.max(1, Math.min(8, normalized));
    }
}
