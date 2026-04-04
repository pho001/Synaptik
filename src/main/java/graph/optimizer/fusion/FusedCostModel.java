package graph.optimizer.fusion;

import graph.codegen.FusedAccessKind;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedExternalInputPlan;
import graph.codegen.FusedNodePlan;
import operations.Operation;
import tensor.DataType;
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

    public static boolean resolveLowCostHint(FusedExpressionPlan plan) {
        if (plan == null || plan.nodes().isEmpty()) {
            return false;
        }
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (input.accessKind() != FusedAccessKind.DIRECT_CONTIGUOUS
                    && input.accessKind() != FusedAccessKind.OFFSET_CONTIGUOUS) {
                return false;
            }
            if (input.dataType() == DataType.BOOL) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (node.outputType() == DataType.BOOL) {
                return false;
            }
            Operation.OpType type = node.opType();
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

    public static int estimateDispatchComplexity(FusedExpressionPlan plan) {
        if (plan == null || plan.nodes().isEmpty()) {
            return 1;
        }
        int total = 0;
        for (FusedExternalInputPlan input : plan.inputs()) {
            total += switch (input.accessKind()) {
                case DIRECT_CONTIGUOUS -> 0;
                case OFFSET_CONTIGUOUS -> 1;
                case DIRECT_STRIDED -> 2;
                case BROADCAST_STRIDED -> 3;
                case OFFSET_STRIDED -> 4;
            };
            if (input.dataType() == DataType.BOOL) {
                total += 2;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            total += switch (node.opType()) {
                case GT, GE, LT, LE, EQ, NE -> 2;
                case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> 2;
                case WHERE -> 3;
                default -> 1;
            };
            if (node.outputType() == DataType.BOOL) {
                total += 1;
            }
        }
        return Math.max(1, total);
    }

    public static int resolveDispatchScale(int dispatchComplexity) {
        int normalized = (Math.max(1, dispatchComplexity) + 7) / 8;
        return Math.max(1, Math.min(8, normalized));
    }
}
