package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import operations.Operation;
import tensor.DataType;

/**
 * Internal cost heuristics for fused CPU planning and dispatch.
 */
public final class FusedDispatchPlanner {
    private FusedDispatchPlanner() {
    }

    /**
     * Classifies a fused plan by operation cost and access contiguity.
     */
    public static FusedDispatchFamily resolveDispatchFamily(FusedExpressionPlan plan) {
        if (plan == null || plan.nodes().isEmpty()) {
            return FusedDispatchFamily.NON_CHEAP_STRIDED;
        }
        boolean contiguous = isContiguousAccessPlan(plan);
        boolean cheap = containsOnlyCheapNumericOps(plan);
        if (cheap) {
            return contiguous ? FusedDispatchFamily.CHEAP_CONTIGUOUS : FusedDispatchFamily.CHEAP_STRIDED;
        }
        return contiguous ? FusedDispatchFamily.NON_CHEAP_CONTIGUOUS : FusedDispatchFamily.NON_CHEAP_STRIDED;
    }

    /**
     * Resolves the low-cost hint from a lowered expression plan.
     */
    public static boolean resolveLowCostHint(FusedExpressionPlan plan) {
        return resolveDispatchFamily(plan) == FusedDispatchFamily.CHEAP_CONTIGUOUS;
    }

    private static boolean isContiguousAccessPlan(FusedExpressionPlan plan) {
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (input.accessKind() != FusedAccessKind.DIRECT_CONTIGUOUS
                    && input.accessKind() != FusedAccessKind.OFFSET_CONTIGUOUS) {
                return false;
            }
        }
        return true;
    }

    private static boolean containsOnlyCheapNumericOps(FusedExpressionPlan plan) {
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (input.dataType() == DataType.BOOL) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (node.outputType() == DataType.BOOL) {
                return false;
            }
            if (!isCheapNumericOp(node.opType())) {
                return false;
            }
        }
        return true;
    }

    private static boolean isCheapNumericOp(Operation.OpType opType) {
        if (opType == null) {
            return false;
        }
        return switch (opType) {
            case ADD, SUB, MUL, MIN, MAX, NEG, CONST_SCALAR, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, ABS, NOOP -> true;
            case DIV, INV, SQRT, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SIGMOID, POW, POW_TENSOR -> false;
            case GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE -> false;
            default -> false;
        };
    }
}
