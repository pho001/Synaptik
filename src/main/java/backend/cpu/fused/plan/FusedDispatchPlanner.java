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
        if (opType == Operation.OpType.CONST_SCALAR || opType == Operation.OpType.NOOP) {
            return true;
        }
        return opType.isFusable()
                && opType.resultKind() == Operation.OpResultKind.NUMERIC
                && opType.semanticFamily() == Operation.OpSemanticFamily.ARITHMETIC
                && opType.computationalCost() == Operation.OpComputationalCost.CHEAP;
    }
}
