package backend.cpu.fused.optimize;

import backend.cpu.fused.codegen.FusedAccessKind;
import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import operations.Operation;
import tensor.DataType;

/**
 * Internal cost heuristics for fused CPU planning and dispatch.
 */
public class FusedCostModel {
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

    /**
     * Estimates dispatch complexity from a lowered expression plan.
     */
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

    /**
     * Maps dispatch complexity into a bounded scheduler scale.
     */
    public static int resolveDispatchScale(int dispatchComplexity) {
        int normalized = (Math.max(1, dispatchComplexity) + 7) / 8;
        return Math.max(1, Math.min(8, normalized));
    }

    /**
     * Estimates the extra access cost introduced by non-direct fused inputs.
     */
    public static double estimateFusionAccessPenalty(FusedExpressionPlan plan) {
        if (plan == null || plan.inputs().isEmpty()) {
            return 0.0d;
        }
        double total = 0.0d;
        for (FusedExternalInputPlan input : plan.inputs()) {
            total += switch (input.accessKind()) {
                case DIRECT_CONTIGUOUS -> 0.0d;
                case OFFSET_CONTIGUOUS -> 0.10d;
                case DIRECT_STRIDED -> 0.35d;
                case OFFSET_STRIDED -> 0.55d;
                case BROADCAST_STRIDED -> 0.75d;
            };
        }
        return total;
    }

    /**
     * Returns whether a small affine plan should avoid fusion because broadcast overhead dominates.
     */
    public static boolean rejectBroadcastHeavySmallAffinePlan(FusedExpressionPlan plan) {
        if (plan == null || plan.nodes().isEmpty()) {
            return false;
        }
        long broadcastInputs = plan.inputs().stream()
                .filter(input -> input.accessKind() == FusedAccessKind.BROADCAST_STRIDED)
                .count();
        if (broadcastInputs < 2) {
            return false;
        }
        boolean normalizationStyle = false;
        for (FusedNodePlan node : plan.nodes()) {
            switch (node.opType()) {
                case DIV, SQRT -> normalizationStyle = true;
                case ADD, SUB, MUL, RELU, CLAMP_MIN, CLAMP_MAX, ABS, NOOP -> { }
                default -> {
                    return false;
                }
            }
        }
        return normalizationStyle;
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
