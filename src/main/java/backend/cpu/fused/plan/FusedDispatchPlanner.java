package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import graph.CompiledNode;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.function.IntFunction;

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
     * Classifies a fused plan using the original operation descriptors that produced the plan.
     */
    public static FusedDispatchFamily resolveDispatchFamily(
            FusedExpressionPlan plan,
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        if (plan == null || plan.nodes().isEmpty()) {
            return FusedDispatchFamily.NON_CHEAP_STRIDED;
        }
        boolean contiguous = isContiguousAccessPlan(plan);
        boolean cheap = containsOnlyCheapNumericOps(plan, orderedNodeIds, compiledNodeResolver);
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
     * Resolves the low-cost hint from a lowered expression plan and original operation descriptors.
     */
    public static boolean resolveLowCostHint(
            FusedExpressionPlan plan,
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        return resolveDispatchFamily(plan, orderedNodeIds, compiledNodeResolver) == FusedDispatchFamily.CHEAP_CONTIGUOUS;
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

    private static boolean containsOnlyCheapNumericOps(
            FusedExpressionPlan plan,
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        if (orderedNodeIds == null || compiledNodeResolver == null || orderedNodeIds.size() != plan.nodes().size()) {
            return containsOnlyCheapNumericOps(plan);
        }
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (input.dataType() == DataType.BOOL) {
                return false;
            }
        }
        for (int i = 0; i < plan.nodes().size(); i++) {
            FusedNodePlan node = plan.nodes().get(i);
            if (node.outputType() == DataType.BOOL) {
                return false;
            }
            CompiledNode compiledNode = compiledNodeResolver.apply(orderedNodeIds.get(i));
            if (compiledNode == null || !isCheapNumericOp(compiledNode.operation())) {
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
        return isLegacyCheapFusableNumericArithmeticOp(opType);
    }

    private static boolean isCheapNumericOp(Operation operation) {
        if (operation == null || operation.opType() == null) {
            return false;
        }
        return operation.isFusable()
                && operation.resultKind() == Operation.OpResultKind.NUMERIC
                && operation.semanticFamily() == Operation.OpSemanticFamily.ARITHMETIC
                && operation.computationalCost() == Operation.OpComputationalCost.CHEAP;
    }

    private static boolean isLegacyCheapFusableNumericArithmeticOp(Operation.OpType opType) {
        // Old CPU fused plans and tests can reach this planner after canonicalization with only opType identity.
        return switch (opType) {
            case ADD, SUB, MUL, MIN, MAX, NEG, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX -> true;
            default -> false;
        };
    }
}
