package graph.fused.vector;

import graph.codegen.FusedExpressionPlan;
import graph.optimizer.fusion.FusedCostModel;
import operations.Operation;
import tensor.DataType;

final class DirectFusedPlanSupport {
    private DirectFusedPlanSupport() {
    }

    static boolean supportsPlan(FusedExpressionPlan plan, DataType storageType) {
        if (plan == null || storageType == null) {
            return false;
        }
        if (storageType != DataType.FLOAT64 && storageType != DataType.FLOAT32) {
            return false;
        }
        if (!FusedCostModel.resolveLowCostHint(plan)) {
            return false;
        }
        if (plan.outputNode().outputType() == DataType.BOOL) {
            return false;
        }
        for (var input : plan.inputs()) {
            if (!input.isLinearAccess()) {
                return false;
            }
            if (input.dataType() == DataType.BOOL) {
                return false;
            }
            if (input.dataType() != storageType) {
                return false;
            }
        }
        for (var node : plan.nodes()) {
            if (node.outputType() == DataType.BOOL || !supportsVector(node.opType())) {
                return false;
            }
        }
        return true;
    }

    private static boolean supportsVector(Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                    SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, POW, NOOP -> true;
            default -> false;
        };
    }
}
