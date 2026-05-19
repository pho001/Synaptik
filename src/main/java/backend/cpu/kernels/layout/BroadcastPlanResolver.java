package backend.cpu.kernels.layout;

import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import operations.Operation;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;
import tensor.Tensor;
import tensor.layout.WhereBroadcastPlan;
import tensor.layout.WhereBroadcastPlanner;

import java.util.Arrays;
import java.util.List;

public final class BroadcastPlanResolver {
    private BroadcastPlanResolver() {
    }

    public static ResolvedBroadcastPlan resolve(Operation op, List<Tensor> runtimeInputs, Tensor node) {
        if (!supportsBinaryBroadcast(op) || runtimeInputs.size() != 2) {
            return null;
        }

        BroadcastPlan plan = BroadcastPlanner.plan(runtimeInputs.get(0), runtimeInputs.get(1));
        if (!Arrays.equals(plan.outShape(), node.getShapeUnsafe())) {
            throw new IllegalStateException(
                    "Resolved broadcast output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.getShapeUnsafe())
            );
        }
        return ResolvedBroadcastPlan.from(plan);
    }

    public static ResolvedWhereBroadcastPlan resolveWhere(Operation op, List<Tensor> runtimeInputs, Tensor node) {
        if (op == null || op.opType() != Operation.OpType.WHERE || runtimeInputs.size() != 3) {
            return null;
        }
        WhereBroadcastPlan plan = WhereBroadcastPlanner.plan(runtimeInputs.get(0), runtimeInputs.get(1), runtimeInputs.get(2));
        if (!Arrays.equals(plan.outShape(), node.getShapeUnsafe())) {
            throw new IllegalStateException(
                    "Resolved where output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.getShapeUnsafe())
            );
        }
        return ResolvedWhereBroadcastPlan.from(plan);
    }

    public static boolean requiresBinaryBroadcast(Operation op, List<Tensor> inputs, Tensor node) {
        if (!supportsBinaryBroadcast(op) || inputs.size() != 2) {
            return false;
        }
        return !Arrays.equals(inputs.get(0).getShapeUnsafe(), node.getShapeUnsafe())
                || !Arrays.equals(inputs.get(1).getShapeUnsafe(), node.getShapeUnsafe());
    }

    public static boolean supportsBinaryBroadcast(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR -> true;
            default -> false;
        };
    }
}
