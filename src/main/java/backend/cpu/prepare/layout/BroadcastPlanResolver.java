package backend.cpu.prepare.layout;

import planning.descriptor.CompiledTensorDescriptor;
import backend.cpu.plan.layout.ResolvedBroadcastPlan;
import backend.cpu.plan.layout.ResolvedWhereBroadcastPlan;
import operations.Operation;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;
import tensor.layout.WhereBroadcastPlan;
import tensor.layout.WhereBroadcastPlanner;

import java.util.Arrays;
import java.util.List;

public final class BroadcastPlanResolver {
    private BroadcastPlanResolver() {
    }

    public static ResolvedBroadcastPlan resolve(
            Operation op,
            List<CompiledTensorDescriptor> runtimeInputs,
            CompiledTensorDescriptor node
    ) {
        if (!supportsBinaryBroadcast(op) || runtimeInputs.size() != 2) {
            return null;
        }

        BroadcastPlan plan = BroadcastPlanner.plan(
                runtimeInputs.get(0).shape(),
                runtimeInputs.get(0).strides(),
                runtimeInputs.get(1).shape(),
                runtimeInputs.get(1).strides()
        );
        if (!Arrays.equals(plan.outShape(), node.shape())) {
            throw new IllegalStateException(
                    "Resolved broadcast output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.shape())
            );
        }
        return ResolvedBroadcastPlan.from(plan);
    }

    public static ResolvedWhereBroadcastPlan resolveWhere(
            Operation op,
            List<CompiledTensorDescriptor> runtimeInputs,
            CompiledTensorDescriptor node
    ) {
        if (op == null || op.opType() != Operation.OpType.WHERE || runtimeInputs.size() != 3) {
            return null;
        }
        WhereBroadcastPlan plan = WhereBroadcastPlanner.plan(
                runtimeInputs.get(0).shape(),
                runtimeInputs.get(0).strides(),
                runtimeInputs.get(1).shape(),
                runtimeInputs.get(1).strides(),
                runtimeInputs.get(2).shape(),
                runtimeInputs.get(2).strides()
        );
        if (!Arrays.equals(plan.outShape(), node.shape())) {
            throw new IllegalStateException(
                    "Resolved where output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.shape())
            );
        }
        return ResolvedWhereBroadcastPlan.from(plan);
    }

    public static boolean requiresBinaryBroadcast(
            Operation op,
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node
    ) {
        if (!supportsBinaryBroadcast(op) || inputs.size() != 2) {
            return false;
        }
        return !Arrays.equals(inputs.get(0).shape(), node.shape())
                || !Arrays.equals(inputs.get(1).shape(), node.shape());
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
