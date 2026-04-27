package backend.cpu.kernels.plan;

import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.cpu.kernels.nn.conv2d.plan.ResolvedConv2dHints;
import backend.cpu.kernels.reduction.ReductionLogicalSize;
import backend.cpu.kernels.reduction.plan.ResolvedReductionHints;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

final class CpuOperationPlanResolver {
    private CpuOperationPlanResolver() {
    }

    static ResolvedCpuOperationPlans resolve(
            Operation op,
            List<Tensor> runtimeInputs,
            Tensor node,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        ResolvedMatMulHints matMulHints =
                (op != null && (op.opType() == Operation.OpType.MATMUL || op.opType() == Operation.OpType.LINEAR) && runtimeInputs.size() >= 2)
                        ? planner.resolveMatMulHints(runtimeInputs.get(0), runtimeInputs.get(1), node, blasConfig)
                        : null;

        ResolvedConv2dHints conv2dHints = planner.resolveConv2dHints(op, runtimeInputs, node, conv2dConfig);
        ResolvedScaledDotProductAttentionPlan attentionPlan =
                planner.resolveScaledDotProductAttentionPlan(op, runtimeInputs, node, blasConfig);

        ResolvedCpuComputeContract computeContract = planner.resolveComputeContract(
                op,
                runtimeInputs,
                node,
                blasConfig,
                matMulHints,
                conv2dHints
        );

        ResolvedDispatchHints dispatchHints = shouldResolveDispatch(op)
                ? (dispatchHintsOverride != null ? dispatchHintsOverride : planner.resolveDispatchHints(op, node, computeContract))
                : null;

        ResolvedReductionHints reductionHints = shouldResolveReduction(op)
                ? planner.resolveReductionHints(ReductionLogicalSize.estimate(runtimeInputs, node), computeContract)
                : null;

        return new ResolvedCpuOperationPlans(
                matMulHints,
                conv2dHints,
                attentionPlan,
                computeContract,
                dispatchHints,
                reductionHints
        );
    }

    private static boolean shouldResolveDispatch(Operation op) {
        return op != null && (op.opType().category() == Operation.OpArityClass.ELEMENT_WISE || op.opType() == Operation.OpType.FUSED);
    }

    private static boolean shouldResolveReduction(Operation op) {
        if (op == null) {
            return false;
        }
        return switch (op.opType()) {
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD -> true;
            default -> false;
        };
    }
}
