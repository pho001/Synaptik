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
import config.runtime.CpuStorageProfile;
import graph.compile.descriptor.CompiledTensorDescriptor;
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
            List<CompiledTensorDescriptor> inputDescriptors,
            CompiledTensorDescriptor nodeDescriptor,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        ResolvedMatMulHints matMulHints =
                (op != null
                        && (op.opType() == Operation.OpType.MATMUL || op.opType() == Operation.OpType.LINEAR)
                        && inputDescriptors != null
                        && inputDescriptors.size() >= 2
                        && nodeDescriptor != null)
                        ? planner.resolveMatMulHints(
                                inputDescriptors.get(0),
                                inputDescriptors.get(1),
                                nodeDescriptor,
                                blasConfig,
                                cpuStorageProfile,
                                publishFloatContinuation
                        )
                        : null;

        ResolvedConv2dHints conv2dHints = planner.resolveConv2dHints(op, runtimeInputs, node, conv2dConfig);
        ResolvedScaledDotProductAttentionPlan attentionPlan =
                planner.resolveScaledDotProductAttentionPlan(op, runtimeInputs, node, blasConfig);

        ResolvedCpuComputeContract computeContract = planner.resolveComputeContract(
                op,
                nodeDescriptor,
                blasConfig,
                matMulHints,
                conv2dHints
        );

        ResolvedDispatchHints dispatchHints = shouldResolveDispatch(op)
                ? (dispatchHintsOverride != null ? dispatchHintsOverride : planner.resolveDispatchHints(op, nodeDescriptor, computeContract))
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
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD -> true;
            default -> false;
        };
    }
}
