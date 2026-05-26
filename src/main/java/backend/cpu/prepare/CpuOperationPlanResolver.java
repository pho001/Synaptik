package backend.cpu.prepare;

import backend.cpu.plan.ResolvedCpuOperationPlans;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.kernels.reduction.ReductionLogicalSize;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;

import java.util.List;

final class CpuOperationPlanResolver {
    private CpuOperationPlanResolver() {
    }

    static ResolvedCpuOperationPlans resolve(
            Operation op,
            List<CompiledTensorDescriptor> runtimeInputs,
            List<CompiledTensorDescriptor> inputDescriptors,
            CompiledTensorDescriptor nodeDescriptor,
            CompiledTensorDescriptorIndex descriptorIndex,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
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

        ResolvedScaledDotProductAttentionPlan attentionPlan =
                planner.resolveScaledDotProductAttentionPlan(op, runtimeInputs, nodeDescriptor, descriptorIndex, blasConfig);

        ResolvedCpuComputeContract computeContract = planner.resolveComputeContract(
                op,
                nodeDescriptor,
                blasConfig,
                matMulHints
        );

        ResolvedDispatchHints dispatchHints = shouldResolveDispatch(op)
                ? (dispatchHintsOverride != null ? dispatchHintsOverride : planner.resolveDispatchHints(op, nodeDescriptor, computeContract))
                : null;

        ResolvedReductionHints reductionHints = shouldResolveReduction(op)
                ? planner.resolveReductionHints(ReductionLogicalSize.estimate(runtimeInputs, nodeDescriptor), computeContract)
                : null;

        return new ResolvedCpuOperationPlans(
                matMulHints,
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
                    SOFTMAX, LOG_SOFTMAX,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES -> true;
            default -> false;
        };
    }
}
