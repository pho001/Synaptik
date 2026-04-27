package backend.cpu.kernels.plan;

import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.cpu.kernels.nn.conv2d.plan.ResolvedConv2dHints;
import backend.cpu.kernels.reduction.plan.ResolvedReductionHints;

public record ResolvedCpuOperationPlans(
        ResolvedMatMulHints matMulHints,
        ResolvedConv2dHints conv2dHints,
        ResolvedScaledDotProductAttentionPlan attentionPlan,
        ResolvedCpuComputeContract computeContract,
        ResolvedDispatchHints dispatchHints,
        ResolvedReductionHints reductionHints
) {
}
