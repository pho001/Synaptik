package backend.kernels.cpu.plan;

import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.nn.conv2d.plan.ResolvedConv2dHints;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;

public record ResolvedCpuOperationPlans(
        ResolvedMatMulHints matMulHints,
        ResolvedConv2dHints conv2dHints,
        ResolvedScaledDotProductAttentionPlan attentionPlan,
        ResolvedCpuComputeContract computeContract,
        ResolvedDispatchHints dispatchHints,
        ResolvedReductionHints reductionHints
) {
}
