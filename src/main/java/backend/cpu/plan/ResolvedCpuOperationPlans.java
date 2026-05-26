package backend.cpu.plan;

import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.plan.reduction.ResolvedReductionHints;

public record ResolvedCpuOperationPlans(
        ResolvedMatMulHints matMulHints,
        ResolvedScaledDotProductAttentionPlan attentionPlan,
        ResolvedCpuComputeContract computeContract,
        ResolvedDispatchHints dispatchHints,
        ResolvedReductionHints reductionHints
) {
}
