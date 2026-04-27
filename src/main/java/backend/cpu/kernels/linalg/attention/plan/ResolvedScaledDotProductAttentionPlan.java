package backend.cpu.kernels.linalg.attention.plan;

import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;

public record ResolvedScaledDotProductAttentionPlan(
        ResolvedAttentionHints forwardDirectHints,
        ResolvedAttentionHints backwardSoftmaxGradHints,
        ResolvedMatMulHints backwardQueryGradMatMulHints,
        ResolvedMatMulHints backwardDWeightsMatMulHints,
        ResolvedMatMulHints backwardValueGradMatMulHints,
        ResolvedMatMulHints backwardKeyGradMatMulHints
) {
}
