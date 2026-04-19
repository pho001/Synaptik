package backend.kernels.cpu.linalg.attention.plan;

import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;

public record ResolvedScaledDotProductAttentionPlan(
        ResolvedAttentionHints forwardDirectHints,
        ResolvedAttentionHints backwardSoftmaxGradHints,
        ResolvedMatMulHints backwardQueryGradMatMulHints,
        ResolvedMatMulHints backwardDWeightsMatMulHints,
        ResolvedMatMulHints backwardValueGradMatMulHints,
        ResolvedMatMulHints backwardKeyGradMatMulHints
) {
}
