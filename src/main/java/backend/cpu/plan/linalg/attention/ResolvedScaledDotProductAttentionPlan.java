package backend.cpu.plan.linalg.attention;

import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;

public record ResolvedScaledDotProductAttentionPlan(
        ResolvedAttentionHints forwardDirectHints,
        ResolvedAttentionHints backwardSoftmaxGradHints,
        ResolvedMatMulHints backwardQueryGradMatMulHints,
        ResolvedMatMulHints backwardDWeightsMatMulHints,
        ResolvedMatMulHints backwardValueGradMatMulHints,
        ResolvedMatMulHints backwardKeyGradMatMulHints
) {
}
