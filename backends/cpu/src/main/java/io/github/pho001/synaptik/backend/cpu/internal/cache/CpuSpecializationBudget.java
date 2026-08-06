package io.github.pho001.synaptik.backend.cpu.internal.cache;

/**
 * Immutable hard ceiling on generated specialization growth for one current CPU analysis.
 *
 * <p>The present slice permits the direct candidate plus three one-input copy candidates, realizes
 * exactly one selected artifact after assignment, and permits no fixed-shape or unrolled class.
 *
 * @param candidatePlans positive complete-candidate count, at most four
 * @param realizedArtifacts realized artifact count; exactly one
 * @param fixedShapeVariants fixed-shape variant count; exactly zero
 * @param unrolledVariants unrolled variant count; exactly zero
 */
public record CpuSpecializationBudget(int candidatePlans, int realizedArtifacts,
        int fixedShapeVariants, int unrolledVariants) {
    public static final int MAX_CANDIDATE_PLANS = 4;
    public static final int MAX_REALIZED_ARTIFACTS = 1;
    /**
     * Validates the complete current budget.
     *
     * @throws IllegalArgumentException if any count exceeds or disagrees with the current hard
     *     limits
     */
    public CpuSpecializationBudget {
        if (candidatePlans < 1 || candidatePlans > MAX_CANDIDATE_PLANS
                || realizedArtifacts != MAX_REALIZED_ARTIFACTS
                || fixedShapeVariants != 0 || unrolledVariants != 0) {
            throw new IllegalArgumentException("CPU specialization budget exceeded");
        }
    }
}
