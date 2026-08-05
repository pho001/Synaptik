package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import java.util.Objects;

/**
 * Immutable opaque selected portable plan carried from CPU analysis to finalization.
 *
 * <p>The exact candidate and parallel configuration are retained by reference. The plan is
 * selected before shared assignment and contains no Runtime slot, generated artifact, physical
 * resource, per-run state, or close lifecycle.</p>
 */
final class CpuPortablePreparationPlan implements BackendPreparationPlan {
    private final CpuPortablePartitionCandidate candidate;
    private final CpuPreparedParallelConfiguration parallelConfiguration;

    /**
     * Creates one selected plan without generating code or assigning a resource.
     *
     * @param candidate exact non-null selected candidate
     * @param parallelConfiguration exact non-null prepared range configuration
     * @throws NullPointerException if either argument is {@code null}, in declaration order
     */
    CpuPortablePreparationPlan(
            CpuPortablePartitionCandidate candidate,
            CpuPreparedParallelConfiguration parallelConfiguration) {
        this.candidate = Objects.requireNonNull(candidate, "candidate");
        this.parallelConfiguration =
                Objects.requireNonNull(parallelConfiguration, "parallelConfiguration");
    }

    /**
     * Returns the candidate selected during analysis.
     *
     * @return exact non-null selected candidate retained at construction
     */
    CpuPortablePartitionCandidate candidate() { return candidate; }
    /**
     * Returns the immutable range-dispatch recipe selected during analysis.
     *
     * @return exact non-null prepared parallel configuration retained at construction
     */
    CpuPreparedParallelConfiguration parallelConfiguration() { return parallelConfiguration; }
}
