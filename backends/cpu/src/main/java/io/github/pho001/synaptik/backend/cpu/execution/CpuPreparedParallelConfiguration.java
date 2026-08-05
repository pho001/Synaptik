package io.github.pho001.synaptik.backend.cpu.execution;

/**
 * Immutable prepared range-dispatch configuration borrowed by portable CPU invocations.
 *
 * <p>This value records already-selected dispatch facts only. It neither owns workers nor chooses
 * parallelism, and it is safe to share between concurrent bindings because all state is final.</p>
 *
 * @param workerCount positive number of workers already owned by the CPU composition
 * @param minimumRangeSize positive minimum number of elements assigned to one range
 * @param deterministic whether stable range-index ordering is required
 */
record CpuPreparedParallelConfiguration(
        int workerCount, long minimumRangeSize, boolean deterministic) {
    /**
     * Validates one prepared parallel recipe without creating or owning workers.
     *
     * @param workerCount positive number of workers already owned by CPU composition
     * @param minimumRangeSize positive minimum number of elements assigned to one range
     * @param deterministic whether stable range-index ordering is required
     * @throws IllegalArgumentException if {@code workerCount} or {@code minimumRangeSize} is not
     *     positive, in declaration order
     */
    CpuPreparedParallelConfiguration {
        if (workerCount <= 0) {
            throw new IllegalArgumentException("workerCount must be positive");
        }
        if (minimumRangeSize <= 0) {
            throw new IllegalArgumentException("minimumRangeSize must be positive");
        }
    }
}
