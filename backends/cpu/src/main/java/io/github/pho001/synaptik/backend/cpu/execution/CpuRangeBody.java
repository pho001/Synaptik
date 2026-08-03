package io.github.pho001.synaptik.backend.cpu.execution;

/**
 * Executes one non-empty half-open element range assigned by a {@link CpuWorkerGroup}.
 * Implementations may run concurrently on different owned workers and must cooperate with
 * cancellation at range boundaries; the interface owns no state or thread.
 */
@FunctionalInterface
interface CpuRangeBody {
    /**
     * Executes one assigned range.
     * @param startInclusive inclusive first element index
     * @param endExclusive exclusive end element index
     * @param rangeIndex stable zero-based index of the predetermined range geometry
     */
    void execute(long startInclusive, long endExclusive, int rangeIndex);
}
