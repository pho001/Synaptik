package backend.cpu1.kernels;

import backend.cpu1.exec.Cpu1KernelArgs;

/**
 * Prepared range runner for one concrete cpu1 loop.
 */
@FunctionalInterface
public interface Cpu1KernelRangeRunner {
    void computeRange(Cpu1KernelArgs args, int startInclusive, int endExclusive);
}
