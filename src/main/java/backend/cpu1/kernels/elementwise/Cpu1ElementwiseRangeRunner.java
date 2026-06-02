package backend.cpu1.kernels.elementwise;

import backend.cpu1.exec.Cpu1KernelArgs;

/**
 * Prepared range runner for one concrete cpu1 elementwise loop.
 */
@FunctionalInterface
public interface Cpu1ElementwiseRangeRunner {
    void computeRange(Cpu1KernelArgs args, int startInclusive, int endExclusive);
}
