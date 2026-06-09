package backend.cpu1.kernels.fused;

import backend.cpu1.exec.Cpu1FusedKernelArgs;

@FunctionalInterface
public interface Cpu1FusedElementwiseRangeRunner {
    void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive);
}
