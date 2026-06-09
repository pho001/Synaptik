package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.kernels.fused.Cpu1FusedElementwiseRangeRunner;

/**
 * Prepared generated-kernel handle produced during cpu1 fused prepare.
 */
public record Cpu1FusedCodegenKernel(
        Cpu1FusedCodegenClassSignature classSignature,
        String generatedClassName,
        Cpu1FusedElementwiseRangeRunner rangeRunner
) {
    public Cpu1FusedCodegenKernel {
        if (classSignature == null) {
            throw new IllegalArgumentException("classSignature cannot be null");
        }
        if (generatedClassName == null || generatedClassName.isBlank()) {
            throw new IllegalArgumentException("generatedClassName cannot be blank");
        }
        if (rangeRunner == null) {
            throw new IllegalArgumentException("rangeRunner cannot be null");
        }
    }

    public void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        rangeRunner.computeRange(args, startInclusive, endExclusive);
    }
}
