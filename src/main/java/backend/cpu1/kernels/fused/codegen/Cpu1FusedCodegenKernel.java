package backend.cpu1.kernels.fused.codegen;

/**
 * Prepared generated-kernel handle produced during cpu1 fused prepare.
 */
public record Cpu1FusedCodegenKernel(
        Cpu1FusedCodegenClassSignature classSignature,
        String generatedClassName
) {
    public Cpu1FusedCodegenKernel {
        if (classSignature == null) {
            throw new IllegalArgumentException("classSignature cannot be null");
        }
        if (generatedClassName == null || generatedClassName.isBlank()) {
            throw new IllegalArgumentException("generatedClassName cannot be blank");
        }
    }
}
