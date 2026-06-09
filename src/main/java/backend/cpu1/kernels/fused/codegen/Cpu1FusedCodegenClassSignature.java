package backend.cpu1.kernels.fused.codegen;

/**
 * Structural generated-class identity. It excludes graph node ids, unit ids, and concrete scalar values.
 */
public record Cpu1FusedCodegenClassSignature(String canonicalSignature) {
    public Cpu1FusedCodegenClassSignature {
        if (canonicalSignature == null || canonicalSignature.isBlank()) {
            throw new IllegalArgumentException("canonicalSignature cannot be blank");
        }
    }
}
