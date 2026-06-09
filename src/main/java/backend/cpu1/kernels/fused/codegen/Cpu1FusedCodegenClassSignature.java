package backend.cpu1.kernels.fused.codegen;

import java.util.List;

/**
 * Structural generated-class identity. It excludes graph node ids, unit ids, and concrete scalar values.
 */
public record Cpu1FusedCodegenClassSignature(
        String canonicalSignature,
        int supportAbiVersion,
        List<String> helperDependencies
) {
    public Cpu1FusedCodegenClassSignature {
        if (canonicalSignature == null || canonicalSignature.isBlank()) {
            throw new IllegalArgumentException("canonicalSignature cannot be blank");
        }
        if (supportAbiVersion < 0) {
            throw new IllegalArgumentException("supportAbiVersion must be >= 0");
        }
        if (helperDependencies == null) {
            throw new IllegalArgumentException("helperDependencies cannot be null");
        }
        helperDependencies = List.copyOf(helperDependencies);
    }
}
