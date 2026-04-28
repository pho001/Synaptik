package backend.cpu.fused.codegen;

/**
 * Cache key for generated fused ASM executable constructors.
 */
public record FusedKernelCacheKey(
        String signature,
        int precisionMode,
        int vectorWidth,
        FusedAsmSpecializationKind specializationKind
) {}
