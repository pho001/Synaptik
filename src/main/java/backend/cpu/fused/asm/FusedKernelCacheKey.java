package backend.cpu.fused.asm;

/**
 * Cache key for generated fused ASM executable constructors.
 */
public record FusedKernelCacheKey(
        String signature,
        String numericContract,
        int vectorWidth,
        FusedAsmSpecializationKind specializationKind
) {}
