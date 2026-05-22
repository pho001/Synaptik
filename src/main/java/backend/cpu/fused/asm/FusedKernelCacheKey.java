package backend.cpu.fused.asm;

/**
 * Cache key for generated fused ASM executable constructors.
 */
public record FusedKernelCacheKey(
        String signature,
        String numericContract,
        String approximationContract,
        int vectorWidth,
        FusedAsmSpecializationKind specializationKind
) {}
