package backend.cpu.fused.codegen;

public record FusedKernelCacheKey(
        String signature,
        int precisionMode,
        int vectorWidth,
        FusedAsmSpecializationKind specializationKind
) {}
