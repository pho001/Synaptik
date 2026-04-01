package graph.codegen;

public record FusedKernelCacheKey(
        String signature,
        int precisionMode
) {}
