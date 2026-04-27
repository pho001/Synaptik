package backend.cpu.fused.codegen;

public enum FusedAsmSpecializationKind {
    NONE("generic"),
    F32_MASKED_SCALE_WHERE("f32MaskedScaleWhere"),
    F32_MASKED_SCALE_WHERE_INVERTED("f32MaskedScaleWhereInverted");

    private final String cacheToken;

    FusedAsmSpecializationKind(String cacheToken) {
        this.cacheToken = cacheToken;
    }

    public String cacheToken() {
        return cacheToken;
    }
}
