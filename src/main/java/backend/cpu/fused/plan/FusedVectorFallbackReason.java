package backend.cpu.fused.plan;

public enum FusedVectorFallbackReason {
    NONE(false, false),
    MEMORY_SEGMENT_VECTOR_UNSUPPORTED(false, true),
    MASKED_SCALE_WHERE_VECTOR_DISABLED(true, true),
    BF16_STRIDED_RATIONAL_VECTOR_DISABLED(false, true),
    VECTOR_PATH_UNSUPPORTED(false, true),
    PREFERRED_WIDTH_IS_SCALAR(false, false),
    BELOW_VECTOR_THRESHOLD(false, false);

    private final boolean requiresSerialScalarDispatch;
    private final boolean requiresScalarAsmWidth;

    FusedVectorFallbackReason(boolean requiresSerialScalarDispatch, boolean requiresScalarAsmWidth) {
        this.requiresSerialScalarDispatch = requiresSerialScalarDispatch;
        this.requiresScalarAsmWidth = requiresScalarAsmWidth;
    }

    public boolean requiresSerialScalarDispatch() {
        return requiresSerialScalarDispatch;
    }

    public boolean requiresScalarAsmWidth() {
        return requiresScalarAsmWidth;
    }
}
