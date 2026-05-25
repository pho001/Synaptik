package backend.cpu.fused.plan;

public enum FusedVectorBlockReason {
    NONE(false, false),
    MEMORY_SEGMENT_SCALAR_ONLY(false, true),
    MASKED_SCALE_WHERE_SCALAR_ONLY(true, true),
    BF16_STRIDED_RATIONAL_SCALAR_ONLY(false, true),
    UNSUPPORTED_ALLOCATION_FREE_VECTOR_PATH(false, true),
    PREFERRED_WIDTH_SCALAR(false, false),
    BELOW_VECTOR_THRESHOLD(false, false);

    private final boolean forceSerialScalarDispatch;
    private final boolean forceScalarAsmWidth;

    FusedVectorBlockReason(boolean forceSerialScalarDispatch, boolean forceScalarAsmWidth) {
        this.forceSerialScalarDispatch = forceSerialScalarDispatch;
        this.forceScalarAsmWidth = forceScalarAsmWidth;
    }

    public boolean forceSerialScalarDispatch() {
        return forceSerialScalarDispatch;
    }

    public boolean forceScalarAsmWidth() {
        return forceScalarAsmWidth;
    }
}
