package backend.cpu1.kernels.fused.codegen;

/**
 * Prepare-time reason why a cpu1 fused expression cannot receive an ASM generated kernel.
 */
public enum Cpu1FusedCodegenRejectionReason {
    NONE("supported"),
    UNSUPPORTED_DTYPE("unsupported dtype"),
    UNSUPPORTED_OPERATION("unsupported operation"),
    UNSUPPORTED_INTRINSIC("unsupported intrinsic"),
    UNSUPPORTED_LAYOUT_OR_ACCESS("unsupported layout or access pattern"),
    UNSUPPORTED_STORAGE_KIND("unsupported storage kind"),
    UNSUPPORTED_SEGMENT_VECTOR("MemorySegment vector codegen is not implemented"),
    UNSUPPORTED_BF16_VECTOR("BF16 vector codegen is not implemented"),
    UNSUPPORTED_BF16_SEGMENT("BF16 MemorySegment codegen is not implemented"),
    UNSUPPORTED_BF16_OPERATION("unsupported BF16 operation"),
    UNSUPPORTED_LOOP_KIND("unsupported loop kind"),
    UNSUPPORTED_VECTOR_OPERATION("unsupported vector operation"),
    UNSUPPORTED_SCALAR_BINDING("unsupported scalar binding");

    private final String description;

    Cpu1FusedCodegenRejectionReason(String description) {
        this.description = description;
    }

    public String description() {
        return description;
    }
}
