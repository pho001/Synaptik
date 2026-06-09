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
    UNSUPPORTED_LOOP_KIND("unsupported loop kind"),
    UNSUPPORTED_SCALAR_BINDING("unsupported scalar binding");

    private final String description;

    Cpu1FusedCodegenRejectionReason(String description) {
        this.description = description;
    }

    public String description() {
        return description;
    }
}
