package backend.cpu1.kernels.fused.codegen;

/**
 * Prepare-time factory for cpu1 fused generated kernels.
 */
public final class Cpu1FusedCodegenKernelFactory {
    private Cpu1FusedCodegenKernelFactory() {
    }

    public static Cpu1FusedCodegenKernel prepareKernel(Cpu1FusedCodegenPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        Cpu1FusedCodegenRejectionReason rejectionReason = plan.rejectionReason();
        if (rejectionReason != Cpu1FusedCodegenRejectionReason.NONE) {
            throw rejection(plan, rejectionReason);
        }
        throw rejection(plan, Cpu1FusedCodegenRejectionReason.MISSING_ASM_EMITTER);
    }

    public static UnsupportedOperationException rejection(
            Cpu1FusedCodegenPlan plan,
            Cpu1FusedCodegenRejectionReason rejectionReason
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (rejectionReason == null) {
            throw new IllegalArgumentException("rejectionReason cannot be null");
        }
        return new UnsupportedOperationException("cpu1 fused ASM codegen rejected: "
                + rejectionReason + " (" + rejectionReason.description() + "), signature="
                + plan.classSignature().canonicalSignature());
    }
}
