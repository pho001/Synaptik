package backend.cpu.fused.asm.emit;

/**
 * Internal JVM descriptors shared by fused ASM emitters.
 */
public final class FusedMethodDescriptors {
    private FusedMethodDescriptors() {}

    public static final String RANGE_METHOD_DESC =
            "(Ljava/util/List;Ltensor/Tensor;Lbackend/cpu/kernels/CpuKernelContext;II)V";
}
