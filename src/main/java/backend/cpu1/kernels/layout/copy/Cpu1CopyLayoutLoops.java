package backend.cpu1.kernels.layout.copy;

import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.exec.Cpu1TensorView;

public final class Cpu1CopyLayoutLoops {
    private Cpu1CopyLayoutLoops() {
    }

    public static void reshapeCopyLinearizedScalar(Cpu1LayoutKernelSupport support) {
        support.copyLinearizedScalar();
    }

    public static void contiguousCopyScalar(Cpu1LayoutKernelSupport support) {
        support.copyContiguousScalar();
    }

    public static void contiguousCopyVector(Cpu1LayoutKernelSupport support) {
        support.copyContiguousVector();
    }

    public static void contiguousOffsetDenseBlockScalar(Cpu1LayoutKernelSupport support) {
        copyOffsetDenseBlock(support, false);
    }

    public static void contiguousOffsetDenseBlockVector(Cpu1LayoutKernelSupport support) {
        copyOffsetDenseBlock(support, true);
    }

    private static void copyOffsetDenseBlock(
            Cpu1LayoutKernelSupport support,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        if (vectorized) {
            support.copyDenseBlockVector(
                    input,
                    input.storageOffset(),
                    output,
                    output.storageOffset(),
                    output.elementCount()
            );
        } else {
            support.copyDenseBlockScalar(
                    input,
                    input.storageOffset(),
                    output,
                    output.storageOffset(),
                    output.elementCount()
            );
        }
        support.markOutputWritten(call);
    }
}
