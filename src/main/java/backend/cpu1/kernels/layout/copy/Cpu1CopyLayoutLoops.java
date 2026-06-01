package backend.cpu1.kernels.layout.copy;

import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;
import backend.cpu1.exec.Cpu1TensorView;

public final class Cpu1CopyLayoutLoops {
    private Cpu1CopyLayoutLoops() {
    }

    public static void reshapeCopyLinearizedScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        new Cpu1LayoutKernelSupport(unit, context).copyLinearizedScalar();
    }

    public static void contiguousCopyScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        new Cpu1LayoutKernelSupport(unit, context).copyContiguousScalar();
    }

    public static void contiguousCopyVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        new Cpu1LayoutKernelSupport(unit, context).copyContiguousVector();
    }

    public static void contiguousOffsetDenseBlockScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyOffsetDenseBlock(unit, context, false);
    }

    public static void contiguousOffsetDenseBlockVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyOffsetDenseBlock(unit, context, true);
    }

    private static void copyOffsetDenseBlock(
            Cpu1PreparedLayoutUnit unit,
            ExecutionContext context,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
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
