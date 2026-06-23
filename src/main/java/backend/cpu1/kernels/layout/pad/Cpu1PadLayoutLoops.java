package backend.cpu1.kernels.layout.pad;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;

public final class Cpu1PadLayoutLoops {
    private Cpu1PadLayoutLoops() {
    }

    public static void padScalar(Cpu1LayoutKernelSupport support) {
        copyPad(support, false, false);
    }

    public static void padVector(Cpu1LayoutKernelSupport support) {
        copyPad(support, true, false);
    }

    public static void padDenseInnerBlockScalar(Cpu1LayoutKernelSupport support) {
        copyPad(support, false, true);
    }

    public static void padDenseInnerBlockVector(Cpu1LayoutKernelSupport support) {
        copyPad(support, true, true);
    }

    private static void copyPad(
            Cpu1LayoutKernelSupport support,
            boolean vectorized,
            boolean denseInnerBlock
    ) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        int[] before = support.unit().padBefore();
        if (vectorized) {
            support.fillOutputVector(output, support.unit().padConstantValue());
        } else {
            support.fillOutputScalar(output, support.unit().padConstantValue());
        }
        if (denseInnerBlock) {
            copyDenseInnerBlock(support, input, output, before, vectorized);
        } else {
            copyGeneric(support, input, output, before);
        }
        support.markOutputWritten(call);
    }

    private static void copyGeneric(
            Cpu1LayoutKernelSupport support,
            Cpu1TensorView input,
            Cpu1TensorView output,
            int[] before
    ) {
        int[] inputShape = input.shape();
        int[] inputDense = Cpu1LayoutKernelSupport.denseStrides(inputShape);
        support.launchRange(input.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                int remaining = logical;
                int inputOffset = input.storageOffset();
                int outputOffset = output.storageOffset();
                for (int dim = 0; dim < inputShape.length; dim++) {
                    int coordinate = remaining / inputDense[dim];
                    remaining %= inputDense[dim];
                    inputOffset += coordinate * input.stride(dim);
                    outputOffset += (coordinate + before[dim]) * output.stride(dim);
                }
                support.writeElement(output, outputOffset, support.readElement(input, inputOffset));
            }
        });
    }

    private static void copyDenseInnerBlock(
            Cpu1LayoutKernelSupport support,
            Cpu1TensorView input,
            Cpu1TensorView output,
            int[] before,
            boolean vectorized
    ) {
        int rank = input.rank();
        int inner = input.shape(rank - 1);
        int rows = input.elementCount() / inner;
        int[] outerShape = input.shape();
        support.launchRange(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                int outputBase = output.storageOffset() + before[rank - 1];
                if (rank > 1) {
                    outputBase += paddedOuterOffset(row, outerShape, before, output);
                }
                if (vectorized) {
                    support.copyDenseBlockVector(input, row * inner, output, outputBase, inner);
                } else {
                    support.copyDenseBlockScalar(input, row * inner, output, outputBase, inner);
                }
            }
        });
    }

    private static int paddedOuterOffset(int row, int[] inputShape, int[] before, Cpu1TensorView output) {
        int remaining = row;
        int outputOffset = 0;
        for (int dim = inputShape.length - 2; dim >= 0; dim--) {
            int coordinate = remaining % inputShape[dim];
            remaining /= inputShape[dim];
            outputOffset += (coordinate + before[dim]) * output.stride(dim);
        }
        return outputOffset;
    }

}
