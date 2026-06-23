package backend.cpu1.kernels.layout.concat;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;

public final class Cpu1ConcatLayoutLoops {
    private Cpu1ConcatLayoutLoops() {
    }

    public static void concatScalar(Cpu1LayoutKernelSupport support) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        int axis = support.unit().axis();
        int axisOffset = 0;
        for (Cpu1TensorView input : call.inputs()) {
            copyConcatInputScalar(support, input, call.output(), axis, axisOffset);
            axisOffset += input.shape(axis);
        }
        support.markOutputWritten(call);
    }

    public static void concatAxis0BlockScalar(Cpu1LayoutKernelSupport support) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        int outputOffset = 0;
        for (Cpu1TensorView input : call.inputs()) {
            support.copyDenseBlockScalar(input, 0, call.output(), outputOffset, input.elementCount());
            outputOffset += input.elementCount();
        }
        support.markOutputWritten(call);
    }

    public static void concatAxis0BlockVector(Cpu1LayoutKernelSupport support) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        int outputOffset = 0;
        for (Cpu1TensorView input : call.inputs()) {
            support.copyDenseBlockVector(input, 0, call.output(), outputOffset, input.elementCount());
            outputOffset += input.elementCount();
        }
        support.markOutputWritten(call);
    }

    public static void concatInnerAxisBlockScalar(Cpu1LayoutKernelSupport support) {
        copyInnerAxisBlock(support, false);
    }

    public static void concatInnerAxisBlockVector(Cpu1LayoutKernelSupport support) {
        copyInnerAxisBlock(support, true);
    }

    public static void concatMiddleAxisBlockScalar(Cpu1LayoutKernelSupport support) {
        copyMiddleAxisBlock(support, false);
    }

    public static void concatMiddleAxisBlockVector(Cpu1LayoutKernelSupport support) {
        copyMiddleAxisBlock(support, true);
    }

    private static void copyInnerAxisBlock(
            Cpu1LayoutKernelSupport support,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView output = call.output();
        int axis = support.unit().axis();
        int rows = output.elementCount() / output.shape(axis);
        int outputBlock = output.shape(axis);
        int axisOffset = 0;
        for (Cpu1TensorView input : call.inputs()) {
            int inputBlock = input.shape(axis);
            int blockOutputOffset = axisOffset;
            support.launchRange(rows, (start, end) -> {
                for (int row = start; row < end; row++) {
                    if (vectorized) {
                        support.copyDenseBlockVector(
                                input,
                                row * inputBlock,
                                output,
                                row * outputBlock + blockOutputOffset,
                                inputBlock
                        );
                    } else {
                        support.copyDenseBlockScalar(
                                input,
                                row * inputBlock,
                                output,
                                row * outputBlock + blockOutputOffset,
                                inputBlock
                        );
                    }
                }
            });
            axisOffset += inputBlock;
        }
        support.markOutputWritten(call);
    }

    private static void copyMiddleAxisBlock(
            Cpu1LayoutKernelSupport support,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView output = call.output();
        int axis = support.unit().axis();
        int inner = denseInnerSize(output, axis);
        int outputAxisBlock = output.shape(axis) * inner;
        int outerRows = output.elementCount() / outputAxisBlock;
        int axisOffset = 0;
        for (Cpu1TensorView input : call.inputs()) {
            int inputAxisBlock = input.shape(axis) * inner;
            int blockOutputOffset = axisOffset * inner;
            support.launchRange(outerRows, (start, end) -> {
                for (int row = start; row < end; row++) {
                    int inputOffset = row * inputAxisBlock;
                    int outputOffset = row * outputAxisBlock + blockOutputOffset;
                    if (vectorized) {
                        support.copyDenseBlockVector(input, inputOffset, output, outputOffset, inputAxisBlock);
                    } else {
                        support.copyDenseBlockScalar(input, inputOffset, output, outputOffset, inputAxisBlock);
                    }
                }
            });
            axisOffset += input.shape(axis);
        }
        support.markOutputWritten(call);
    }

    private static void copyConcatInputScalar(
            Cpu1LayoutKernelSupport support,
            Cpu1TensorView input,
            Cpu1TensorView output,
            int axis,
            int axisOffset
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
                    outputOffset += (dim == axis ? coordinate + axisOffset : coordinate) * output.stride(dim);
                }
                support.writeElement(output, outputOffset, support.readElement(input, inputOffset));
            }
        });
    }

    private static int denseInnerSize(Cpu1TensorView view, int axis) {
        int inner = 1;
        for (int dim = axis + 1; dim < view.rank(); dim++) {
            inner = Math.multiplyExact(inner, view.shape(dim));
        }
        return inner;
    }
}
