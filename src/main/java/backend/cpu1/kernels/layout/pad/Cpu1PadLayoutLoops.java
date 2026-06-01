package backend.cpu1.kernels.layout.pad;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;
import operations.layout.pad;

public final class Cpu1PadLayoutLoops {
    private Cpu1PadLayoutLoops() {
    }

    public static void padScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyPad(unit, context, false, false);
    }

    public static void padVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyPad(unit, context, true, false);
    }

    public static void padDenseInnerBlockScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyPad(unit, context, false, true);
    }

    public static void padDenseInnerBlockVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyPad(unit, context, true, true);
    }

    private static void copyPad(
            Cpu1PreparedLayoutUnit unit,
            ExecutionContext context,
            boolean vectorized,
            boolean denseInnerBlock
    ) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        pad padOp = requirePad(support);
        if (vectorized) {
            support.fillOutputVector(output, padOp.getConstantValue());
        } else {
            support.fillOutputScalar(output, padOp.getConstantValue());
        }
        if (denseInnerBlock) {
            copyDenseInnerBlock(support, input, output, padOp.getBefore(), vectorized);
        } else {
            copyGeneric(support, input, output, padOp.getBefore());
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

    private static pad requirePad(Cpu1LayoutKernelSupport support) {
        if (support.context().runtimeTensorForNodeId(support.unit().nodeId()).getOperation() instanceof pad padOp) {
            return padOp;
        }
        throw new IllegalArgumentException("cpu1 PAD requires operations.layout.pad.");
    }
}
