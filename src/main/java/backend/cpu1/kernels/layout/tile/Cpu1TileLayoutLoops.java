package backend.cpu1.kernels.layout.tile;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;

public final class Cpu1TileLayoutLoops {
    private Cpu1TileLayoutLoops() {
    }

    public static void tileScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        int[] outputShape = output.shape();
        int[] outputDense = Cpu1LayoutKernelSupport.denseStrides(outputShape);
        support.launchRange(output.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                int remaining = logical;
                int inputOffset = input.storageOffset();
                int outputOffset = output.storageOffset();
                for (int dim = 0; dim < outputShape.length; dim++) {
                    int coordinate = remaining / outputDense[dim];
                    remaining %= outputDense[dim];
                    inputOffset += (coordinate % input.shape(dim)) * input.stride(dim);
                    outputOffset += coordinate * output.stride(dim);
                }
                support.writeElement(output, outputOffset, support.readElement(input, inputOffset));
            }
        });
        support.markOutputWritten(call);
    }

    public static void tileLastAxisBlockScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyLastAxisBlock(unit, context, false);
    }

    public static void tileLastAxisBlockVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyLastAxisBlock(unit, context, true);
    }

    public static void tileAxis0BlockScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyAxis0Block(unit, context, false);
    }

    public static void tileAxis0BlockVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        copyAxis0Block(unit, context, true);
    }

    public static void tileDenseBlockRepeatScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        if (output.elementCount() == 0) {
            support.markOutputWritten(call);
            return;
        }
        DenseBlockRepeatGeometry geometry = denseBlockRepeatGeometry(input, output);
        support.launchRange(geometry.outerBlocks(), (start, end) -> {
            for (int outer = start; outer < end; outer++) {
                int inputBase = outer * geometry.inputBlockElements();
                int outputBase = outer * geometry.outputBlockElements();
                for (int repeat = 0; repeat < geometry.repeats(); repeat++) {
                    support.copyDenseBlockScalar(
                            input,
                            inputBase,
                            output,
                            outputBase + repeat * geometry.inputBlockElements(),
                            geometry.inputBlockElements()
                    );
                }
            }
        });
        support.markOutputWritten(call);
    }

    public static void tileDenseBlockRepeatVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        if (output.elementCount() == 0) {
            support.markOutputWritten(call);
            return;
        }
        DenseBlockRepeatGeometry geometry = denseBlockRepeatGeometry(input, output);
        support.launchRange(geometry.outerBlocks(), (start, end) -> {
            for (int outer = start; outer < end; outer++) {
                int inputBase = outer * geometry.inputBlockElements();
                int outputBase = outer * geometry.outputBlockElements();
                for (int repeat = 0; repeat < geometry.repeats(); repeat++) {
                    support.copyDenseBlockVector(
                            input,
                            inputBase,
                            output,
                            outputBase + repeat * geometry.inputBlockElements(),
                            geometry.inputBlockElements()
                    );
                }
            }
        });
        support.markOutputWritten(call);
    }

    private static void copyAxis0Block(
            Cpu1PreparedLayoutUnit unit,
            ExecutionContext context,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        int block = input.elementCount();
        int repeats = output.elementCount() / block;
        support.launchRange(repeats, (start, end) -> {
            for (int repeat = start; repeat < end; repeat++) {
                int outputBase = repeat * block;
                if (vectorized) {
                    support.copyDenseBlockVector(input, 0, output, outputBase, block);
                } else {
                    support.copyDenseBlockScalar(input, 0, output, outputBase, block);
                }
            }
        });
        support.markOutputWritten(call);
    }

    private static void copyLastAxisBlock(
            Cpu1PreparedLayoutUnit unit,
            ExecutionContext context,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        int rank = input.rank();
        int inner = input.shape(rank - 1);
        int outputInner = output.shape(rank - 1);
        int repeats = outputInner / inner;
        int rows = input.elementCount() / inner;
        support.launchRange(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                int inputBase = row * inner;
                int outputBase = row * outputInner;
                for (int repeat = 0; repeat < repeats; repeat++) {
                    if (vectorized) {
                        support.copyDenseBlockVector(input, inputBase, output, outputBase + repeat * inner, inner);
                    } else {
                        support.copyDenseBlockScalar(input, inputBase, output, outputBase + repeat * inner, inner);
                    }
                }
            }
        });
        support.markOutputWritten(call);
    }

    private static DenseBlockRepeatGeometry denseBlockRepeatGeometry(Cpu1TensorView input, Cpu1TensorView output) {
        int axis = repeatedMiddleAxis(input, output);
        int inputBlockElements = 1;
        for (int dim = axis; dim < input.rank(); dim++) {
            inputBlockElements = Math.multiplyExact(inputBlockElements, input.shape(dim));
        }
        int repeats = output.shape(axis) / input.shape(axis);
        int outputBlockElements = Math.multiplyExact(inputBlockElements, repeats);
        int outerBlocks = input.elementCount() / inputBlockElements;
        return new DenseBlockRepeatGeometry(inputBlockElements, outputBlockElements, repeats, outerBlocks);
    }

    private static int repeatedMiddleAxis(Cpu1TensorView input, Cpu1TensorView output) {
        int rank = input.rank();
        if (rank != output.rank()) {
            throw new IllegalStateException("cpu1 TILE dense block repeat rank mismatch.");
        }
        int repeatedAxis = -1;
        for (int dim = 0; dim < rank; dim++) {
            int inputDim = input.shape(dim);
            int outputDim = output.shape(dim);
            if (inputDim == outputDim) {
                continue;
            }
            if (repeatedAxis != -1
                    || dim == 0
                    || dim == rank - 1
                    || inputDim <= 0
                    || outputDim % inputDim != 0) {
                throw new IllegalStateException("cpu1 TILE dense block repeat requires exactly one middle repeated axis.");
            }
            repeatedAxis = dim;
        }
        if (repeatedAxis == -1) {
            throw new IllegalStateException("cpu1 TILE dense block repeat found no repeated middle axis.");
        }
        return repeatedAxis;
    }

    private record DenseBlockRepeatGeometry(
            int inputBlockElements,
            int outputBlockElements,
            int repeats,
            int outerBlocks
    ) {
    }
}
