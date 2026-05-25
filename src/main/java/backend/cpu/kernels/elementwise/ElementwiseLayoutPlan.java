package backend.cpu.kernels.elementwise;

import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class ElementwiseLayoutPlan {
    private final int[] shape;
    private final int length;
    private final Operand output;
    private final List<Operand> inputs;

    private ElementwiseLayoutPlan(int[] shape, int length, Operand output, List<Operand> inputs) {
        this.shape = shape.clone();
        this.length = length;
        this.output = output;
        this.inputs = List.copyOf(inputs);
    }

    public static ElementwiseLayoutPlan binary(Tensor left, Tensor right, Tensor output) {
        int[] shape = output.getShapeUnsafe();
        return new ElementwiseLayoutPlan(
                shape,
                output.getFlatDataSize(),
                outputOperand(output),
                List.of(inputOperand(left, shape), inputOperand(right, shape))
        );
    }

    private static Operand outputOperand(Tensor tensor) {
        return new Operand(tensor.getStorageOffsetUnsafe(), tensor.getStridesUnsafe());
    }

    public static Operand inputOperand(Tensor tensor, int[] outputShape) {
        return new Operand(
                tensor.getStorageOffsetUnsafe(),
                broadcastStrides(tensor.getShapeUnsafe(), tensor.getStridesUnsafe(), outputShape)
        );
    }

    public static int[] broadcastStrides(int[] inputShape, int[] inputStrides, int[] outputShape) {
        int outRank = outputShape.length;
        int inRank = inputShape.length;
        if (inRank > outRank) {
            throw new IllegalArgumentException("Input rank " + inRank + " cannot broadcast to output rank " + outRank);
        }
        int[] aligned = new int[outRank];
        int rankOffset = outRank - inRank;
        for (int outDim = 0; outDim < outRank; outDim++) {
            if (outDim < rankOffset) {
                aligned[outDim] = 0;
                continue;
            }
            int inputDim = outDim - rankOffset;
            int inSize = inputShape[inputDim];
            int outSize = outputShape[outDim];
            if (inSize == outSize) {
                aligned[outDim] = inputStrides[inputDim];
            } else if (inSize == 1 && outSize >= 1) {
                aligned[outDim] = 0;
            } else {
                throw new IllegalArgumentException(
                        "Input shape " + Arrays.toString(inputShape)
                                + " cannot broadcast to output shape " + Arrays.toString(outputShape)
                );
            }
        }
        return aligned;
    }

    public static boolean canBroadcastTo(int[] inputShape, int[] outputShape) {
        if (inputShape == null || outputShape == null || inputShape.length > outputShape.length) {
            return false;
        }
        int offset = outputShape.length - inputShape.length;
        for (int i = 0; i < outputShape.length; i++) {
            int inputDim = i < offset ? 1 : inputShape[i - offset];
            int outputDim = outputShape[i];
            if (inputDim != outputDim && inputDim != 1) {
                return false;
            }
        }
        return true;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int length() {
        return length;
    }

    public int[][] cursorStrides() {
        int[][] strides = new int[inputs.size() + 1][];
        strides[0] = output.strides();
        for (int i = 0; i < inputs.size(); i++) {
            strides[i + 1] = inputs.get(i).strides();
        }
        return strides;
    }

    public int[] cursorBaseOffsets() {
        int[] offsets = new int[inputs.size() + 1];
        offsets[0] = output.baseOffset();
        for (int i = 0; i < inputs.size(); i++) {
            offsets[i + 1] = inputs.get(i).baseOffset();
        }
        return offsets;
    }

    public record Operand(int baseOffset, int[] strides) {
        public Operand {
            strides = strides.clone();
        }
    }
}
