package tensor.ops.linalg;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class LinalgSupport {
    private LinalgSupport() {
    }

    static void requireFloating(Tensor tensor, String name) {
        if (tensor == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    static DataType promote(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    static int[] broadcastLeadingShape(int[] first, int[] second, String errorMessage) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int a = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int b = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (a != b && a != 1 && b != 1) {
                throw new IllegalArgumentException(errorMessage);
            }
            out[i] = Math.max(a, b);
        }
        return out;
    }

    static Tensor transposeLastTwoAxes(Tensor tensor) {
        int rank = tensor.getShapeUnsafe().length;
        if (rank == 2) {
            return tensor.transpose();
        }
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        int tmp = axes[rank - 1];
        axes[rank - 1] = axes[rank - 2];
        axes[rank - 2] = tmp;
        return tensor.permute(axes);
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }

    static Tensor sumToShape(Tensor gradOut, int[] targetShape) {
        int[] outShape = gradOut.getShape();
        int[] normalizedTarget = targetShape.length == 0 ? new int[]{1} : targetShape.clone();
        if (Arrays.equals(outShape, normalizedTarget)) {
            return gradOut;
        }

        int outRank = outShape.length;
        int inRank = normalizedTarget.length;
        if (inRank > outRank) {
            throw new IllegalArgumentException("Target rank cannot exceed grad rank. target="
                    + Arrays.toString(normalizedTarget) + ", grad=" + Arrays.toString(outShape));
        }

        int[] alignedTarget = new int[outRank];
        int offset = outRank - inRank;
        for (int d = 0; d < outRank; d++) {
            alignedTarget[d] = d < offset ? 1 : normalizedTarget[d - offset];
        }

        List<Integer> reduceAxes = new ArrayList<>();
        for (int d = 0; d < outRank; d++) {
            int td = alignedTarget[d];
            int od = outShape[d];
            if (td != od) {
                if (td != 1) {
                    throw new IllegalArgumentException("Incompatible target shape for broadcast reduction. target="
                            + Arrays.toString(normalizedTarget) + ", grad=" + Arrays.toString(outShape));
                }
                reduceAxes.add(d);
            }
        }

        Tensor reduced = gradOut;
        for (int i = reduceAxes.size() - 1; i >= 0; i--) {
            reduced = reduced.sum(reduceAxes.get(i));
        }

        if (reduced.getDataType() != gradOut.getDataType()) {
            reduced.setDataType(gradOut.getDataType());
        }
        if (!Arrays.equals(reduced.getShape(), normalizedTarget)) {
            reduced = reduced.reshape(normalizedTarget);
        }
        return reduced;
    }
}
