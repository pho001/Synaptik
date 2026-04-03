package tensor;

import operations.Operation;
import operations.contiguous;
import operations.expand;
import operations.expandDims;
import operations.permute;
import operations.reshape;
import operations.squeeze;

import java.util.List;

final class TensorLayoutOps {
    private TensorLayoutOps() {}

    static Tensor contiguous(Tensor input) {
        Operation op = new contiguous();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "contiguous", input.getDataType());
        return out;
    }

    static Tensor reshape(Tensor input, int[] requestedShape) {
        int[] newShape = TensorLayoutTransform.inferReshape(input.getShape(), requestedShape);
        Operation op = new reshape(newShape);
        Tensor out = input.isContiguous()
                ? new Tensor(newShape, TensorMetadata.computeStrides(newShape), input.getStorageOffsetUnsafe(), List.of(input), op, "reshape", input.getDataType())
                : new Tensor(newShape, List.of(input), op, "reshape", input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.reshape(input.getShape());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor expand(Tensor input, int[] requestedShape) {
        int[] targetShape = TensorLayoutTransform.inferExpandShape(input.getShape(), requestedShape);
        int[] targetStrides = buildExpandedStrides(input.getShapeUnsafe(), input.getStridesUnsafe(), targetShape);
        Operation op = new expand(targetShape);
        Tensor out = new Tensor(targetShape, targetStrides, input.getStorageOffsetUnsafe(), List.of(input), op, "expand", input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = TensorBroadcastOps.sumToShape(outGrad, input.getShape());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    private static int[] buildExpandedStrides(int[] sourceShape, int[] sourceStrides, int[] targetShape) {
        int targetRank = targetShape.length;
        int sourceRank = sourceShape.length;
        int rankOffset = targetRank - sourceRank;
        int[] outStrides = new int[targetRank];

        for (int d = 0; d < targetRank; d++) {
            int sourceDim = d - rankOffset;
            if (sourceDim < 0) {
                outStrides[d] = 0;
                continue;
            }
            outStrides[d] = sourceShape[sourceDim] == 1 && targetShape[d] != 1
                    ? 0
                    : sourceStrides[sourceDim];
        }
        return outStrides;
    }

    static Tensor permute(Tensor input, int[] axes) {
        int rank = input.getShape().length;
        int[] normalizedAxes = TensorLayoutTransform.normalizeAxes(rank, axes);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStrides();
        int[] outShape = new int[rank];
        int[] outStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            outShape[i] = inShape[normalizedAxes[i]];
            outStrides[i] = inStrides[normalizedAxes[i]];
        }

        Operation op = new permute(normalizedAxes);
        Tensor out = new Tensor(outShape, outStrides, input.getStorageOffsetUnsafe(), List.of(input), op, "permute", input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            int[] inverse = TensorLayoutTransform.inverseAxes(normalizedAxes);
            Tensor grad = outGrad.permute(inverse);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor expandDims(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeInsertAxis(axis, rank);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStridesUnsafe();
        int[] outShape = new int[rank + 1];
        int[] outStrides = new int[rank + 1];
        for (int i = 0, j = 0; i < outShape.length; i++) {
            if (i == normalizedAxis) {
                outShape[i] = 1;
                outStrides[i] = insertedAxisStride(inShape, inStrides, normalizedAxis);
            } else {
                outShape[i] = inShape[j];
                outStrides[i] = inStrides[j];
                j++;
            }
        }
        Operation op = new expandDims(normalizedAxis);
        Tensor out = new Tensor(outShape, outStrides, input.getStorageOffsetUnsafe(), List.of(input), op, "expandDims", input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.squeeze(normalizedAxis);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor squeeze(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, rank);
        if (input.getShape()[normalizedAxis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze dimension " + normalizedAxis + " with size " + input.getShape()[normalizedAxis]);
        }
        int[] inShape = input.getShape();
        int[] inStrides = input.getStridesUnsafe();
        int[] outShape = new int[rank - 1];
        int[] outStrides = new int[rank - 1];
        for (int i = 0, j = 0; i < inShape.length; i++) {
            if (i != normalizedAxis) {
                outShape[j] = inShape[i];
                outStrides[j] = inStrides[i];
                j++;
            }
        }
        Operation op = new squeeze(normalizedAxis);
        Tensor out = new Tensor(outShape, outStrides, input.getStorageOffsetUnsafe(), List.of(input), op, "squeeze", input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.expandDims(normalizedAxis);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    private static int insertedAxisStride(int[] shape, int[] strides, int axis) {
        if (axis >= shape.length) {
            return 1;
        }
        return strides[axis] * shape[axis];
    }
}
