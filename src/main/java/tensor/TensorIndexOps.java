package tensor;

import operations.select;
import operations.gather;
import operations.gatherGrad;
import operations.scatterAdd;
import operations.takeAlongAxis;
import operations.takeAlongAxisGrad;

import java.util.List;

final class TensorIndexOps {
    private TensorIndexOps() {
    }

    static Tensor select(Tensor input, int dimension, int index) {
        if (input == null) {
            throw new IllegalArgumentException("select input cannot be null");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int normalizedIndex = normalizeIndex(index, inputShape[normalizedDimension]);
        int[] outShape = reduceShape(inputShape, normalizedDimension);
        int[] outStrides = reduceStrides(input.getStridesUnsafe(), normalizedDimension);
        int outStorageOffset = input.getStorageOffsetUnsafe() + normalizedIndex * input.getStridesUnsafe()[normalizedDimension];

        Tensor out = new Tensor(
                outShape,
                outStrides,
                outStorageOffset,
                List.of(input),
                new select(normalizedDimension, normalizedIndex),
                "select",
                input.getDataType()
        );
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor zeroBase = Tensor.zerosLike(input);
            Tensor indices = constantIndexTensor(reduceShape(input.getShapeUnsafe(), normalizedDimension), normalizedIndex);
            Tensor grad = zeroBase.scatterAdd(indices, outGrad, normalizedDimension);
            accumulateGradient(input, grad);
        });
        return out;
    }

    static Tensor gather(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gather inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int[] outputShape = reduceShape(inputShape, normalizedDimension);
        validateGatherIndicesShape(indices.getShape(), outputShape);

        Tensor out = new Tensor(outputShape, List.of(input, indices), new gather(normalizedDimension), "gather", input.getDataType());
        out.setRequiresGrad(input.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor grad = new Tensor(input.getShape().clone(), List.of(indices, outGrad), new gatherGrad(normalizedDimension), "gather_grad", input.getDataType());
            accumulateGradient(input, grad);
        });
        return out;
    }

    static Tensor scatterAdd(Tensor base, Tensor indices, Tensor src, int dimension) {
        if (base == null || indices == null || src == null) {
            throw new IllegalArgumentException("scatterAdd inputs cannot be null");
        }
        if (base.getDataType() == DataType.BOOL || src.getDataType() == DataType.BOOL
                || base.getDataType() == DataType.INT32 || src.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("scatterAdd requires floating numeric base and source tensors.");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAdd indices must be numeric integral values.");
        }
        if (base.getDataType() != src.getDataType()) {
            throw new IllegalArgumentException("scatterAdd requires base and source tensors to have matching dtypes.");
        }
        int[] baseShape = base.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, baseShape.length);
        int[] expectedSrcShape = reduceShape(baseShape, normalizedDimension);
        validateGatherIndicesShape(indices.getShape(), expectedSrcShape);
        validateGatherIndicesShape(src.getShape(), expectedSrcShape);

        Tensor out = new Tensor(base.getShape().clone(), List.of(base, indices, src), new scatterAdd(normalizedDimension), "scatterAdd", base.getDataType());
        out.setRequiresGrad(base.getRequiresGrad() || src.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (base.getRequiresGrad()) {
                accumulateGradient(base, outGrad);
            }
            if (src.getRequiresGrad()) {
                accumulateGradient(src, outGrad.gather(indices, normalizedDimension));
            }
        });
        return out;
    }

    static Tensor takeAlongAxis(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("takeAlongAxis inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxis indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        validateTakeAlongAxisShape(inputShape, indices.getShape(), normalizedDimension);

        Tensor out = new Tensor(indices.getShape().clone(), List.of(input, indices), new takeAlongAxis(normalizedDimension), "takeAlongAxis", input.getDataType());
        out.setRequiresGrad(input.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor grad = new Tensor(input.getShape().clone(), List.of(indices, outGrad), new takeAlongAxisGrad(normalizedDimension), "take_along_axis_grad", input.getDataType());
            accumulateGradient(input, grad);
        });
        return out;
    }

    private static void validateGatherIndicesShape(int[] indicesShape, int[] expectedShape) {
        if (indicesShape.length != expectedShape.length) {
            throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
        }
        for (int i = 0; i < indicesShape.length; i++) {
            if (indicesShape[i] != expectedShape[i]) {
                throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
            }
        }
    }

    private static void validateTakeAlongAxisShape(int[] inputShape, int[] indicesShape, int axis) {
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis indices rank must match input rank.");
        }
        for (int i = 0; i < inputShape.length; i++) {
            if (i == axis) {
                continue;
            }
            if (indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("takeAlongAxis indices must match input shape on all non-axis dimensions.");
            }
        }
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int[] reduceStrides(int[] strides, int axis) {
        if (strides.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[strides.length - 1];
        for (int i = 0, j = 0; i < strides.length; i++) {
            if (i != axis) {
                reduced[j++] = strides[i];
            }
        }
        return reduced;
    }

    private static Tensor constantIndexTensor(int[] shape, int index) {
        int[] data = new int[elementCount(shape)];
        java.util.Arrays.fill(data, index);
        return new Tensor(data, shape, null, "select_indices", DataType.INT32);
    }

    private static int elementCount(int[] shape) {
        int size = 1;
        for (int dimension : shape) {
            size *= dimension;
        }
        return size;
    }

    private static int normalizeIndex(int index, int axisSize) {
        int normalized = index < 0 ? index + axisSize : index;
        if (normalized < 0 || normalized >= axisSize) {
            throw new IndexOutOfBoundsException(
                    "Index out of bounds for selected axis. index=" + index + ", axisSize=" + axisSize
            );
        }
        return normalized;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
