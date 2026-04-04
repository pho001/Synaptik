package tensor;

import operations.linear;

import java.util.List;

final class TensorLinearOps {
    private TensorLinearOps() {
    }

    static Tensor linear(Tensor input, Tensor weight) {
        validateInputAndWeight(input, weight, "linear");
        int[] outShape = input.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = weight.getShapeUnsafe()[1];
        Tensor out = new Tensor(outShape, List.of(input, weight), new linear(false), "linear",
                TensorDataTypeUtil.binary(input, weight));
        out.setRequiresGrad(input.getRequiresGrad() || weight.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(swapLastTwoAxes(weight));
                accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = swapLastTwoAxes(input).matmul(outGrad);
                accumulateGradient(weight, TensorBroadcastOps.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
        });
        out.setLabel("linear");
        return out;
    }

    static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        validateInputAndWeight(input, weight, "linear");
        if (bias == null) {
            throw new IllegalArgumentException("linear bias cannot be null");
        }
        if (bias.getDataType() == DataType.BOOL || bias.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("linear bias must use a floating dtype.");
        }

        int outFeatures = weight.getShapeUnsafe()[1];
        int[] biasShape = bias.getShapeUnsafe();
        if (biasShape.length != 1 || biasShape[0] != outFeatures) {
            throw new IllegalArgumentException("linear bias must have shape [outFeatures].");
        }

        int[] outShape = input.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = outFeatures;
        Tensor out = new Tensor(outShape, List.of(input, weight, bias), new linear(true), "linear",
                TensorDataTypeUtil.promote(TensorDataTypeUtil.binary(input, weight), bias.getDataType()));
        out.setRequiresGrad(input.getRequiresGrad() || weight.getRequiresGrad() || bias.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (input.getRequiresGrad()) {
                Tensor gradInput = outGrad.matmul(swapLastTwoAxes(weight));
                accumulateGradient(input, gradInput);
            }
            if (weight.getRequiresGrad()) {
                Tensor gradWeight = swapLastTwoAxes(input).matmul(outGrad);
                accumulateGradient(weight, TensorBroadcastOps.sumToShape(gradWeight, weight.getShapeUnsafe()));
            }
            if (bias.getRequiresGrad()) {
                accumulateGradient(bias, TensorBroadcastOps.sumToShape(outGrad, bias.getShapeUnsafe()));
            }
        });
        out.setLabel("linear");
        return out;
    }

    private static Tensor swapLastTwoAxes(Tensor tensor) {
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

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }

    private static void validateInputAndWeight(Tensor input, Tensor weight, String opName) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException(opName + " input and weight cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(opName + " input must use a floating dtype.");
        }
        if (weight.getDataType() == DataType.BOOL || weight.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(opName + " weight must use a floating dtype.");
        }
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException(opName + " input must have rank >= 2.");
        }
        if (weightShape.length != 2) {
            throw new IllegalArgumentException(opName + " weight must have rank 2 with shape [inFeatures, outFeatures].");
        }
        int inFeatures = inputShape[inputShape.length - 1];
        if (weightShape[0] != inFeatures) {
            throw new IllegalArgumentException(opName + " shape mismatch: input last dimension " + inFeatures
                    + " must match weight first dimension " + weightShape[0] + ".");
        }
    }
}
