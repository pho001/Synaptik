package tensor.ops.linalg;

import operations.linalg.linear;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;

record LinearSpec(
        int[] outShape,
        DataType outputType,
        boolean hasBias
) {
    static LinearSpec resolve(Tensor input, Tensor weight, Tensor bias) {
        validateInputAndWeight(input, weight);
        boolean hasBias = bias != null;
        if (hasBias) {
            validateBias(weight, bias);
        }

        int outFeatures = weight.getShapeUnsafe()[1];
        int[] outShape = input.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = outFeatures;
        DataType outputType = TensorDataTypeUtil.binary(input, weight);
        if (hasBias) {
            outputType = TensorDataTypeUtil.promote(outputType, bias.getDataType());
        }
        return new LinearSpec(outShape, outputType, hasBias);
    }

    linear operation() {
        return new linear(hasBias);
    }

    private static void validateInputAndWeight(Tensor input, Tensor weight) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException("linear input and weight cannot be null");
        }
        LinalgSupport.requireFloating(input, "linear input");
        LinalgSupport.requireFloating(weight, "linear weight");
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("linear input must have rank >= 2.");
        }
        if (weightShape.length != 2) {
            throw new IllegalArgumentException("linear weight must have rank 2 with shape [inFeatures, outFeatures].");
        }
        int inFeatures = inputShape[inputShape.length - 1];
        if (weightShape[0] != inFeatures) {
            throw new IllegalArgumentException("linear shape mismatch: input last dimension " + inFeatures
                    + " must match weight first dimension " + weightShape[0] + ".");
        }
    }

    private static void validateBias(Tensor weight, Tensor bias) {
        LinalgSupport.requireFloating(bias, "linear bias");
        int outFeatures = weight.getShapeUnsafe()[1];
        int[] biasShape = bias.getShapeUnsafe();
        if (biasShape.length != 1 || biasShape[0] != outFeatures) {
            throw new IllegalArgumentException("linear bias must have shape [outFeatures].");
        }
    }
}
