package tensor.ops.pool;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.options.Pool2dOptions;

final class PoolSupport {
    private PoolSupport() {
    }

    static void validateInput(Tensor input, Pool2dOptions options, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException(opName + " options cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32 || input.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException(opName + " requires floating numeric input.");
        }
        int[] inputShape = input.getShapeUnsafe();
        if (inputShape.length != 4) {
            throw new IllegalArgumentException(opName + " currently requires rank-4 NCHW input.");
        }
    }

    static int inferOutputSize(int inputSize, int kernelSize, int pad, int stride, boolean ceilMode, String axisName) {
        int numerator = inputSize + 2 * pad - kernelSize;
        if (numerator < 0) {
            throw new IllegalArgumentException("pool2d kernel does not fit input " + axisName + ".");
        }
        return (ceilMode ? (numerator + stride - 1) / stride : numerator / stride) + 1;
    }

    static void validateWindowCoverage(
            int inputSize,
            int kernelSize,
            int pad,
            int stride,
            int outSize,
            String axisName
    ) {
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int origin = outIndex * stride - pad;
            boolean hasValid = false;
            for (int k = 0; k < kernelSize; k++) {
                int inputIndex = origin + k;
                if (inputIndex >= 0 && inputIndex < inputSize) {
                    hasValid = true;
                    break;
                }
            }
            if (!hasValid) {
                throw new IllegalArgumentException(
                        "pool2d configuration creates an all-padding window on " + axisName + " axis at output index " + outIndex + "."
                );
            }
        }
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }
}
