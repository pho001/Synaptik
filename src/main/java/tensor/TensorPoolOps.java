package tensor;

import operations.avgPool2d;
import operations.avgPool2dBackwardInput;
import operations.maxPool2d;
import operations.maxPool2dBackwardInput;

import java.util.List;

final class TensorPoolOps {
    private TensorPoolOps() {
    }

    static Tensor maxPool2d(Tensor input, Pool2dOptions options) {
        validateInput(input, options, "maxPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = new Tensor(
                new int[]{inputShape[0], inputShape[1], outH, outW},
                List.of(input),
                new maxPool2d(options),
                "maxPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = new Tensor(
                    inputShape.clone(),
                    List.of(outGrad, out),
                    new maxPool2dBackwardInput(options, inputShape),
                    "maxPool2dBackwardInput",
                    input.getDataType()
            );
            accumulateGradient(input, grad);
        });
        return out;
    }

    static Tensor avgPool2d(Tensor input, Pool2dOptions options) {
        validateInput(input, options, "avgPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = new Tensor(
                new int[]{inputShape[0], inputShape[1], outH, outW},
                List.of(input),
                new avgPool2d(options),
                "avgPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = new Tensor(
                    inputShape.clone(),
                    List.of(outGrad),
                    new avgPool2dBackwardInput(options, inputShape),
                    "avgPool2dBackwardInput",
                    input.getDataType()
            );
            accumulateGradient(input, grad);
        });
        return out;
    }

    private static void validateInput(Tensor input, Pool2dOptions options, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException(opName + " options cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires floating numeric input.");
        }
        int[] inputShape = input.getShapeUnsafe();
        if (inputShape.length != 4) {
            throw new IllegalArgumentException(opName + " currently requires rank-4 NCHW input.");
        }
    }

    private static int inferOutputSize(int inputSize, int kernelSize, int pad, int stride, String axisName) {
        int numerator = inputSize + 2 * pad - kernelSize;
        if (numerator < 0) {
            throw new IllegalArgumentException("pool2d kernel does not fit input " + axisName + ".");
        }
        return numerator / stride + 1;
    }

    private static void validateWindowCoverage(
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

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
