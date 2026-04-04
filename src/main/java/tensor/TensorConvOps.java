package tensor;

import operations.conv2d;
import operations.conv2dBackwardInput;
import operations.conv2dBackwardWeight;

import java.util.List;

final class TensorConvOps {
    private TensorConvOps() {
    }

    static Tensor conv2d(Tensor input, Tensor weight, Conv2dOptions options) {
        return conv2d(input, weight, null, options);
    }

    static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, Conv2dOptions options) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException("conv2d input and weight cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException("conv2d options cannot be null");
        }
        validateFloatingTensor(input, "conv2d input");
        validateFloatingTensor(weight, "conv2d weight");
        if (bias != null) {
            validateFloatingTensor(bias, "conv2d bias");
        }

        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        if (inputShape.length != 4 || weightShape.length != 4) {
            throw new IllegalArgumentException("conv2d currently requires rank-4 input and weight tensors.");
        }

        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];

        if (inChannels % options.groups() != 0) {
            throw new IllegalArgumentException("conv2d input channels must be divisible by groups.");
        }
        if (outChannels % options.groups() != 0) {
            throw new IllegalArgumentException("conv2d output channels must be divisible by groups.");
        }
        if (channelsPerGroup * options.groups() != inChannels) {
            throw new IllegalArgumentException("conv2d weight shape is incompatible with input channels and groups.");
        }

        if (bias != null) {
            int[] biasShape = bias.getShapeUnsafe();
            if (biasShape.length != 1 || biasShape[0] != outChannels) {
                throw new IllegalArgumentException("conv2d bias must have shape [outChannels].");
            }
        }

        int outH = inferOutputSize(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        int outW = inferOutputSize(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");

        DataType outputType = TensorDataTypeUtil.binary(input, weight);
        if (bias != null) {
            outputType = TensorDataTypeUtil.promote(outputType, bias.getDataType());
        }

        List<Tensor> inputs = bias == null ? List.of(input, weight) : List.of(input, weight, bias);
        final DataType gradType = outputType;
        Tensor out = new Tensor(
                new int[]{n, outChannels, outH, outW},
                inputs,
                new conv2d(options, bias != null),
                "conv2d",
                outputType
        );
        out.setRequiresGrad(input.getRequiresGrad() || weight.getRequiresGrad() || (bias != null && bias.getRequiresGrad()));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (input.getRequiresGrad()) {
                Tensor grad = new Tensor(
                        inputShape.clone(),
                        List.of(weight, outGrad),
                        new conv2dBackwardInput(options, inputShape),
                        "conv2d_backward_input",
                        gradType
                );
                accumulateGradient(input, grad);
            }
            if (weight.getRequiresGrad()) {
                Tensor grad = new Tensor(
                        weightShape.clone(),
                        List.of(input, outGrad),
                        new conv2dBackwardWeight(options, weightShape),
                        "conv2d_backward_weight",
                        gradType
                );
                accumulateGradient(weight, grad);
            }
            if (bias != null && bias.getRequiresGrad()) {
                Tensor grad = outGrad.sum(0).sum(1).sum(1);
                accumulateGradient(bias, grad);
            }
        });
        return out;
    }

    private static int inferOutputSize(int inputSize, int kernelSize, int pad, int stride, int dilation, String axisName) {
        int effectiveKernel = dilation * (kernelSize - 1) + 1;
        int numerator = inputSize + 2 * pad - effectiveKernel;
        if (numerator < 0) {
            throw new IllegalArgumentException("conv2d effective kernel does not fit input " + axisName + ".");
        }
        return numerator / stride + 1;
    }

    private static void validateFloatingTensor(Tensor tensor, String name) {
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
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
