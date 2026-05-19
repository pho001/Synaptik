package tensor.ops.conv;

import operations.nn.conv.conv2d;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Conv2dOptions;

import java.util.List;

/**
 * Graph-building definition for NCHW 2-D convolution.
 */
public final class Conv2dOp {
    private Conv2dOp() {
    }

    /**
     * Applies 2-D convolution without bias.
     *
     * @param input rank-4 input tensor with shape {@code [N, C_in, H, W]}
     * @param weight rank-4 weight tensor with shape
     *               {@code [C_out, C_in / groups, kernelH, kernelW]}
     * @param options convolution options; must be non-null
     * @return output tensor with shape {@code [N, C_out, outH, outW]}
     * @throws IllegalArgumentException if tensors/options are null, non-floating,
     *                                  rank-incompatible, or channel/group constraints fail
     */
    public static Tensor build(Tensor input, Tensor weight, Conv2dOptions options) {
        return build(input, weight, null, options);
    }

    /**
     * Applies 2-D convolution with optional bias.
     *
     * @param input rank-4 input tensor with shape {@code [N, C_in, H, W]}
     * @param weight rank-4 weight tensor with shape
     *               {@code [C_out, C_in / groups, kernelH, kernelW]}
     * @param bias optional rank-1 bias tensor with shape {@code [C_out]}; may be null
     * @param options convolution options; must be non-null
     * @return output tensor with shape {@code [N, C_out, outH, outW]}
     * @throws IllegalArgumentException if required tensors/options are null,
     *                                  dtypes/ranks/shapes are invalid, or output size is invalid
     */
    public static Tensor build(Tensor input, Tensor weight, Tensor bias, Conv2dOptions options) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException("conv2d input and weight cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException("conv2d options cannot be null");
        }
        ConvSupport.validateFloatingTensor(input, "conv2d input");
        ConvSupport.validateFloatingTensor(weight, "conv2d weight");
        if (bias != null) {
            ConvSupport.validateFloatingTensor(bias, "conv2d bias");
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

        int outH = ConvSupport.inferOutputSize(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        int outW = ConvSupport.inferOutputSize(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");

        DataType outputType = TensorDataTypeUtil.binary(input, weight);
        if (bias != null) {
            outputType = TensorDataTypeUtil.promote(outputType, bias.getDataType());
        }

        List<Tensor> inputs = bias == null ? List.of(input, weight) : List.of(input, weight, bias);
        DataType gradType = outputType;
        Tensor out = bias == null
                ? TensorPrimitiveBuilder.nary(new int[]{n, outChannels, outH, outW}, inputs, new conv2d(options, false), "conv2d", outputType)
                : TensorPrimitiveBuilder.nary(new int[]{n, outChannels, outH, outW}, inputs, new conv2d(options, true), "conv2d", outputType);
        out.setRequiresGrad(input.getRequiresGrad() || weight.getRequiresGrad() || (bias != null && bias.getRequiresGrad()));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                        weight,
                        outGrad,
                        inputShape.clone(),
                        new operations.nn.conv.conv2dBackwardInput(options, inputShape),
                        "conv2d_backward_input",
                        gradType
                );
                ConvSupport.accumulateGradient(input, grad);
            }
            if (weight.getRequiresGrad()) {
                Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                        input,
                        outGrad,
                        weightShape.clone(),
                        new operations.nn.conv.conv2dBackwardWeight(options, weightShape),
                        "conv2d_backward_weight",
                        gradType
                );
                ConvSupport.accumulateGradient(weight, grad);
            }
            if (bias != null && bias.getRequiresGrad()) {
                Tensor grad = outGrad.sum(0).sum(1).sum(1);
                ConvSupport.accumulateGradient(bias, grad);
            }
        });
        return out;
    }
}
