package tensor.ops.conv;

import operations.nn.conv.conv2d;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorOps;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Conv2dOptions;
import tensor.options.Window2dOptions;

import java.util.ArrayList;
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
        Conv2dShapeRules.validateFloatingTensor(input, "conv2d input");
        Conv2dShapeRules.validateFloatingTensor(weight, "conv2d weight");
        if (bias != null) {
            Conv2dShapeRules.validateFloatingTensor(bias, "conv2d bias");
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

        int outH = Conv2dShapeRules.inferOutputSize(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        int outW = Conv2dShapeRules.inferOutputSize(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");

        DataType outputType = TensorDTypes.promoteFloating(input.getDataType(), weight.getDataType());
        if (bias != null) {
            outputType = TensorDTypes.promoteFloating(outputType, bias.getDataType());
        }

        List<Tensor> inputs = bias == null ? List.of(input, weight) : List.of(input, weight, bias);
        DataType gradType = outputType;
        Tensor out = bias == null
                ? TensorPrimitiveBuilder.nary(new int[]{n, outChannels, outH, outW}, inputs, new conv2d(options, false), "conv2d", outputType)
                : TensorPrimitiveBuilder.nary(new int[]{n, outChannels, outH, outW}, inputs, new conv2d(options, true), "conv2d", outputType);
        out.setRequiresGrad(input.getRequiresGrad() || weight.getRequiresGrad() || (bias != null && bias.getRequiresGrad()));
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (input.getRequiresGrad()) {
                context.accumulate(input, inputGradient(weight, outGrad, inputShape, weightShape, outH, outW, options));
            }
            if (weight.getRequiresGrad()) {
                context.accumulate(weight, weightGradient(input, outGrad, inputShape, weightShape, outH, outW, options));
            }
            if (bias != null && bias.getRequiresGrad()) {
                Tensor grad = outGrad.sum(0).sum(1).sum(1);
                context.accumulate(bias, grad);
            }
        });
        return out;
    }

    private static Tensor inputGradient(
            Tensor weight,
            Tensor outGrad,
            int[] inputShape,
            int[] weightShape,
            int outH,
            int outW,
            Conv2dOptions options
    ) {
        int n = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int windowArea = Math.multiplyExact(kernelH, kernelW);
        int windowCount = Math.multiplyExact(outH, outW);
        int groups = options.groups();
        int outChannelsPerGroup = outChannels / groups;
        int flatChannelsPerGroup = Math.multiplyExact(channelsPerGroup, windowArea);
        Tensor outGradColumns = outGrad.reshape(n, outChannels, windowCount);

        Tensor columnsGrad;
        if (groups == 1) {
            Tensor weight2dT = weight.reshape(outChannels, flatChannelsPerGroup).transpose();
            columnsGrad = weight2dT.matmul(outGradColumns);
        } else {
            List<Tensor> groupColumns = new ArrayList<>(groups);
            for (int group = 0; group < groups; group++) {
                int outStart = group * outChannelsPerGroup;
                int outEnd = outStart + outChannelsPerGroup;
                Tensor weightGroupT = weight.sliceAxis(0, outStart, outEnd)
                        .reshape(outChannelsPerGroup, flatChannelsPerGroup)
                        .transpose();
                Tensor outGradGroup = outGradColumns.sliceAxis(1, outStart, outEnd);
                groupColumns.add(weightGroupT.matmul(outGradGroup));
            }
            columnsGrad = TensorOps.concat(1, groupColumns);
        }

        return columnsGrad.fold2d(inputShape.clone(), windowOptions(options, kernelH, kernelW));
    }

    private static Tensor weightGradient(
            Tensor input,
            Tensor outGrad,
            int[] inputShape,
            int[] weightShape,
            int outH,
            int outW,
            Conv2dOptions options
    ) {
        int n = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int windowArea = Math.multiplyExact(kernelH, kernelW);
        int windowCount = Math.multiplyExact(outH, outW);
        int groups = options.groups();
        int outChannelsPerGroup = outChannels / groups;
        int flatChannelsPerGroup = Math.multiplyExact(channelsPerGroup, windowArea);
        Tensor columns = input.unfold2d(windowOptions(options, kernelH, kernelW));
        Tensor outGradColumns = outGrad.reshape(n, outChannels, windowCount);

        Tensor flatWeightGrad;
        if (groups == 1) {
            Tensor batchGrad = outGradColumns.matmul(transposeLastTwo(columns));
            flatWeightGrad = batchGrad.sum(0);
        } else {
            List<Tensor> groupWeights = new ArrayList<>(groups);
            for (int group = 0; group < groups; group++) {
                int outStart = group * outChannelsPerGroup;
                int outEnd = outStart + outChannelsPerGroup;
                int columnStart = group * flatChannelsPerGroup;
                int columnEnd = columnStart + flatChannelsPerGroup;
                Tensor outGradGroup = outGradColumns.sliceAxis(1, outStart, outEnd);
                Tensor columnsGroupT = transposeLastTwo(columns.sliceAxis(1, columnStart, columnEnd));
                groupWeights.add(outGradGroup.matmul(columnsGroupT).sum(0));
            }
            flatWeightGrad = TensorOps.concat(0, groupWeights);
        }

        return flatWeightGrad.reshape(weightShape.clone());
    }

    private static Tensor transposeLastTwo(Tensor tensor) {
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

    private static Window2dOptions windowOptions(Conv2dOptions options, int kernelH, int kernelW) {
        return new Window2dOptions(
                kernelH,
                kernelW,
                options.strideH(),
                options.strideW(),
                options.padH(),
                options.padW(),
                options.dilationH(),
                options.dilationW()
        );
    }
}
