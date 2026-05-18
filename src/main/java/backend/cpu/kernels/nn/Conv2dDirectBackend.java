package backend.cpu.kernels.nn;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import graph.execution.trace.ConvTraceMetadata;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardWeight;
import tensor.options.Conv2dOptions;
import tensor.Tensor;

final class Conv2dDirectBackend {
    private Conv2dDirectBackend() {
    }

    static void forwardF64(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runForwardF64(TensorInternalAccess.float64Data(input), TensorInternalAccess.float64Data(weight), bias == null ? null : TensorInternalAccess.float64Data(bias),
                input.getShapeUnsafe(), weight.getShapeUnsafe(), TensorInternalAccess.float64Data(out), out.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(out, context);
    }

    static void forwardF32(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runForwardF32(TensorInternalAccess.float32Data(input), TensorInternalAccess.float32Data(weight), bias == null ? null : TensorInternalAccess.float32Data(bias),
                input.getShapeUnsafe(), weight.getShapeUnsafe(), TensorInternalAccess.float32Data(out), out.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(out, context);
    }

    static void forwardBF16(conv2d op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runForwardF16(TensorInternalAccess.bfloat16Data(input), TensorInternalAccess.bfloat16Data(weight), bias == null ? null : TensorInternalAccess.bfloat16Data(bias),
                input.getShapeUnsafe(), weight.getShapeUnsafe(), TensorInternalAccess.bfloat16Data(out), out.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(out, context);
    }

    static void backwardInputF64(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputF64(TensorInternalAccess.float64Data(weight), TensorInternalAccess.float64Data(outGrad), TensorInternalAccess.float64Data(gradInput),
                op.getInputShape(), weight.getShapeUnsafe(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradInput, context);
    }

    static void backwardInputF32(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputF32(TensorInternalAccess.float32Data(weight), TensorInternalAccess.float32Data(outGrad), TensorInternalAccess.float32Data(gradInput),
                op.getInputShape(), weight.getShapeUnsafe(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradInput, context);
    }

    static void backwardInputF16(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputF16(TensorInternalAccess.bfloat16Data(weight), TensorInternalAccess.bfloat16Data(outGrad), TensorInternalAccess.bfloat16Data(gradInput),
                op.getInputShape(), weight.getShapeUnsafe(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradInput, context);
    }

    static void backwardWeightF64(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightF64(TensorInternalAccess.float64Data(input), TensorInternalAccess.float64Data(outGrad), TensorInternalAccess.float64Data(gradWeight),
                input.getShapeUnsafe(), op.getWeightShape(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradWeight, context);
    }

    static void backwardWeightF32(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightF32(TensorInternalAccess.float32Data(input), TensorInternalAccess.float32Data(outGrad), TensorInternalAccess.float32Data(gradWeight),
                input.getShapeUnsafe(), op.getWeightShape(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradWeight, context);
    }

    static void backwardWeightF16(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightF16(TensorInternalAccess.bfloat16Data(input), TensorInternalAccess.bfloat16Data(outGrad), TensorInternalAccess.bfloat16Data(gradWeight),
                input.getShapeUnsafe(), op.getWeightShape(), outGrad.getShapeUnsafe(), op.getOptions());
        publishDirectTrace(gradWeight, context);
    }

    private static void publishDirectTrace(Tensor node, CpuKernelContext context) {
        if (context == null) {
            return;
        }
        context.publishConvTrace(node, new ConvTraceMetadata(
                "DIRECT",
                false,
                false,
                "NONE",
                0,
                0,
                0,
                0,
                1
        ));
    }

    private static void runForwardF64(
            double[] input, double[] weight, double[] bias,
            int[] inputShape, int[] weightShape,
            double[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    double biasValue = bias == null ? 0.0d : bias[oc];
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            int inOriginW = ow * options.strideW() - options.padW();
                            double acc = biasValue;
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)]
                                                * weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                                    }
                                }
                            }
                            out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void runForwardF32(
            float[] input, float[] weight, float[] bias,
            int[] inputShape, int[] weightShape,
            float[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int n = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float biasValue = bias == null ? 0.0f : bias[oc];
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            int inOriginW = ow * options.strideW() - options.padW();
                            float acc = biasValue;
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += input[indexNCHW(batch, ic, ih, iw, inputShape[1], inH, inW)]
                                                * weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                                    }
                                }
                            }
                            out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void runForwardF16(
            short[] input, short[] weight, short[] bias,
            int[] inputShape, int[] weightShape,
            short[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int n = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float biasValue = bias == null ? 0.0f : CpuDTypeOps.fromBFloat16Bits(bias[oc]);
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            int inOriginW = ow * options.strideW() - options.padW();
                            float acc = biasValue;
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += CpuDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, ic, ih, iw, inputShape[1], inH, inW)])
                                                * CpuDTypeOps.fromBFloat16Bits(weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)]);
                                    }
                                }
                            }
                            out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = CpuDTypeOps.toBFloat16Bits(acc);
                        }
                    }
                }
            }
        }
    }

    private static void runBackwardInputF64(
            double[] weight, double[] outGrad, double[] gradInput,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        java.util.Arrays.fill(gradInput, 0.0d);
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            double grad = outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                            int inOriginW = ow * options.strideW() - options.padW();
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        gradInput[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] +=
                                                grad * weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void runBackwardInputF32(
            float[] weight, float[] outGrad, float[] gradInput,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        java.util.Arrays.fill(gradInput, 0.0f);
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            float grad = outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                            int inOriginW = ow * options.strideW() - options.padW();
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        gradInput[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] +=
                                                grad * weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void runBackwardInputF16(
            short[] weight, short[] outGrad, short[] gradInput,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        java.util.Arrays.fill(gradInput, (short) 0);
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        float[] accum = new float[gradInput.length];
        for (int batch = 0; batch < n; batch++) {
            for (int g = 0; g < options.groups(); g++) {
                int inChannelBase = g * channelsPerGroup;
                int outChannelBase = g * outChannelsPerGroup;
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    for (int oh = 0; oh < outH; oh++) {
                        int inOriginH = oh * options.strideH() - options.padH();
                        for (int ow = 0; ow < outW; ow++) {
                            float grad = CpuDTypeOps.fromBFloat16Bits(outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)]);
                            int inOriginW = ow * options.strideW() - options.padW();
                            for (int icg = 0; icg < channelsPerGroup; icg++) {
                                int ic = inChannelBase + icg;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    int ih = inOriginH + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int iw = inOriginW + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        accum[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] +=
                                                grad * CpuDTypeOps.fromBFloat16Bits(weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < accum.length; i++) {
            gradInput[i] = CpuDTypeOps.toBFloat16Bits(accum[i]);
        }
    }

    private static void runBackwardWeightF64(
            double[] input, double[] outGrad, double[] gradWeight,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        java.util.Arrays.fill(gradWeight, 0.0d);
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int g = 0; g < options.groups(); g++) {
            int inChannelBase = g * channelsPerGroup;
            int outChannelBase = g * outChannelsPerGroup;
            for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                int oc = outChannelBase + ocg;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            double acc = 0.0d;
                            for (int batch = 0; batch < n; batch++) {
                                for (int oh = 0; oh < outH; oh++) {
                                    int ih = oh * options.strideH() - options.padH() + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int ow = 0; ow < outW; ow++) {
                                        int iw = ow * options.strideW() - options.padW() + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)]
                                                * outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                                    }
                                }
                            }
                            gradWeight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void runBackwardWeightF32(
            float[] input, float[] outGrad, float[] gradWeight,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        java.util.Arrays.fill(gradWeight, 0.0f);
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int g = 0; g < options.groups(); g++) {
            int inChannelBase = g * channelsPerGroup;
            int outChannelBase = g * outChannelsPerGroup;
            for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                int oc = outChannelBase + ocg;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            float acc = 0.0f;
                            for (int batch = 0; batch < n; batch++) {
                                for (int oh = 0; oh < outH; oh++) {
                                    int ih = oh * options.strideH() - options.padH() + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int ow = 0; ow < outW; ow++) {
                                        int iw = ow * options.strideW() - options.padW() + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)]
                                                * outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                                    }
                                }
                            }
                            gradWeight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void runBackwardWeightF16(
            short[] input, short[] outGrad, short[] gradWeight,
            int[] inputShape, int[] weightShape, int[] outGradShape,
            Conv2dOptions options
    ) {
        int size = gradWeight.length;
        float[] accum = new float[size];
        int n = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();

        for (int g = 0; g < options.groups(); g++) {
            int inChannelBase = g * channelsPerGroup;
            int outChannelBase = g * outChannelsPerGroup;
            for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                int oc = outChannelBase + ocg;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            float acc = 0.0f;
                            for (int batch = 0; batch < n; batch++) {
                                for (int oh = 0; oh < outH; oh++) {
                                    int ih = oh * options.strideH() - options.padH() + kh * options.dilationH();
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int ow = 0; ow < outW; ow++) {
                                        int iw = ow * options.strideW() - options.padW() + kw * options.dilationW();
                                        if (iw < 0 || iw >= inW) continue;
                                        acc += CpuDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)])
                                                * CpuDTypeOps.fromBFloat16Bits(outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)]);
                                    }
                                }
                            }
                            accum[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] = acc;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < size; i++) {
            gradWeight[i] = CpuDTypeOps.toBFloat16Bits(accum[i]);
        }
    }

    private static int indexNCHW(int n, int c, int h, int w, int channels, int height, int width) {
        return ((n * channels + c) * height + h) * width + w;
    }

    private static int indexOIHW(int o, int i, int h, int w, int channelsPerGroup, int kernelH, int kernelW) {
        return ((o * channelsPerGroup + i) * kernelH + h) * kernelW + w;
    }
}
