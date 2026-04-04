package backend.kernels.cpu;

import backend.blas.OpenBlasFfmBridge;
import operations.conv2dGemm;
import tensor.Conv2dOptions;
import tensor.Tensor;

final class Conv2dGemmKernelSupport {
    private Conv2dGemmKernelSupport() {
    }

    static void forwardF64(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        runF64(input.getFloat64Data(), input.getShapeUnsafe(), weight.getFloat64Data(), weight.getShapeUnsafe(),
                bias == null ? null : bias.getFloat64Data(), out.getFloat64Data(), out.getShapeUnsafe(), op.getOptions());
    }

    static void forwardF32(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        runF32(input.getFloat32Data(), input.getShapeUnsafe(), weight.getFloat32Data(), weight.getShapeUnsafe(),
                bias == null ? null : bias.getFloat32Data(), out.getFloat32Data(), out.getShapeUnsafe(), op.getOptions());
    }

    static void forwardF16(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out) {
        runF16(input.getFloat16Data(), input.getShapeUnsafe(), weight.getFloat16Data(), weight.getShapeUnsafe(),
                bias == null ? null : bias.getFloat16Data(), out.getFloat16Data(), out.getShapeUnsafe(), op.getOptions());
    }

    private static void runF64(
            double[] input, int[] inputShape,
            double[] weight, int[] weightShape,
            double[] bias,
            double[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;

        double[] packedWeight = new double[kSize * outChannelsPerGroup];
        double[] im2col = new double[outSpatial * kSize];
        double[] gemmOut = new double[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF64(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colF64(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                if (!tryBlasF64(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize)) {
                    runJavaGemmF64(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                }
                scatterOutputF64(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
    }

    private static void runF32(
            float[] input, int[] inputShape,
            float[] weight, int[] weightShape,
            float[] bias,
            float[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;

        float[] packedWeight = new float[kSize * outChannelsPerGroup];
        float[] im2col = new float[outSpatial * kSize];
        float[] gemmOut = new float[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF32(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colF32(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                if (!tryBlasF32(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize)) {
                    runJavaGemmF32(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                }
                scatterOutputF32(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
    }

    private static void runF16(
            short[] input, int[] inputShape,
            short[] weight, int[] weightShape,
            short[] bias,
            short[] out, int[] outShape,
            Conv2dOptions options
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;

        float[] packedWeight = new float[kSize * outChannelsPerGroup];
        float[] im2col = new float[outSpatial * kSize];
        float[] gemmOut = new float[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF16(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colF16(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                runJavaGemmF32(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                scatterOutputF16(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
    }

    private static void fillIm2colF64(double[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, double[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? 0.0d
                                            : input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)];
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void fillIm2colF32(float[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, float[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? 0.0f
                                            : input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)];
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void fillIm2colF16(short[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, float[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? 0.0f
                                            : CpuDTypeOps.fromHalfBits(input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)]);
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void packWeightF64(double[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, double[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void packWeightF32(float[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, float[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void packWeightF16(short[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, float[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                CpuDTypeOps.fromHalfBits(weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)]);
                        kIndex++;
                    }
                }
            }
        }
    }

    private static boolean tryBlasF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            return false;
        }
        try {
            OpenBlasFfmBridge.dgemmRowMajorNoTrans(m, n, k, 1.0d, a, k, b, n, 0.0d, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean tryBlasF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (!OpenBlasFfmBridge.isAvailable()) {
            return false;
        }
        try {
            OpenBlasFfmBridge.sgemmRowMajorNoTrans(m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static void runJavaGemmF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        java.util.Arrays.fill(c, 0.0d);
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                double av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void runJavaGemmF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        java.util.Arrays.fill(c, 0.0f);
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                float av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void scatterOutputF64(double[] gemmOut, double[] out, double[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    double value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0d : bias[oc]);
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = value;
                }
                row++;
            }
        }
    }

    private static void scatterOutputF32(float[] gemmOut, float[] out, float[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0f : bias[oc]);
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = value;
                }
                row++;
            }
        }
    }

    private static void scatterOutputF16(float[] gemmOut, short[] out, short[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0f : CpuDTypeOps.fromHalfBits(bias[oc]));
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = CpuDTypeOps.toHalfBits(value);
                }
                row++;
            }
        }
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int channels, int height, int width) {
        return ((batch * channels + channel) * height + h) * width + w;
    }

    private static int indexOIHW(int outChannel, int inChannel, int kh, int kw, int channelsPerGroup, int kernelH, int kernelW) {
        return ((outChannel * channelsPerGroup + inChannel) * kernelH + kh) * kernelW + kw;
    }
}
