package backend.cpu.kernels.nn;

import tensor.dtype.TensorDTypeOps;

import tensor.TensorInternalAccess;

import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.options.Pool2dOptions;
import tensor.Tensor;

final class Pool2dDirectBackend {
    private Pool2dDirectBackend() {
    }

    static void maxForwardF64(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        runMaxForwardF64(TensorInternalAccess.float64Data(input), input.getShapeUnsafe(), TensorInternalAccess.float64Data(out), out.getShapeUnsafe(), op.getOptions(), argmaxWorkspace);
    }

    static void maxForwardF32(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        runMaxForwardF32(TensorInternalAccess.float32Data(input), input.getShapeUnsafe(), TensorInternalAccess.float32Data(out), out.getShapeUnsafe(), op.getOptions(), argmaxWorkspace);
    }

    static void maxForwardBF16(maxPool2d op, Tensor input, Tensor out, int[] argmaxWorkspace) {
        runMaxForwardF16(TensorInternalAccess.bfloat16Data(input), input.getShapeUnsafe(), TensorInternalAccess.bfloat16Data(out), out.getShapeUnsafe(), op.getOptions(), argmaxWorkspace);
    }

    static void avgForwardF64(avgPool2d op, Tensor input, Tensor out) {
        runAvgForwardF64(TensorInternalAccess.float64Data(input), input.getShapeUnsafe(), TensorInternalAccess.float64Data(out), out.getShapeUnsafe(), op.getOptions());
    }

    static void avgForwardF32(avgPool2d op, Tensor input, Tensor out) {
        runAvgForwardF32(TensorInternalAccess.float32Data(input), input.getShapeUnsafe(), TensorInternalAccess.float32Data(out), out.getShapeUnsafe(), op.getOptions());
    }

    static void avgForwardBF16(avgPool2d op, Tensor input, Tensor out) {
        runAvgForwardF16(TensorInternalAccess.bfloat16Data(input), input.getShapeUnsafe(), TensorInternalAccess.bfloat16Data(out), out.getShapeUnsafe(), op.getOptions());
    }

    private static void runMaxForwardF64(double[] input, int[] inputShape, double[] out, int[] outShape, Pool2dOptions options, int[] argmaxWorkspace) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        double best = 0.0d;
                        int bestIndex = -1;
                        boolean found = false;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                double value = input[indexNCHW(batch, channel, ih, iw, c, inH, inW)];
                                if (!found || value > best) {
                                    best = value;
                                    bestIndex = indexNCHW(batch, channel, ih, iw, c, inH, inW);
                                    found = true;
                                }
                            }
                        }
                        if (!found) {
                            throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
                        }
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        out[outputIndex] = best;
                        argmaxWorkspace[outputIndex] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runMaxForwardF32(float[] input, int[] inputShape, float[] out, int[] outShape, Pool2dOptions options, int[] argmaxWorkspace) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        float best = 0.0f;
                        int bestIndex = -1;
                        boolean found = false;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                float value = input[indexNCHW(batch, channel, ih, iw, c, inH, inW)];
                                if (!found || value > best) {
                                    best = value;
                                    bestIndex = indexNCHW(batch, channel, ih, iw, c, inH, inW);
                                    found = true;
                                }
                            }
                        }
                        if (!found) {
                            throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
                        }
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        out[outputIndex] = best;
                        argmaxWorkspace[outputIndex] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runMaxForwardF16(short[] input, int[] inputShape, short[] out, int[] outShape, Pool2dOptions options, int[] argmaxWorkspace) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        float best = 0.0f;
                        int bestIndex = -1;
                        boolean found = false;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                float value = TensorDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, channel, ih, iw, c, inH, inW)]);
                                if (!found || value > best) {
                                    best = value;
                                    bestIndex = indexNCHW(batch, channel, ih, iw, c, inH, inW);
                                    found = true;
                                }
                            }
                        }
                        if (!found) {
                            throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
                        }
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        out[outputIndex] = TensorDTypeOps.toBFloat16Bits(best);
                        argmaxWorkspace[outputIndex] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runAvgForwardF64(double[] input, int[] inputShape, double[] out, int[] outShape, Pool2dOptions options) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        double acc = 0.0d;
                        int validCount = 0;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                acc += input[indexNCHW(batch, channel, ih, iw, c, inH, inW)];
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        out[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = acc / divisor;
                    }
                }
            }
        }
    }

    private static void runAvgForwardF32(float[] input, int[] inputShape, float[] out, int[] outShape, Pool2dOptions options) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        float acc = 0.0f;
                        int validCount = 0;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                acc += input[indexNCHW(batch, channel, ih, iw, c, inH, inW)];
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        out[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = acc / divisor;
                    }
                }
            }
        }
    }

    private static void runAvgForwardF16(short[] input, int[] inputShape, short[] out, int[] outShape, Pool2dOptions options) {
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        float acc = 0.0f;
                        int validCount = 0;
                        for (int kh = 0; kh < options.kernelH(); kh++) {
                            int ih = inOriginH + kh;
                            if (ih < 0 || ih >= inH) {
                                continue;
                            }
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) {
                                    continue;
                                }
                                acc += TensorDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, channel, ih, iw, c, inH, inW)]);
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        out[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = TensorDTypeOps.toBFloat16Bits(acc / divisor);
                    }
                }
            }
        }
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int channels, int height, int width) {
        return ((batch * channels + channel) * height + h) * width + w;
    }
}
