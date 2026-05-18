package backend.cpu.kernels.nn;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import operations.nn.pool.avgPool2d;
import operations.nn.pool.avgPool2dBackwardInput;
import operations.nn.pool.maxPool2d;
import operations.nn.pool.maxPool2dBackwardInput;
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

    static void maxBackwardInputF64(maxPool2dBackwardInput op, Tensor outGrad, Tensor source, Tensor gradInput) {
        runMaxBackwardInputF64(
                TensorInternalAccess.float64Data(source),
                source.getShapeUnsafe(),
                TensorInternalAccess.float64Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float64Data(gradInput),
                op.getInputShape(),
                op.getOptions()
        );
    }

    static void maxBackwardInputF32(maxPool2dBackwardInput op, Tensor outGrad, Tensor source, Tensor gradInput) {
        runMaxBackwardInputF32(
                TensorInternalAccess.float32Data(source),
                source.getShapeUnsafe(),
                TensorInternalAccess.float32Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float32Data(gradInput),
                op.getInputShape(),
                op.getOptions()
        );
    }

    static void maxBackwardInputBF16(maxPool2dBackwardInput op, Tensor outGrad, Tensor source, Tensor gradInput) {
        runMaxBackwardInputF16(
                TensorInternalAccess.bfloat16Data(source),
                source.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(gradInput),
                op.getInputShape(),
                op.getOptions()
        );
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

    static void avgBackwardInputF64(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        runAvgBackwardInputF64(TensorInternalAccess.float64Data(outGrad), outGrad.getShapeUnsafe(), TensorInternalAccess.float64Data(gradInput), op.getInputShape(), op.getOptions());
    }

    static void avgBackwardInputF32(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        runAvgBackwardInputF32(TensorInternalAccess.float32Data(outGrad), outGrad.getShapeUnsafe(), TensorInternalAccess.float32Data(gradInput), op.getInputShape(), op.getOptions());
    }

    static void avgBackwardInputBF16(avgPool2dBackwardInput op, Tensor outGrad, Tensor gradInput) {
        runAvgBackwardInputF16(TensorInternalAccess.bfloat16Data(outGrad), outGrad.getShapeUnsafe(), TensorInternalAccess.bfloat16Data(gradInput), op.getInputShape(), op.getOptions());
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
                                float value = CpuDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, channel, ih, iw, c, inH, inW)]);
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
                        out[outputIndex] = CpuDTypeOps.toBFloat16Bits(best);
                        argmaxWorkspace[outputIndex] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runMaxBackwardInputF64(
            double[] source,
            int[] sourceShape,
            double[] outGrad,
            int[] outGradShape,
            double[] gradInput,
            int[] inputShape,
            Pool2dOptions options
    ) {
        validateMaxBackwardSourceShape(sourceShape, inputShape);
        java.util.Arrays.fill(gradInput, 0.0d);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        int bestIndex = findMaxIndexF64(source, batch, channel, inOriginH, inOriginW, c, inH, inW, options);
                        gradInput[bestIndex] += outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)];
                    }
                }
            }
        }
    }

    private static void runMaxBackwardInputF32(
            float[] source,
            int[] sourceShape,
            float[] outGrad,
            int[] outGradShape,
            float[] gradInput,
            int[] inputShape,
            Pool2dOptions options
    ) {
        validateMaxBackwardSourceShape(sourceShape, inputShape);
        java.util.Arrays.fill(gradInput, 0.0f);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        int bestIndex = findMaxIndexF32(source, batch, channel, inOriginH, inOriginW, c, inH, inW, options);
                        gradInput[bestIndex] += outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)];
                    }
                }
            }
        }
    }

    private static void runMaxBackwardInputF16(
            short[] source,
            int[] sourceShape,
            short[] outGrad,
            int[] outGradShape,
            short[] gradInput,
            int[] inputShape,
            Pool2dOptions options
    ) {
        validateMaxBackwardSourceShape(sourceShape, inputShape);
        java.util.Arrays.fill(gradInput, (short) 0);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int outputIndex = indexNCHW(batch, channel, oh, ow, c, outH, outW);
                        int bestIndex = findMaxIndexF16(source, batch, channel, inOriginH, inOriginW, c, inH, inW, options);
                        float updated = CpuDTypeOps.fromBFloat16Bits(gradInput[bestIndex])
                                + CpuDTypeOps.fromBFloat16Bits(outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)]);
                        gradInput[bestIndex] = CpuDTypeOps.toBFloat16Bits(updated);
                    }
                }
            }
        }
    }

    private static void validateMaxBackwardSourceShape(int[] sourceShape, int[] inputShape) {
        if (!java.util.Arrays.equals(sourceShape, inputShape)) {
            throw new IllegalArgumentException("maxPool2d backward source shape must match original input shape.");
        }
    }

    private static int findMaxIndexF64(
            double[] source,
            int batch,
            int channel,
            int inOriginH,
            int inOriginW,
            int channels,
            int inH,
            int inW,
            Pool2dOptions options
    ) {
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
                int index = indexNCHW(batch, channel, ih, iw, channels, inH, inW);
                double value = source[index];
                if (!found || value > best) {
                    best = value;
                    bestIndex = index;
                    found = true;
                }
            }
        }
        if (!found) {
            throw new IllegalArgumentException("maxPool2d backward window has no valid input elements.");
        }
        return bestIndex;
    }

    private static int findMaxIndexF32(
            float[] source,
            int batch,
            int channel,
            int inOriginH,
            int inOriginW,
            int channels,
            int inH,
            int inW,
            Pool2dOptions options
    ) {
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
                int index = indexNCHW(batch, channel, ih, iw, channels, inH, inW);
                float value = source[index];
                if (!found || value > best) {
                    best = value;
                    bestIndex = index;
                    found = true;
                }
            }
        }
        if (!found) {
            throw new IllegalArgumentException("maxPool2d backward window has no valid input elements.");
        }
        return bestIndex;
    }

    private static int findMaxIndexF16(
            short[] source,
            int batch,
            int channel,
            int inOriginH,
            int inOriginW,
            int channels,
            int inH,
            int inW,
            Pool2dOptions options
    ) {
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
                int index = indexNCHW(batch, channel, ih, iw, channels, inH, inW);
                float value = CpuDTypeOps.fromBFloat16Bits(source[index]);
                if (!found || value > best) {
                    best = value;
                    bestIndex = index;
                    found = true;
                }
            }
        }
        if (!found) {
            throw new IllegalArgumentException("maxPool2d backward window has no valid input elements.");
        }
        return bestIndex;
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
                                acc += CpuDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, channel, ih, iw, c, inH, inW)]);
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        out[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = CpuDTypeOps.toBFloat16Bits(acc / divisor);
                    }
                }
            }
        }
    }

    private static void runAvgBackwardInputF64(double[] outGrad, int[] outGradShape, double[] gradInput, int[] inputShape, Pool2dOptions options) {
        java.util.Arrays.fill(gradInput, 0.0d);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int validCount = countValidWindow(inOriginH, inOriginW, inH, inW, options);
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d backward window has no valid input elements.");
                        }
                        double contribution = outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)] / divisor;
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
                                gradInput[indexNCHW(batch, channel, ih, iw, c, inH, inW)] += contribution;
                            }
                        }
                    }
                }
            }
        }
    }

    private static void runAvgBackwardInputF32(float[] outGrad, int[] outGradShape, float[] gradInput, int[] inputShape, Pool2dOptions options) {
        java.util.Arrays.fill(gradInput, 0.0f);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int validCount = countValidWindow(inOriginH, inOriginW, inH, inW, options);
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d backward window has no valid input elements.");
                        }
                        float contribution = outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)] / divisor;
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
                                gradInput[indexNCHW(batch, channel, ih, iw, c, inH, inW)] += contribution;
                            }
                        }
                    }
                }
            }
        }
    }

    private static void runAvgBackwardInputF16(short[] outGrad, int[] outGradShape, short[] gradInput, int[] inputShape, Pool2dOptions options) {
        java.util.Arrays.fill(gradInput, (short) 0);
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];

        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    int inOriginH = oh * options.strideH() - options.padH();
                    for (int ow = 0; ow < outW; ow++) {
                        int inOriginW = ow * options.strideW() - options.padW();
                        int validCount = countValidWindow(inOriginH, inOriginW, inH, inW, options);
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d backward window has no valid input elements.");
                        }
                        float contribution = CpuDTypeOps.fromBFloat16Bits(outGrad[indexNCHW(batch, channel, oh, ow, c, outH, outW)]) / divisor;
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
                                int inputIndex = indexNCHW(batch, channel, ih, iw, c, inH, inW);
                                float updated = CpuDTypeOps.fromBFloat16Bits(gradInput[inputIndex]) + contribution;
                                gradInput[inputIndex] = CpuDTypeOps.toBFloat16Bits(updated);
                            }
                        }
                    }
                }
            }
        }
    }

    private static int countValidWindow(int inOriginH, int inOriginW, int inH, int inW, Pool2dOptions options) {
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
                validCount++;
            }
        }
        return validCount;
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int channels, int height, int width) {
        return ((batch * channels + channel) * height + h) * width + w;
    }
}
