package backend.cpu.kernels.nn;

import backend.cpu.storage.CpuStorageView;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;
import tensor.options.Pool2dOptions;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class Pool2dDirectBackend {
    private Pool2dDirectBackend() {
    }

    static void maxForwardF64(maxPool2d op, CpuStorageView input, CpuStorageView out, int[] argmaxWorkspace) {
        requireDType(DataType.FLOAT64, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runMaxForwardF64(input.requireF64Array(), input.shape(), out.requireF64Array(), out.shape(), op.getOptions(), argmaxWorkspace);
        } else {
            runMaxForwardF64Storage(input, out, op.getOptions(), argmaxWorkspace);
        }
    }

    static void maxForwardF32(maxPool2d op, CpuStorageView input, CpuStorageView out, int[] argmaxWorkspace) {
        requireDType(DataType.FLOAT32, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runMaxForwardF32(input.requireF32Array(), input.shape(), out.requireF32Array(), out.shape(), op.getOptions(), argmaxWorkspace);
        } else {
            runMaxForwardF32Storage(input, out, op.getOptions(), argmaxWorkspace);
        }
    }

    static void maxForwardBF16(maxPool2d op, CpuStorageView input, CpuStorageView out, int[] argmaxWorkspace) {
        requireDType(DataType.BFLOAT16, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runMaxForwardF16(input.requireBF16Array(), input.shape(), out.requireBF16Array(), out.shape(), op.getOptions(), argmaxWorkspace);
        } else {
            runMaxForwardBF16Storage(input, out, op.getOptions(), argmaxWorkspace);
        }
    }

    static void avgForwardF64(avgPool2d op, CpuStorageView input, CpuStorageView out) {
        requireDType(DataType.FLOAT64, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runAvgForwardF64(input.requireF64Array(), input.shape(), out.requireF64Array(), out.shape(), op.getOptions());
        } else {
            runAvgForwardF64Storage(input, out, op.getOptions());
        }
    }

    static void avgForwardF32(avgPool2d op, CpuStorageView input, CpuStorageView out) {
        requireDType(DataType.FLOAT32, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runAvgForwardF32(input.requireF32Array(), input.shape(), out.requireF32Array(), out.shape(), op.getOptions());
        } else {
            runAvgForwardF32Storage(input, out, op.getOptions());
        }
    }

    static void avgForwardBF16(avgPool2d op, CpuStorageView input, CpuStorageView out) {
        requireDType(DataType.BFLOAT16, input, out);
        if (allArrayDenseZeroOffset(input, out)) {
            runAvgForwardF16(input.requireBF16Array(), input.shape(), out.requireBF16Array(), out.shape(), op.getOptions());
        } else {
            runAvgForwardBF16Storage(input, out, op.getOptions());
        }
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

    private static void runMaxForwardF64Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options, int[] argmaxWorkspace) {
        double[] inputArray = f64Array(input);
        MemorySegment inputSegment = f64Segment(input);
        double[] outArray = f64Array(out);
        MemorySegment outSegment = f64Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                double value = readF64(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
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
                        writeF64(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), best);
                        argmaxWorkspace[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runMaxForwardF32Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options, int[] argmaxWorkspace) {
        float[] inputArray = f32Array(input);
        MemorySegment inputSegment = f32Segment(input);
        float[] outArray = f32Array(out);
        MemorySegment outSegment = f32Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                float value = readF32(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
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
                        writeF32(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), best);
                        argmaxWorkspace[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runMaxForwardBF16Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options, int[] argmaxWorkspace) {
        short[] inputArray = bf16Array(input);
        MemorySegment inputSegment = bf16Segment(input);
        short[] outArray = bf16Array(out);
        MemorySegment outSegment = bf16Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                float value = readBF16(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
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
                        writeBF16(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), best);
                        argmaxWorkspace[indexNCHW(batch, channel, oh, ow, c, outH, outW)] = bestIndex;
                    }
                }
            }
        }
    }

    private static void runAvgForwardF64Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options) {
        double[] inputArray = f64Array(input);
        MemorySegment inputSegment = f64Segment(input);
        double[] outArray = f64Array(out);
        MemorySegment outSegment = f64Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                acc += readF64(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        writeF64(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), acc / divisor);
                    }
                }
            }
        }
    }

    private static void runAvgForwardF32Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options) {
        float[] inputArray = f32Array(input);
        MemorySegment inputSegment = f32Segment(input);
        float[] outArray = f32Array(out);
        MemorySegment outSegment = f32Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                acc += readF32(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        writeF32(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), acc / divisor);
                    }
                }
            }
        }
    }

    private static void runAvgForwardBF16Storage(CpuStorageView input, CpuStorageView out, Pool2dOptions options) {
        short[] inputArray = bf16Array(input);
        MemorySegment inputSegment = bf16Segment(input);
        short[] outArray = bf16Array(out);
        MemorySegment outSegment = bf16Segment(out);
        int[] inputShape = input.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int outBase = out.storageOffset();
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
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < options.kernelW(); kw++) {
                                int iw = inOriginW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                acc += readBF16(inputArray, inputSegment,
                                        indexNCHW(batch, channel, ih, iw, inputStrides, inputBase));
                                validCount++;
                            }
                        }
                        int divisor = options.countIncludePad() ? options.kernelH() * options.kernelW() : validCount;
                        if (divisor <= 0) {
                            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
                        }
                        writeBF16(outArray, outSegment, indexNCHW(batch, channel, oh, ow, outStrides, outBase), acc / divisor);
                    }
                }
            }
        }
    }

    private static void requireDType(DataType expected, CpuStorageView input, CpuStorageView out) {
        if (input.dtype() != expected || out.dtype() != expected) {
            throw new IllegalArgumentException("Pool2dDirectBackend requires " + expected + " storage views.");
        }
    }

    private static boolean allArrayDenseZeroOffset(CpuStorageView input, CpuStorageView out) {
        return input.isArray()
                && out.isArray()
                && denseZeroOffset(input)
                && denseZeroOffset(out);
    }

    private static boolean denseZeroOffset(CpuStorageView view) {
        return view.storageOffset() == 0 && isDenseContiguous(view);
    }

    private static boolean isDenseContiguous(CpuStorageView view) {
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (strides[dim] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[dim]);
        }
        return true;
    }

    private static double[] f64Array(CpuStorageView view) {
        return view.isArray() ? view.requireF64Array() : null;
    }

    private static MemorySegment f64Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    private static double readF64(double[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(double[] array, MemorySegment segment, int offset, double value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static float[] f32Array(CpuStorageView view) {
        return view.isArray() ? view.requireF32Array() : null;
    }

    private static MemorySegment f32Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    private static float readF32(float[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(float[] array, MemorySegment segment, int offset, float value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static short[] bf16Array(CpuStorageView view) {
        return view.isArray() ? view.requireBF16Array() : null;
    }

    private static MemorySegment bf16Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    private static float readBF16(short[] array, MemorySegment segment, int offset) {
        short bits = array != null ? array[offset] : segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    private static void writeBF16(short[] array, MemorySegment segment, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (array != null) {
            array[offset] = bits;
        } else {
            segment.set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int channels, int height, int width) {
        return ((batch * channels + channel) * height + h) * width + w;
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int[] strides, int base) {
        return base + batch * strides[0] + channel * strides[1] + h * strides[2] + w * strides[3];
    }
}
