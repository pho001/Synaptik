package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.storage.CpuStorageView;
import graph.execution.trace.ConvTraceMetadata;
import operations.nn.conv.conv2d;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.options.Conv2dOptions;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class Conv2dDirectBackend {
    private Conv2dDirectBackend() {
    }

    static void forwardF64(
            conv2d op,
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Tensor outputTensor,
            CpuKernelContext context
    ) {
        requireDType(DataType.FLOAT64, input, weight, bias, out);
        if (allArrayDenseZeroOffset(input, weight, bias, out)) {
            runForwardF64(
                    input.requireF64Array(),
                    weight.requireF64Array(),
                    bias == null ? null : bias.requireF64Array(),
                    input.shape(),
                    weight.shape(),
                    out.requireF64Array(),
                    out.shape(),
                    op.getOptions());
        } else {
            runForwardF64Storage(input, weight, bias, out, op.getOptions());
        }
        publishDirectTrace(outputTensor, context);
    }

    static void forwardF32(
            conv2d op,
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Tensor outputTensor,
            CpuKernelContext context
    ) {
        requireDType(DataType.FLOAT32, input, weight, bias, out);
        if (allArrayDenseZeroOffset(input, weight, bias, out)) {
            runForwardF32(
                    input.requireF32Array(),
                    weight.requireF32Array(),
                    bias == null ? null : bias.requireF32Array(),
                    input.shape(),
                    weight.shape(),
                    out.requireF32Array(),
                    out.shape(),
                    op.getOptions());
        } else {
            runForwardF32Storage(input, weight, bias, out, op.getOptions());
        }
        publishDirectTrace(outputTensor, context);
    }

    static void forwardBF16(
            conv2d op,
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Tensor outputTensor,
            CpuKernelContext context
    ) {
        requireDType(DataType.BFLOAT16, input, weight, bias, out);
        if (allArrayDenseZeroOffset(input, weight, bias, out)) {
            runForwardF16(
                    input.requireBF16Array(),
                    weight.requireBF16Array(),
                    bias == null ? null : bias.requireBF16Array(),
                    input.shape(),
                    weight.shape(),
                    out.requireBF16Array(),
                    out.shape(),
                    op.getOptions());
        } else {
            runForwardBF16Storage(input, weight, bias, out, op.getOptions());
        }
        publishDirectTrace(outputTensor, context);
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
                    float biasValue = bias == null ? 0.0f : TensorDTypeOps.fromBFloat16Bits(bias[oc]);
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
                                        acc += TensorDTypeOps.fromBFloat16Bits(input[indexNCHW(batch, ic, ih, iw, inputShape[1], inH, inW)])
                                                * TensorDTypeOps.fromBFloat16Bits(weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)]);
                                    }
                                }
                            }
                            out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = TensorDTypeOps.toBFloat16Bits(acc);
                        }
                    }
                }
            }
        }
    }

    private static void runForwardF64Storage(
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Conv2dOptions options
    ) {
        double[] inputArray = f64Array(input);
        MemorySegment inputSegment = f64Segment(input);
        double[] weightArray = f64Array(weight);
        MemorySegment weightSegment = f64Segment(weight);
        double[] biasArray = bias == null ? null : f64Array(bias);
        MemorySegment biasSegment = bias == null ? null : f64Segment(bias);
        double[] outArray = f64Array(out);
        MemorySegment outSegment = f64Segment(out);
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] weightStrides = weight.strides();
        int[] biasStrides = bias == null ? null : bias.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int weightBase = weight.storageOffset();
        int biasBase = bias == null ? 0 : bias.storageOffset();
        int outBase = out.storageOffset();

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
                    double biasValue = bias == null ? 0.0d
                            : readF64(biasArray, biasSegment, biasBase + oc * biasStrides[0]);
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
                                        acc += readF64(inputArray, inputSegment,
                                                indexNCHW(batch, ic, ih, iw, inputStrides, inputBase))
                                                * readF64(weightArray, weightSegment,
                                                indexOIHW(oc, icg, kh, kw, weightStrides, weightBase));
                                    }
                                }
                            }
                            writeF64(outArray, outSegment, indexNCHW(batch, oc, oh, ow, outStrides, outBase), acc);
                        }
                    }
                }
            }
        }
    }

    private static void runForwardF32Storage(
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Conv2dOptions options
    ) {
        float[] inputArray = f32Array(input);
        MemorySegment inputSegment = f32Segment(input);
        float[] weightArray = f32Array(weight);
        MemorySegment weightSegment = f32Segment(weight);
        float[] biasArray = bias == null ? null : f32Array(bias);
        MemorySegment biasSegment = bias == null ? null : f32Segment(bias);
        float[] outArray = f32Array(out);
        MemorySegment outSegment = f32Segment(out);
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] weightStrides = weight.strides();
        int[] biasStrides = bias == null ? null : bias.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int weightBase = weight.storageOffset();
        int biasBase = bias == null ? 0 : bias.storageOffset();
        int outBase = out.storageOffset();

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
                    float biasValue = bias == null ? 0.0f
                            : readF32(biasArray, biasSegment, biasBase + oc * biasStrides[0]);
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
                                        acc += readF32(inputArray, inputSegment,
                                                indexNCHW(batch, ic, ih, iw, inputStrides, inputBase))
                                                * readF32(weightArray, weightSegment,
                                                indexOIHW(oc, icg, kh, kw, weightStrides, weightBase));
                                    }
                                }
                            }
                            writeF32(outArray, outSegment, indexNCHW(batch, oc, oh, ow, outStrides, outBase), acc);
                        }
                    }
                }
            }
        }
    }

    private static void runForwardBF16Storage(
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out,
            Conv2dOptions options
    ) {
        short[] inputArray = bf16Array(input);
        MemorySegment inputSegment = bf16Segment(input);
        short[] weightArray = bf16Array(weight);
        MemorySegment weightSegment = bf16Segment(weight);
        short[] biasArray = bias == null ? null : bf16Array(bias);
        MemorySegment biasSegment = bias == null ? null : bf16Segment(bias);
        short[] outArray = bf16Array(out);
        MemorySegment outSegment = bf16Segment(out);
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outShape = out.shape();
        int[] inputStrides = input.strides();
        int[] weightStrides = weight.strides();
        int[] biasStrides = bias == null ? null : bias.strides();
        int[] outStrides = out.strides();
        int inputBase = input.storageOffset();
        int weightBase = weight.storageOffset();
        int biasBase = bias == null ? 0 : bias.storageOffset();
        int outBase = out.storageOffset();

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
                    float biasValue = bias == null ? 0.0f
                            : readBF16(biasArray, biasSegment, biasBase + oc * biasStrides[0]);
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
                                        acc += readBF16(inputArray, inputSegment,
                                                indexNCHW(batch, ic, ih, iw, inputStrides, inputBase))
                                                * readBF16(weightArray, weightSegment,
                                                indexOIHW(oc, icg, kh, kw, weightStrides, weightBase));
                                    }
                                }
                            }
                            writeBF16(outArray, outSegment, indexNCHW(batch, oc, oh, ow, outStrides, outBase), acc);
                        }
                    }
                }
            }
        }
    }

    private static void requireDType(
            DataType expected,
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out
    ) {
        if (input.dtype() != expected || weight.dtype() != expected || out.dtype() != expected
                || (bias != null && bias.dtype() != expected)) {
            throw new IllegalArgumentException("Conv2dDirectBackend requires " + expected + " storage views.");
        }
    }

    private static boolean allArrayDenseZeroOffset(
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out
    ) {
        return input.isArray()
                && weight.isArray()
                && (bias == null || bias.isArray())
                && out.isArray()
                && denseZeroOffset(input)
                && denseZeroOffset(weight)
                && (bias == null || denseZeroOffset(bias))
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

    private static int indexNCHW(int n, int c, int h, int w, int channels, int height, int width) {
        return ((n * channels + c) * height + h) * width + w;
    }

    private static int indexNCHW(int n, int c, int h, int w, int[] strides, int base) {
        return base + n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
    }

    private static int indexOIHW(int o, int i, int h, int w, int channelsPerGroup, int kernelH, int kernelW) {
        return ((o * channelsPerGroup + i) * kernelH + h) * kernelW + w;
    }

    private static int indexOIHW(int o, int i, int h, int w, int[] strides, int base) {
        return base + o * strides[0] + i * strides[1] + h * strides[2] + w * strides[3];
    }
}
