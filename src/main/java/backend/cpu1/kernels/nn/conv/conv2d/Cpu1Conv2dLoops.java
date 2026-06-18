package backend.cpu1.kernels.nn.conv.conv2d;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous NCHW/OIHW direct CONV2D loops for cpu1.
 */
public final class Cpu1Conv2dLoops {
    private Cpu1Conv2dLoops() {
    }

    public static void runF32DenseArray(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        float[] inputArray = input.float32Array();
        float[] weightArray = weight.float32Array();
        float[] biasArray = bias == null ? null : bias.float32Array();
        float[] outputArray = output.float32Array();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF32ArrayRange(
                        unit,
                        inputArray,
                        weightArray,
                        biasArray,
                        outputArray,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseArray(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        double[] inputArray = input.float64Array();
        double[] weightArray = weight.float64Array();
        double[] biasArray = bias == null ? null : bias.float64Array();
        double[] outputArray = output.float64Array();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF64ArrayRange(
                        unit,
                        inputArray,
                        weightArray,
                        biasArray,
                        outputArray,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseArray(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        short[] inputArray = input.bfloat16Array();
        short[] weightArray = weight.bfloat16Array();
        short[] biasArray = bias == null ? null : bias.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeBf16ArrayRange(
                        unit,
                        inputArray,
                        weightArray,
                        biasArray,
                        outputArray,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF32DenseSegment(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment weightSegment = weight.segment();
        MemorySegment biasSegment = bias == null ? null : bias.segment();
        MemorySegment outputSegment = output.segment();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF32SegmentRange(
                        unit,
                        inputSegment,
                        weightSegment,
                        biasSegment,
                        outputSegment,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseSegment(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment weightSegment = weight.segment();
        MemorySegment biasSegment = bias == null ? null : bias.segment();
        MemorySegment outputSegment = output.segment();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF64SegmentRange(
                        unit,
                        inputSegment,
                        weightSegment,
                        biasSegment,
                        outputSegment,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseSegment(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment weightSegment = weight.segment();
        MemorySegment biasSegment = bias == null ? null : bias.segment();
        MemorySegment outputSegment = output.segment();
        int biasStorageOffset = bias == null ? 0 : bias.storageOffset();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeBf16SegmentRange(
                        unit,
                        inputSegment,
                        weightSegment,
                        biasSegment,
                        outputSegment,
                        input.storageOffset(),
                        weight.storageOffset(),
                        biasStorageOffset,
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    private static void computeF32ArrayRange(
            Cpu1PreparedConv2dUnit unit,
            float[] input,
            float[] weight,
            float[] bias,
            float[] output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            float acc = bias == null ? 0.0f : bias[biasStorageOffset + oc];
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        acc += input[inputStorageOffset + inputIndex(unit, batch, ic, ih, iw)]
                                * weight[weightStorageOffset + weightIndex(unit, oc, icg, kh, kw)];
                    }
                }
            }
            output[outputStorageOffset + outputIndex] = acc;
        }
    }

    private static void computeF64ArrayRange(
            Cpu1PreparedConv2dUnit unit,
            double[] input,
            double[] weight,
            double[] bias,
            double[] output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            double acc = bias == null ? 0.0d : bias[biasStorageOffset + oc];
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        acc += input[inputStorageOffset + inputIndex(unit, batch, ic, ih, iw)]
                                * weight[weightStorageOffset + weightIndex(unit, oc, icg, kh, kw)];
                    }
                }
            }
            output[outputStorageOffset + outputIndex] = acc;
        }
    }

    private static void computeBf16ArrayRange(
            Cpu1PreparedConv2dUnit unit,
            short[] input,
            short[] weight,
            short[] bias,
            short[] output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            float acc = bias == null ? 0.0f
                    : TensorDTypeOps.fromBFloat16Bits(bias[biasStorageOffset + oc]);
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        float inputValue = TensorDTypeOps.fromBFloat16Bits(
                                input[inputStorageOffset + inputIndex(unit, batch, ic, ih, iw)]
                        );
                        float weightValue = TensorDTypeOps.fromBFloat16Bits(
                                weight[weightStorageOffset + weightIndex(unit, oc, icg, kh, kw)]
                        );
                        acc += inputValue * weightValue;
                    }
                }
            }
            output[outputStorageOffset + outputIndex] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private static void computeF32SegmentRange(
            Cpu1PreparedConv2dUnit unit,
            MemorySegment input,
            MemorySegment weight,
            MemorySegment bias,
            MemorySegment output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            float acc = bias == null ? 0.0f : getF32(bias, biasStorageOffset + oc);
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        acc += getF32(input, inputStorageOffset + inputIndex(unit, batch, ic, ih, iw))
                                * getF32(weight, weightStorageOffset + weightIndex(unit, oc, icg, kh, kw));
                    }
                }
            }
            setF32(output, outputStorageOffset + outputIndex, acc);
        }
    }

    private static void computeF64SegmentRange(
            Cpu1PreparedConv2dUnit unit,
            MemorySegment input,
            MemorySegment weight,
            MemorySegment bias,
            MemorySegment output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            double acc = bias == null ? 0.0d : getF64(bias, biasStorageOffset + oc);
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        acc += getF64(input, inputStorageOffset + inputIndex(unit, batch, ic, ih, iw))
                                * getF64(weight, weightStorageOffset + weightIndex(unit, oc, icg, kh, kw));
                    }
                }
            }
            setF64(output, outputStorageOffset + outputIndex, acc);
        }
    }

    private static void computeBf16SegmentRange(
            Cpu1PreparedConv2dUnit unit,
            MemorySegment input,
            MemorySegment weight,
            MemorySegment bias,
            MemorySegment output,
            int inputStorageOffset,
            int weightStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int ow = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int oh = tmp % unit.outputH();
            tmp /= unit.outputH();
            int oc = tmp % unit.outChannels();
            int batch = tmp / unit.outChannels();
            int group = oc / unit.outChannelsPerGroup();
            int inputChannelBase = group * unit.channelsPerGroup();
            float acc = bias == null ? 0.0f : getBf16(bias, biasStorageOffset + oc);
            int inputOriginH = oh * unit.strideH() - unit.padH();
            int inputOriginW = ow * unit.strideW() - unit.padW();
            for (int icg = 0; icg < unit.channelsPerGroup(); icg++) {
                int ic = inputChannelBase + icg;
                for (int kh = 0; kh < unit.kernelH(); kh++) {
                    int ih = inputOriginH + kh * unit.dilationH();
                    if (ih < 0 || ih >= unit.inputH()) {
                        continue;
                    }
                    for (int kw = 0; kw < unit.kernelW(); kw++) {
                        int iw = inputOriginW + kw * unit.dilationW();
                        if (iw < 0 || iw >= unit.inputW()) {
                            continue;
                        }
                        acc += getBf16(input, inputStorageOffset + inputIndex(unit, batch, ic, ih, iw))
                                * getBf16(weight, weightStorageOffset + weightIndex(unit, oc, icg, kh, kw));
                    }
                }
            }
            setBf16(output, outputStorageOffset + outputIndex, acc);
        }
    }

    private static float getF32(MemorySegment segment, int offset) {
        return segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void setF32(MemorySegment segment, int offset, float value) {
        segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
    }

    private static double getF64(MemorySegment segment, int offset) {
        return segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void setF64(MemorySegment segment, int offset, double value) {
        segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
    }

    private static float getBf16(MemorySegment segment, int offset) {
        return TensorDTypeOps.fromBFloat16Bits(segment.get(JAVA_SHORT, (long) offset * Short.BYTES));
    }

    private static void setBf16(MemorySegment segment, int offset, float value) {
        segment.set(JAVA_SHORT, (long) offset * Short.BYTES, TensorDTypeOps.toBFloat16Bits(value));
    }

    private static int inputIndex(
            Cpu1PreparedConv2dUnit unit,
            int batch,
            int channel,
            int inputH,
            int inputW
    ) {
        return ((batch * unit.inChannels() + channel) * unit.inputH() + inputH) * unit.inputW() + inputW;
    }

    private static int weightIndex(
            Cpu1PreparedConv2dUnit unit,
            int outChannel,
            int inChannelGroup,
            int kernelH,
            int kernelW
    ) {
        return ((outChannel * unit.channelsPerGroup() + inChannelGroup) * unit.kernelH() + kernelH)
                * unit.kernelW() + kernelW;
    }
}
