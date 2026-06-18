package backend.cpu1.kernels.nn.pool.avgpool;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous NCHW AVG_POOL2D loops for cpu1.
 */
public final class Cpu1AvgPool2dLoops {
    private Cpu1AvgPool2dLoops() {
    }

    public static void runF32DenseArray(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF32ArrayRange(
                        unit,
                        inputArray,
                        outputArray,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseArray(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF64ArrayRange(
                        unit,
                        inputArray,
                        outputArray,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseArray(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        short[] inputArray = input.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeBf16ArrayRange(
                        unit,
                        inputArray,
                        outputArray,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF32DenseSegment(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF32SegmentRange(
                        unit,
                        inputSegment,
                        outputSegment,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseSegment(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeF64SegmentRange(
                        unit,
                        inputSegment,
                        outputSegment,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseSegment(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.outputElementCount(), (startInclusive, endExclusive) ->
                computeBf16SegmentRange(
                        unit,
                        inputSegment,
                        outputSegment,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    private static void computeF32ArrayRange(
            Cpu1PreparedAvgPool2dUnit unit,
            float[] input,
            float[] output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            float acc = 0.0f;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    acc += input[inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW)];
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output[outputStorageOffset + outputIndex] = acc / divisor;
        }
    }

    private static void computeF64ArrayRange(
            Cpu1PreparedAvgPool2dUnit unit,
            double[] input,
            double[] output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            double acc = 0.0d;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    acc += input[inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW)];
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output[outputStorageOffset + outputIndex] = acc / divisor;
        }
    }

    private static void computeBf16ArrayRange(
            Cpu1PreparedAvgPool2dUnit unit,
            short[] input,
            short[] output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            float acc = 0.0f;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    short bits = input[inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW)];
                    acc += TensorDTypeOps.fromBFloat16Bits(bits);
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output[outputStorageOffset + outputIndex] = TensorDTypeOps.toBFloat16Bits(acc / divisor);
        }
    }

    private static void computeF32SegmentRange(
            Cpu1PreparedAvgPool2dUnit unit,
            MemorySegment input,
            MemorySegment output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            float acc = 0.0f;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    acc += input.get(
                            JAVA_FLOAT,
                            (long) (inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW))
                                    * Float.BYTES
                    );
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output.set(JAVA_FLOAT, (long) (outputStorageOffset + outputIndex) * Float.BYTES, acc / divisor);
        }
    }

    private static void computeF64SegmentRange(
            Cpu1PreparedAvgPool2dUnit unit,
            MemorySegment input,
            MemorySegment output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            double acc = 0.0d;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    acc += input.get(
                            JAVA_DOUBLE,
                            (long) (inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW))
                                    * Double.BYTES
                    );
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output.set(JAVA_DOUBLE, (long) (outputStorageOffset + outputIndex) * Double.BYTES, acc / divisor);
        }
    }

    private static void computeBf16SegmentRange(
            Cpu1PreparedAvgPool2dUnit unit,
            MemorySegment input,
            MemorySegment output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startInclusive,
            int endExclusive
    ) {
        for (int outputIndex = startInclusive; outputIndex < endExclusive; outputIndex++) {
            int outputW = outputIndex % unit.outputW();
            int tmp = outputIndex / unit.outputW();
            int outputH = tmp % unit.outputH();
            tmp /= unit.outputH();
            int channel = tmp % unit.channels();
            int batch = tmp / unit.channels();
            int inputOriginH = outputH * unit.strideH() - unit.padH();
            int inputOriginW = outputW * unit.strideW() - unit.padW();
            float acc = 0.0f;
            int validCount = 0;
            for (int kh = 0; kh < unit.kernelH(); kh++) {
                int inputH = inputOriginH + kh;
                if (inputH < 0 || inputH >= unit.inputH()) {
                    continue;
                }
                for (int kw = 0; kw < unit.kernelW(); kw++) {
                    int inputW = inputOriginW + kw;
                    if (inputW < 0 || inputW >= unit.inputW()) {
                        continue;
                    }
                    short bits = input.get(
                            JAVA_SHORT,
                            (long) (inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW))
                                    * Short.BYTES
                    );
                    acc += TensorDTypeOps.fromBFloat16Bits(bits);
                    validCount++;
                }
            }
            int divisor = divisor(unit, validCount);
            output.set(
                    JAVA_SHORT,
                    (long) (outputStorageOffset + outputIndex) * Short.BYTES,
                    TensorDTypeOps.toBFloat16Bits(acc / divisor)
            );
        }
    }

    private static int divisor(Cpu1PreparedAvgPool2dUnit unit, int validCount) {
        int divisor = unit.countIncludePad() ? unit.kernelH() * unit.kernelW() : validCount;
        if (divisor <= 0) {
            throw new IllegalArgumentException("avgPool2d window has no valid input elements.");
        }
        return divisor;
    }

    private static int inputIndex(
            Cpu1PreparedAvgPool2dUnit unit,
            int batch,
            int channel,
            int inputH,
            int inputW
    ) {
        return ((batch * unit.channels() + channel) * unit.inputH() + inputH) * unit.inputW() + inputW;
    }
}
