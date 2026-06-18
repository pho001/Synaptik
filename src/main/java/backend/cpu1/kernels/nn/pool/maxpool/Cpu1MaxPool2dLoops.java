package backend.cpu1.kernels.nn.pool.maxpool;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedMaxPool2dUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous NCHW MAX_POOL2D loops for cpu1.
 */
public final class Cpu1MaxPool2dLoops {
    private Cpu1MaxPool2dLoops() {
    }

    public static void runF32DenseArray(
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            Cpu1PreparedMaxPool2dUnit unit,
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
            float best = 0.0f;
            boolean found = false;
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
                    float value = input[inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW)];
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output[outputStorageOffset + outputIndex] = best;
        }
    }

    private static void computeF64ArrayRange(
            Cpu1PreparedMaxPool2dUnit unit,
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
            double best = 0.0d;
            boolean found = false;
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
                    double value = input[inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW)];
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output[outputStorageOffset + outputIndex] = best;
        }
    }

    private static void computeBf16ArrayRange(
            Cpu1PreparedMaxPool2dUnit unit,
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
            float best = 0.0f;
            boolean found = false;
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
                    float value = TensorDTypeOps.fromBFloat16Bits(bits);
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output[outputStorageOffset + outputIndex] = TensorDTypeOps.toBFloat16Bits(best);
        }
    }

    private static void computeF32SegmentRange(
            Cpu1PreparedMaxPool2dUnit unit,
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
            float best = 0.0f;
            boolean found = false;
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
                    float value = input.get(
                            JAVA_FLOAT,
                            (long) (inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW))
                                    * Float.BYTES
                    );
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output.set(JAVA_FLOAT, (long) (outputStorageOffset + outputIndex) * Float.BYTES, best);
        }
    }

    private static void computeF64SegmentRange(
            Cpu1PreparedMaxPool2dUnit unit,
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
            double best = 0.0d;
            boolean found = false;
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
                    double value = input.get(
                            JAVA_DOUBLE,
                            (long) (inputStorageOffset + inputIndex(unit, batch, channel, inputH, inputW))
                                    * Double.BYTES
                    );
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output.set(JAVA_DOUBLE, (long) (outputStorageOffset + outputIndex) * Double.BYTES, best);
        }
    }

    private static void computeBf16SegmentRange(
            Cpu1PreparedMaxPool2dUnit unit,
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
            float best = 0.0f;
            boolean found = false;
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
                    float value = TensorDTypeOps.fromBFloat16Bits(bits);
                    if (!found || value > best) {
                        best = value;
                        found = true;
                    }
                }
            }
            if (!found) {
                throw new IllegalArgumentException("maxPool2d window has no valid input elements.");
            }
            output.set(
                    JAVA_SHORT,
                    (long) (outputStorageOffset + outputIndex) * Short.BYTES,
                    TensorDTypeOps.toBFloat16Bits(best)
            );
        }
    }

    private static int inputIndex(
            Cpu1PreparedMaxPool2dUnit unit,
            int batch,
            int channel,
            int inputH,
            int inputW
    ) {
        return ((batch * unit.channels() + channel) * unit.inputH() + inputH) * unit.inputW() + inputW;
    }
}
