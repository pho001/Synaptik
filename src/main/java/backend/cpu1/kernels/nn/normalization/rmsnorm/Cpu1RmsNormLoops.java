package backend.cpu1.kernels.nn.normalization.rmsnorm;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous RMSNorm loops for cpu1.
 */
public final class Cpu1RmsNormLoops {
    private Cpu1RmsNormLoops() {
    }

    public static void runF32DenseArray(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        float[] inputArray = input.float32Array();
        float[] gammaArray = gamma.float32Array();
        float[] outputArray = output.float32Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF32Range(
                        unit,
                        inputArray,
                        gammaArray,
                        outputArray,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseArray(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        double[] inputArray = input.float64Array();
        double[] gammaArray = gamma.float64Array();
        double[] outputArray = output.float64Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF64Range(
                        unit,
                        inputArray,
                        gammaArray,
                        outputArray,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseArray(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        short[] inputArray = input.bfloat16Array();
        float[] gammaValues = decodeBf16Gamma(gamma.bfloat16Array(), gamma.storageOffset(), unit.normalizedSize());
        short[] outputArray = output.bfloat16Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeBf16Range(
                        unit,
                        inputArray,
                        gammaValues,
                        outputArray,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF32DenseSegment(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment gammaSegment = gamma.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF32SegmentRange(
                        unit,
                        inputSegment,
                        gammaSegment,
                        outputSegment,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseSegment(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment gammaSegment = gamma.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF64SegmentRange(
                        unit,
                        inputSegment,
                        gammaSegment,
                        outputSegment,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseSegment(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        float[] gammaValues = decodeBf16Gamma(gamma.segment(), gamma.storageOffset(), unit.normalizedSize());
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeBf16SegmentRange(
                        unit,
                        inputSegment,
                        gammaValues,
                        outputSegment,
                        input.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    private static void computeF32Range(
            Cpu1PreparedRmsNormUnit unit,
            float[] input,
            float[] gamma,
            float[] output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                output[outputBase + i] = input[inputBase + i] * invRms * gamma[gammaStorageOffset + i];
            }
        }
    }

    private static void computeF64Range(
            Cpu1PreparedRmsNormUnit unit,
            double[] input,
            double[] gamma,
            double[] output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        double epsilon = unit.epsilon();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            double invRms = 1.0d / Math.sqrt(sumSquaresF64(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                output[outputBase + i] = input[inputBase + i] * invRms * gamma[gammaStorageOffset + i];
            }
        }
    }

    private static void computeBf16Range(
            Cpu1PreparedRmsNormUnit unit,
            short[] input,
            float[] gamma,
            short[] output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            float invRms = (float) (1.0d / Math.sqrt(sumSquaresBf16(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(input[inputBase + i]);
                output[outputBase + i] = TensorDTypeOps.toBFloat16Bits(value * invRms * gamma[i]);
            }
        }
    }

    private static void computeF32SegmentRange(
            Cpu1PreparedRmsNormUnit unit,
            MemorySegment input,
            MemorySegment gamma,
            MemorySegment output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = input.get(JAVA_FLOAT, f32ByteOffset(inputBase + i));
                float scale = gamma.get(JAVA_FLOAT, f32ByteOffset(gammaStorageOffset + i));
                output.set(JAVA_FLOAT, f32ByteOffset(outputBase + i), value * invRms * scale);
            }
        }
    }

    private static void computeF64SegmentRange(
            Cpu1PreparedRmsNormUnit unit,
            MemorySegment input,
            MemorySegment gamma,
            MemorySegment output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        double epsilon = unit.epsilon();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            double invRms = 1.0d / Math.sqrt(sumSquaresF64(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                double value = input.get(JAVA_DOUBLE, f64ByteOffset(inputBase + i));
                double scale = gamma.get(JAVA_DOUBLE, f64ByteOffset(gammaStorageOffset + i));
                output.set(JAVA_DOUBLE, f64ByteOffset(outputBase + i), value * invRms * scale);
            }
        }
    }

    private static void computeBf16SegmentRange(
            Cpu1PreparedRmsNormUnit unit,
            MemorySegment input,
            float[] gamma,
            MemorySegment output,
            int inputStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            float invRms = (float) (1.0d / Math.sqrt(sumSquaresBf16(input, inputBase, normalizedSize)
                    / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(
                        input.get(JAVA_SHORT, bf16ByteOffset(inputBase + i))
                );
                output.set(
                        JAVA_SHORT,
                        bf16ByteOffset(outputBase + i),
                        TensorDTypeOps.toBFloat16Bits(value * invRms * gamma[i])
                );
            }
        }
    }

    private static double sumSquaresF32(float[] input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = input[base + i];
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF32(MemorySegment input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = input.get(JAVA_FLOAT, f32ByteOffset(base + i));
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF64(double[] input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = input[base + i];
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF64(MemorySegment input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = input.get(JAVA_DOUBLE, f64ByteOffset(base + i));
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresBf16(short[] input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(input[base + i]);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresBf16(MemorySegment input, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(
                    input.get(JAVA_SHORT, bf16ByteOffset(base + i))
            );
            total += value * value;
        }
        return total;
    }

    private static float[] decodeBf16Gamma(short[] gamma, int gammaStorageOffset, int normalizedSize) {
        float[] decoded = new float[normalizedSize];
        for (int i = 0; i < normalizedSize; i++) {
            decoded[i] = TensorDTypeOps.fromBFloat16Bits(gamma[gammaStorageOffset + i]);
        }
        return decoded;
    }

    private static float[] decodeBf16Gamma(MemorySegment gamma, int gammaStorageOffset, int normalizedSize) {
        float[] decoded = new float[normalizedSize];
        for (int i = 0; i < normalizedSize; i++) {
            decoded[i] = TensorDTypeOps.fromBFloat16Bits(
                    gamma.get(JAVA_SHORT, bf16ByteOffset(gammaStorageOffset + i))
            );
        }
        return decoded;
    }

    private static long f32ByteOffset(int index) {
        return (long) index * Float.BYTES;
    }

    private static long f64ByteOffset(int index) {
        return (long) index * Double.BYTES;
    }

    private static long bf16ByteOffset(int index) {
        return (long) index * Short.BYTES;
    }
}
