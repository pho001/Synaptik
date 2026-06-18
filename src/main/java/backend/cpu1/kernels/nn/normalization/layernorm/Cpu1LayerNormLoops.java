package backend.cpu1.kernels.nn.normalization.layernorm;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous LayerNorm loops for cpu1.
 */
public final class Cpu1LayerNormLoops {
    private Cpu1LayerNormLoops() {
    }

    public static void runF32DenseArray(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        float[] inputArray = input.float32Array();
        float[] gammaArray = gamma.float32Array();
        float[] betaArray = beta.float32Array();
        float[] outputArray = output.float32Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF32Range(
                        unit,
                        inputArray,
                        gammaArray,
                        betaArray,
                        outputArray,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseArray(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        double[] inputArray = input.float64Array();
        double[] gammaArray = gamma.float64Array();
        double[] betaArray = beta.float64Array();
        double[] outputArray = output.float64Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF64Range(
                        unit,
                        inputArray,
                        gammaArray,
                        betaArray,
                        outputArray,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseArray(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        short[] inputArray = input.bfloat16Array();
        short[] gammaArray = gamma.bfloat16Array();
        short[] betaArray = beta.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeBf16Range(
                        unit,
                        inputArray,
                        gammaArray,
                        betaArray,
                        outputArray,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF32DenseSegment(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment gammaSegment = gamma.segment();
        MemorySegment betaSegment = beta.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF32SegmentRange(
                        unit,
                        inputSegment,
                        gammaSegment,
                        betaSegment,
                        outputSegment,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runF64DenseSegment(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment gammaSegment = gamma.segment();
        MemorySegment betaSegment = beta.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeF64SegmentRange(
                        unit,
                        inputSegment,
                        gammaSegment,
                        betaSegment,
                        outputSegment,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    public static void runBf16DenseSegment(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    ) {
        MemorySegment inputSegment = input.segment();
        MemorySegment gammaSegment = gamma.segment();
        MemorySegment betaSegment = beta.segment();
        MemorySegment outputSegment = output.segment();
        unit.launchPolicy().launch(unit.groupCount(), (startInclusive, endExclusive) ->
                computeBf16SegmentRange(
                        unit,
                        inputSegment,
                        gammaSegment,
                        betaSegment,
                        outputSegment,
                        input.storageOffset(),
                        gamma.storageOffset(),
                        beta.storageOffset(),
                        output.storageOffset(),
                        startInclusive,
                        endExclusive
                ));
    }

    private static void computeF32Range(
            Cpu1PreparedLayerNormUnit unit,
            float[] input,
            float[] gamma,
            float[] beta,
            float[] output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF32 stats = statsF32(input, inputBase, normalizedSize);
            float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                output[outputBase + i] = ((input[inputBase + i] - stats.mean()) * invStd)
                        * gamma[gammaStorageOffset + i]
                        + beta[betaStorageOffset + i];
            }
        }
    }

    private static void computeF64Range(
            Cpu1PreparedLayerNormUnit unit,
            double[] input,
            double[] gamma,
            double[] beta,
            double[] output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        double epsilon = unit.epsilon();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF64 stats = statsF64(input, inputBase, normalizedSize);
            double variance = Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            double invStd = 1.0d / Math.sqrt(variance + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                output[outputBase + i] = ((input[inputBase + i] - stats.mean()) * invStd)
                        * gamma[gammaStorageOffset + i]
                        + beta[betaStorageOffset + i];
            }
        }
    }

    private static void computeBf16Range(
            Cpu1PreparedLayerNormUnit unit,
            short[] input,
            short[] gamma,
            short[] beta,
            short[] output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF32 stats = statsBf16(input, inputBase, normalizedSize);
            float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(input[inputBase + i]);
                float scale = TensorDTypeOps.fromBFloat16Bits(gamma[gammaStorageOffset + i]);
                float shift = TensorDTypeOps.fromBFloat16Bits(beta[betaStorageOffset + i]);
                output[outputBase + i] = TensorDTypeOps.toBFloat16Bits(
                        ((value - stats.mean()) * invStd) * scale + shift
                );
            }
        }
    }

    private static void computeF32SegmentRange(
            Cpu1PreparedLayerNormUnit unit,
            MemorySegment input,
            MemorySegment gamma,
            MemorySegment beta,
            MemorySegment output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF32 stats = statsF32(input, inputBase, normalizedSize);
            float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = input.get(JAVA_FLOAT, f32ByteOffset(inputBase + i));
                float scale = gamma.get(JAVA_FLOAT, f32ByteOffset(gammaStorageOffset + i));
                float shift = beta.get(JAVA_FLOAT, f32ByteOffset(betaStorageOffset + i));
                output.set(
                        JAVA_FLOAT,
                        f32ByteOffset(outputBase + i),
                        ((value - stats.mean()) * invStd) * scale + shift
                );
            }
        }
    }

    private static void computeF64SegmentRange(
            Cpu1PreparedLayerNormUnit unit,
            MemorySegment input,
            MemorySegment gamma,
            MemorySegment beta,
            MemorySegment output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        double epsilon = unit.epsilon();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF64 stats = statsF64(input, inputBase, normalizedSize);
            double variance = Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            double invStd = 1.0d / Math.sqrt(variance + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                double value = input.get(JAVA_DOUBLE, f64ByteOffset(inputBase + i));
                double scale = gamma.get(JAVA_DOUBLE, f64ByteOffset(gammaStorageOffset + i));
                double shift = beta.get(JAVA_DOUBLE, f64ByteOffset(betaStorageOffset + i));
                output.set(
                        JAVA_DOUBLE,
                        f64ByteOffset(outputBase + i),
                        ((value - stats.mean()) * invStd) * scale + shift
                );
            }
        }
    }

    private static void computeBf16SegmentRange(
            Cpu1PreparedLayerNormUnit unit,
            MemorySegment input,
            MemorySegment gamma,
            MemorySegment beta,
            MemorySegment output,
            int inputStorageOffset,
            int gammaStorageOffset,
            int betaStorageOffset,
            int outputStorageOffset,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        int normalizedSize = unit.normalizedSize();
        float epsilon = unit.epsilonF32();
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int inputBase = inputStorageOffset + group * normalizedSize;
            int outputBase = outputStorageOffset + group * normalizedSize;
            StatsF32 stats = statsBf16(input, inputBase, normalizedSize);
            float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(
                        input.get(JAVA_SHORT, bf16ByteOffset(inputBase + i))
                );
                float scale = TensorDTypeOps.fromBFloat16Bits(
                        gamma.get(JAVA_SHORT, bf16ByteOffset(gammaStorageOffset + i))
                );
                float shift = TensorDTypeOps.fromBFloat16Bits(
                        beta.get(JAVA_SHORT, bf16ByteOffset(betaStorageOffset + i))
                );
                output.set(
                        JAVA_SHORT,
                        bf16ByteOffset(outputBase + i),
                        TensorDTypeOps.toBFloat16Bits(((value - stats.mean()) * invStd) * scale + shift)
                );
            }
        }
    }

    private static StatsF32 statsF32(float[] input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = input[base + i];
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 statsF32(MemorySegment input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = input.get(JAVA_FLOAT, f32ByteOffset(base + i));
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF64 statsF64(double[] input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = input[base + i];
            total += value;
            totalSquares += value * value;
        }
        return new StatsF64(total / length, totalSquares / length);
    }

    private static StatsF64 statsF64(MemorySegment input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = input.get(JAVA_DOUBLE, f64ByteOffset(base + i));
            total += value;
            totalSquares += value * value;
        }
        return new StatsF64(total / length, totalSquares / length);
    }

    private static StatsF32 statsBf16(short[] input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(input[base + i]);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 statsBf16(MemorySegment input, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(
                    input.get(JAVA_SHORT, bf16ByteOffset(base + i))
            );
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
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

    private record StatsF32(float mean, double meanSquares) {
    }

    private record StatsF64(double mean, double meanSquares) {
    }
}
