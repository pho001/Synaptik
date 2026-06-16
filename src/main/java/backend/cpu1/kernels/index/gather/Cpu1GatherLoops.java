package backend.cpu1.kernels.index.gather;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;

/**
 * Dense contiguous JAVA_ARRAY gather loops.
 */
public final class Cpu1GatherLoops {
    private Cpu1GatherLoops() {
    }

    public static void gatherF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherF32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherF32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherF64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherF64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherBf16I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherBf16I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherI32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        int[] indexValues = indices.int32Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherI32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherI32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        long[] indexValues = indices.int64Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherI32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherI64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        int[] indexValues = indices.int32Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherI64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherI64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        long[] indexValues = indices.int64Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherI64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherBoolI32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        int[] indexValues = indices.int32Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                gatherBoolI32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherBoolI64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        long[] indexValues = indices.int64Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                gatherBoolI64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    private static void gatherF32I32Range(
            float[] source,
            int[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherF32I64Range(
            float[] source,
            long[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherF64I32Range(
            double[] source,
            int[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherF64I64Range(
            double[] source,
            long[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherBf16I32Range(
            short[] source,
            int[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherBf16I64Range(
            short[] source,
            long[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherI32I32Range(
            int[] source,
            int[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherI32I64Range(
            int[] source,
            long[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherI64I32Range(
            long[] source,
            int[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherI64I64Range(
            long[] source,
            long[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherBoolI32Range(
            byte[] source,
            int[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static void gatherBoolI64Range(
            byte[] source,
            long[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = validateIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize)];
        }
    }

    private static int sourceOffset(int outputLogical, int axisIndex, int axisSize, int innerSize) {
        int outer = outputLogical / innerSize;
        int inner = outputLogical - outer * innerSize;
        return (outer * axisSize + axisIndex) * innerSize + inner;
    }

    private static int validateIndex(long index, int axisSize) {
        if (index < 0L || index >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + index
                    + " for axis size " + axisSize);
        }
        return (int) index;
    }

    private static void launch(Cpu1PreparedIndexUnit unit, GatherRangeTask task) {
        unit.launchPolicy().launch(unit.outputElementCount(), task::run);
    }

    @FunctionalInterface
    private interface GatherRangeTask {
        void run(int startInclusive, int endExclusive);
    }
}
