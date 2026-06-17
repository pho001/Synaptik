package backend.cpu1.kernels.index.gatheraxis;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous ONNX-style gather-axis loops.
 */
public final class Cpu1GatherAxisLoops {
    private Cpu1GatherAxisLoops() {
    }

    public static void gatherAxisF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBf16I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBf16I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        int[] indexValues = indices.int32Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        long[] indexValues = indices.int64Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        int[] indexValues = indices.int32Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        long[] indexValues = indices.int64Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBoolI32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        int[] indexValues = indices.int32Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBoolI32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBoolI64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        long[] indexValues = indices.int64Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBoolI64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF32I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF32I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF64I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisF64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisF64I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBf16I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBf16I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBf16I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBf16I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI32I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI32I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI64I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisI64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisI64I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBoolI32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBoolI32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void gatherAxisBoolI64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                gatherAxisBoolI64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    private static void gatherAxisF32I32Range(
            float[] source,
            int[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisF32I64Range(
            float[] source,
            long[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisF64I32Range(
            double[] source,
            int[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisF64I64Range(
            double[] source,
            long[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisBf16I32Range(
            short[] source,
            int[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisBf16I64Range(
            short[] source,
            long[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisI32I32Range(
            int[] source,
            int[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisI32I64Range(
            int[] source,
            long[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisI64I32Range(
            long[] source,
            int[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisI64I64Range(
            long[] source,
            long[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisBoolI32Range(
            byte[] source,
            int[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisBoolI64Range(
            byte[] source,
            long[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[indexLogical(logical, innerSize, indexElementCount)], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount)];
        }
    }

    private static void gatherAxisF32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void gatherAxisF32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void gatherAxisF64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void gatherAxisF64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void gatherAxisBf16I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void gatherAxisBf16I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void gatherAxisI32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void gatherAxisI32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void gatherAxisI64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void gatherAxisI64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void gatherAxisBoolI32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static void gatherAxisBoolI64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int indexLogical = indexLogical(logical, innerSize, indexElementCount);
            int axisIndex = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    axisSize
            );
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexElementCount);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static int indexLogical(int outputLogical, int innerSize, int indexElementCount) {
        return (outputLogical / innerSize) % indexElementCount;
    }

    private static int sourceOffset(
            int outputLogical,
            int axisIndex,
            int axisSize,
            int innerSize,
            int indexElementCount
    ) {
        int inner = outputLogical % innerSize;
        int outputBlock = outputLogical / innerSize;
        int outer = outputBlock / indexElementCount;
        return (outer * axisSize + axisIndex) * innerSize + inner;
    }

    private static int normalizeIndex(long index, int axisSize) {
        long normalized = index < 0L ? index + axisSize : index;
        if (normalized < 0L || normalized >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + index
                    + " for axis size " + axisSize);
        }
        return (int) normalized;
    }

    private static void launch(Cpu1PreparedIndexUnit unit, GatherAxisRangeTask task) {
        unit.launchPolicy().launch(unit.outputElementCount(), task::run);
    }

    @FunctionalInterface
    private interface GatherAxisRangeTask {
        void run(int startInclusive, int endExclusive);
    }
}
