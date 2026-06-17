package backend.cpu1.kernels.index.takealongaxis;

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
 * Dense contiguous take-along-axis loops.
 */
public final class Cpu1TakeAlongAxisLoops {
    private Cpu1TakeAlongAxisLoops() {
    }

    public static void takeAlongAxisF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] destination = output.float32Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] destination = output.float64Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBf16I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] destination = output.bfloat16Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBf16I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        int[] indexValues = indices.int32Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI32I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        long[] indexValues = indices.int64Array();
        int[] destination = output.int32Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI32I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        int[] indexValues = indices.int32Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI64I32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        long[] indexValues = indices.int64Array();
        long[] destination = output.int64Array();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI64I64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBoolI32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        int[] indexValues = indices.int32Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBoolI32Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBoolI64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        long[] indexValues = indices.int64Array();
        byte[] destination = output.boolArray();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBoolI64Range(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF32I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF32I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF64I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisF64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisF64I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBf16I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBf16I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBf16I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBf16I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI32I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI32I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI64I32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisI64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisI64I64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBoolI32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBoolI32SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    public static void takeAlongAxisBoolI64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        launch(unit, (startInclusive, endExclusive) ->
                takeAlongAxisBoolI64SegmentRange(source, indexValues, destination, unit, startInclusive, endExclusive));
    }

    private static void takeAlongAxisF32I32Range(
            float[] source,
            int[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisF32I64Range(
            float[] source,
            long[] indexValues,
            float[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisF64I32Range(
            double[] source,
            int[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisF64I64Range(
            double[] source,
            long[] indexValues,
            double[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisBf16I32Range(
            short[] source,
            int[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisBf16I64Range(
            short[] source,
            long[] indexValues,
            short[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisI32I32Range(
            int[] source,
            int[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisI32I64Range(
            int[] source,
            long[] indexValues,
            int[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisI64I32Range(
            long[] source,
            int[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisI64I64Range(
            long[] source,
            long[] indexValues,
            long[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisBoolI32Range(
            byte[] source,
            int[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisBoolI64Range(
            byte[] source,
            long[] indexValues,
            byte[] destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues[logical], axisSize);
            destination[logical] = source[sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize)];
        }
    }

    private static void takeAlongAxisF32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void takeAlongAxisF32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void takeAlongAxisF64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void takeAlongAxisF64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void takeAlongAxisBf16I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void takeAlongAxisBf16I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void takeAlongAxisI32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void takeAlongAxisI32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void takeAlongAxisI64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void takeAlongAxisI64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void takeAlongAxisBoolI32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static void takeAlongAxisBoolI64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            Cpu1PreparedIndexUnit unit,
            int startInclusive,
            int endExclusive
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexAxisSize = unit.indexAxisSize();
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int axisIndex = normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize);
            int sourceOffset = sourceOffset(logical, axisIndex, axisSize, innerSize, indexAxisSize);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static int sourceOffset(
            int outputLogical,
            int axisIndex,
            int axisSize,
            int innerSize,
            int indexAxisSize
    ) {
        int inner = outputLogical % innerSize;
        int outputBlock = outputLogical / innerSize;
        int outer = outputBlock / indexAxisSize;
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

    private static void launch(Cpu1PreparedIndexUnit unit, TakeAlongAxisRangeTask task) {
        unit.launchPolicy().launch(unit.outputElementCount(), task::run);
    }

    @FunctionalInterface
    private interface TakeAlongAxisRangeTask {
        void run(int startInclusive, int endExclusive);
    }
}
