package backend.cpu1.kernels.index.gathernd;

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
 * Dense contiguous GatherND loops.
 */
public final class Cpu1GatherNdLoops {
    private Cpu1GatherNdLoops() {
    }

    public static void gatherNdF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] destination = output.float32Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF32I32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] destination = output.float32Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF32I64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] destination = output.float64Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF64I32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] destination = output.float64Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF64I64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] destination = output.bfloat16Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBf16I32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] destination = output.bfloat16Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBf16I64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        int[] indexValues = indices.int32Array();
        int[] destination = output.int32Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI32I32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        int[] source = input.int32Array();
        long[] indexValues = indices.int64Array();
        int[] destination = output.int32Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI32I64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        int[] indexValues = indices.int32Array();
        long[] destination = output.int64Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI64I32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        long[] source = input.int64Array();
        long[] indexValues = indices.int64Array();
        long[] destination = output.int64Array();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI64I64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBoolI32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        int[] indexValues = indices.int32Array();
        byte[] destination = output.boolArray();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBoolI32Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBoolI64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        byte[] source = input.boolArray();
        long[] indexValues = indices.int64Array();
        byte[] destination = output.boolArray();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBoolI64Range(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF32I32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF32I64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF64I32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdF64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdF64I64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBf16I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBf16I32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBf16I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBf16I64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI32I32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI32I64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI64I32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdI64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdI64I64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBoolI32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBoolI32SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    public static void gatherNdBoolI64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment destination = output.segment();
        GatherNdDensePlan plan = GatherNdDensePlan.from(unit);
        launch(unit, (startInclusive, endExclusive) ->
                gatherNdBoolI64SegmentRange(source, indexValues, destination, plan, startInclusive, endExclusive));
    }

    private static void gatherNdF32I32Range(
            float[] source,
            int[] indexValues,
            float[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdF32I64Range(
            float[] source,
            long[] indexValues,
            float[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdF64I32Range(
            double[] source,
            int[] indexValues,
            double[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdF64I64Range(
            double[] source,
            long[] indexValues,
            double[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdBf16I32Range(
            short[] source,
            int[] indexValues,
            short[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdBf16I64Range(
            short[] source,
            long[] indexValues,
            short[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdI32I32Range(
            int[] source,
            int[] indexValues,
            int[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdI32I64Range(
            int[] source,
            long[] indexValues,
            int[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdI64I32Range(
            long[] source,
            int[] indexValues,
            long[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdI64I64Range(
            long[] source,
            long[] indexValues,
            long[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdBoolI32Range(
            byte[] source,
            int[] indexValues,
            byte[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI32(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdBoolI64Range(
            byte[] source,
            long[] indexValues,
            byte[] destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            destination[logical] = source[sourceOffsetI64(indexValues, plan, logical, coords)];
        }
    }

    private static void gatherNdF32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void gatherNdF32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_FLOAT,
                    (long) logical * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceOffset * Float.BYTES)
            );
        }
    }

    private static void gatherNdF64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void gatherNdF64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_DOUBLE,
                    (long) logical * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceOffset * Double.BYTES)
            );
        }
    }

    private static void gatherNdBf16I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void gatherNdBf16I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_SHORT,
                    (long) logical * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceOffset * Short.BYTES)
            );
        }
    }

    private static void gatherNdI32I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void gatherNdI32I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_INT,
                    (long) logical * Integer.BYTES,
                    source.get(JAVA_INT, (long) sourceOffset * Integer.BYTES)
            );
        }
    }

    private static void gatherNdI64I32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void gatherNdI64I64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_LONG,
                    (long) logical * Long.BYTES,
                    source.get(JAVA_LONG, (long) sourceOffset * Long.BYTES)
            );
        }
    }

    private static void gatherNdBoolI32SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI32(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static void gatherNdBoolI64SegmentRange(
            MemorySegment source,
            MemorySegment indexValues,
            MemorySegment destination,
            GatherNdDensePlan plan,
            int startInclusive,
            int endExclusive
    ) {
        int[] coords = new int[plan.outputShape.length];
        for (int logical = startInclusive; logical < endExclusive; logical++) {
            int sourceOffset = sourceOffsetI64(indexValues, plan, logical, coords);
            destination.set(
                    JAVA_BYTE,
                    (long) logical * Byte.BYTES,
                    source.get(JAVA_BYTE, (long) sourceOffset * Byte.BYTES)
            );
        }
    }

    private static int sourceOffsetI32(
            int[] indexValues,
            GatherNdDensePlan plan,
            int outputLogical,
            int[] coords
    ) {
        fillOutputCoords(outputLogical, plan, coords);
        int indexBaseLogical = indexBaseLogical(plan, coords);
        int source = batchSourceOffset(plan, coords);
        for (int d = 0; d < plan.tupleRank; d++) {
            int inputDim = plan.batchDims + d;
            int coord = normalizeIndex(indexValues[indexBaseLogical + d * plan.tupleStride], plan.inputShape[inputDim]);
            source += coord * plan.inputStrides[inputDim];
        }
        return source + suffixSourceOffset(plan, coords);
    }

    private static int sourceOffsetI64(
            long[] indexValues,
            GatherNdDensePlan plan,
            int outputLogical,
            int[] coords
    ) {
        fillOutputCoords(outputLogical, plan, coords);
        int indexBaseLogical = indexBaseLogical(plan, coords);
        int source = batchSourceOffset(plan, coords);
        for (int d = 0; d < plan.tupleRank; d++) {
            int inputDim = plan.batchDims + d;
            int coord = normalizeIndex(indexValues[indexBaseLogical + d * plan.tupleStride], plan.inputShape[inputDim]);
            source += coord * plan.inputStrides[inputDim];
        }
        return source + suffixSourceOffset(plan, coords);
    }

    private static int sourceOffsetI32(
            MemorySegment indexValues,
            GatherNdDensePlan plan,
            int outputLogical,
            int[] coords
    ) {
        fillOutputCoords(outputLogical, plan, coords);
        int indexBaseLogical = indexBaseLogical(plan, coords);
        int source = batchSourceOffset(plan, coords);
        for (int d = 0; d < plan.tupleRank; d++) {
            int inputDim = plan.batchDims + d;
            int indexLogical = indexBaseLogical + d * plan.tupleStride;
            int coord = normalizeIndex(
                    indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES),
                    plan.inputShape[inputDim]
            );
            source += coord * plan.inputStrides[inputDim];
        }
        return source + suffixSourceOffset(plan, coords);
    }

    private static int sourceOffsetI64(
            MemorySegment indexValues,
            GatherNdDensePlan plan,
            int outputLogical,
            int[] coords
    ) {
        fillOutputCoords(outputLogical, plan, coords);
        int indexBaseLogical = indexBaseLogical(plan, coords);
        int source = batchSourceOffset(plan, coords);
        for (int d = 0; d < plan.tupleRank; d++) {
            int inputDim = plan.batchDims + d;
            int indexLogical = indexBaseLogical + d * plan.tupleStride;
            int coord = normalizeIndex(
                    indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES),
                    plan.inputShape[inputDim]
            );
            source += coord * plan.inputStrides[inputDim];
        }
        return source + suffixSourceOffset(plan, coords);
    }

    private static void fillOutputCoords(int outputLogical, GatherNdDensePlan plan, int[] coords) {
        int rem = outputLogical;
        for (int d = 0; d < plan.outputShape.length; d++) {
            int coord = rem / plan.outputDenseStrides[d];
            rem -= coord * plan.outputDenseStrides[d];
            coords[d] = coord;
        }
    }

    private static int indexBaseLogical(GatherNdDensePlan plan, int[] coords) {
        int indexBaseLogical = 0;
        for (int d = 0; d < plan.prefixRank; d++) {
            indexBaseLogical += coords[d] * plan.indicesDenseStrides[d];
        }
        return indexBaseLogical;
    }

    private static int batchSourceOffset(GatherNdDensePlan plan, int[] coords) {
        int source = 0;
        for (int d = 0; d < plan.batchDims; d++) {
            source += coords[d] * plan.inputStrides[d];
        }
        return source;
    }

    private static int suffixSourceOffset(GatherNdDensePlan plan, int[] coords) {
        int source = 0;
        for (int d = plan.batchDims + plan.tupleRank; d < plan.inputShape.length; d++) {
            int suffixCoord = coords[plan.prefixRank + d - plan.batchDims - plan.tupleRank];
            source += suffixCoord * plan.inputStrides[d];
        }
        return source;
    }

    private static int normalizeIndex(long index, int dimensionSize) {
        long normalized = index < 0L ? index + dimensionSize : index;
        if (normalized < 0L || normalized >= dimensionSize) {
            throw new IllegalArgumentException("GatherND index out of bounds: " + index
                    + " for dimension size " + dimensionSize);
        }
        return (int) normalized;
    }

    private static void launch(Cpu1PreparedIndexUnit unit, GatherNdRangeTask task) {
        unit.launchPolicy().launch(unit.outputElementCount(), task::run);
    }

    @FunctionalInterface
    private interface GatherNdRangeTask {
        void run(int startInclusive, int endExclusive);
    }

    private record GatherNdDensePlan(
            int batchDims,
            int tupleRank,
            int prefixRank,
            int tupleStride,
            int[] inputShape,
            int[] inputStrides,
            int[] indicesDenseStrides,
            int[] outputShape,
            int[] outputDenseStrides
    ) {
        static GatherNdDensePlan from(Cpu1PreparedIndexUnit unit) {
            return new GatherNdDensePlan(
                    unit.batchDims(),
                    unit.tupleRank(),
                    unit.prefixRank(),
                    unit.tupleStride(),
                    unit.gatherNdInputShape(),
                    unit.gatherNdInputStrides(),
                    unit.gatherNdIndicesDenseStrides(),
                    unit.gatherNdOutputShape(),
                    unit.gatherNdOutputDenseStrides()
            );
        }
    }
}
