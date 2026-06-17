package backend.cpu1.kernels.index.scatter;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous deterministic scatter-add loops.
 */
public final class Cpu1ScatterLoops {
    private Cpu1ScatterLoops() {
    }

    public static void scatterAddF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] updateValues = updates.float32Array();
        float[] destination = output.float32Array();
        copyF32Array(source, destination, unit.outputElementCount());
        scatterAddF32I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] updateValues = updates.float32Array();
        float[] destination = output.float32Array();
        copyF32Array(source, destination, unit.outputElementCount());
        scatterAddF32I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] updateValues = updates.float64Array();
        double[] destination = output.float64Array();
        copyF64Array(source, destination, unit.outputElementCount());
        scatterAddF64I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] updateValues = updates.float64Array();
        double[] destination = output.float64Array();
        copyF64Array(source, destination, unit.outputElementCount());
        scatterAddF64I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] updateValues = updates.bfloat16Array();
        short[] destination = output.bfloat16Array();
        copyBf16Array(source, destination, unit.outputElementCount());
        scatterAddBf16I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] updateValues = updates.bfloat16Array();
        short[] destination = output.bfloat16Array();
        copyBf16Array(source, destination, unit.outputElementCount());
        scatterAddBf16I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF32Segment(source, destination, unit.outputElementCount());
        scatterAddF32I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF32Segment(source, destination, unit.outputElementCount());
        scatterAddF32I64Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF64Segment(source, destination, unit.outputElementCount());
        scatterAddF64I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddF64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF64Segment(source, destination, unit.outputElementCount());
        scatterAddF64I64Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddBf16I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyBf16Segment(source, destination, unit.outputElementCount());
        scatterAddBf16I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAddBf16I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyBf16Segment(source, destination, unit.outputElementCount());
        scatterAddBf16I64Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF32I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        int[] indexValues = indices.int32Array();
        float[] updateValues = updates.float32Array();
        float[] destination = output.float32Array();
        copyF32Array(source, destination, unit.outputElementCount());
        scatterAxisAddF32I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF32I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        float[] source = input.float32Array();
        long[] indexValues = indices.int64Array();
        float[] updateValues = updates.float32Array();
        float[] destination = output.float32Array();
        copyF32Array(source, destination, unit.outputElementCount());
        scatterAxisAddF32I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF64I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        int[] indexValues = indices.int32Array();
        double[] updateValues = updates.float64Array();
        double[] destination = output.float64Array();
        copyF64Array(source, destination, unit.outputElementCount());
        scatterAxisAddF64I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF64I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        double[] source = input.float64Array();
        long[] indexValues = indices.int64Array();
        double[] updateValues = updates.float64Array();
        double[] destination = output.float64Array();
        copyF64Array(source, destination, unit.outputElementCount());
        scatterAxisAddF64I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddBf16I32DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        int[] indexValues = indices.int32Array();
        short[] updateValues = updates.bfloat16Array();
        short[] destination = output.bfloat16Array();
        copyBf16Array(source, destination, unit.outputElementCount());
        scatterAxisAddBf16I32Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddBf16I64DenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        short[] source = input.bfloat16Array();
        long[] indexValues = indices.int64Array();
        short[] updateValues = updates.bfloat16Array();
        short[] destination = output.bfloat16Array();
        copyBf16Array(source, destination, unit.outputElementCount());
        scatterAxisAddBf16I64Array(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF32I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF32Segment(source, destination, unit.outputElementCount());
        scatterAxisAddF32I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF32I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF32Segment(source, destination, unit.outputElementCount());
        scatterAxisAddF32I64Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF64I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF64Segment(source, destination, unit.outputElementCount());
        scatterAxisAddF64I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddF64I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyF64Segment(source, destination, unit.outputElementCount());
        scatterAxisAddF64I64Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddBf16I32DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyBf16Segment(source, destination, unit.outputElementCount());
        scatterAxisAddBf16I32Segment(unit, indexValues, updateValues, destination);
    }

    public static void scatterAxisAddBf16I64DenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        MemorySegment source = input.segment();
        MemorySegment indexValues = indices.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        copyBf16Segment(source, destination, unit.outputElementCount());
        scatterAxisAddBf16I64Segment(unit, indexValues, updateValues, destination);
    }

    private static void scatterAddF32I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            float[] updateValues,
            float[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int axisIndex = validateScatterAddIndex(indexValues[logical], axisSize);
            destination[scatterAddTargetOffset(logical, axisIndex, axisSize, innerSize)] += updateValues[logical];
        }
    }

    private static void scatterAddF32I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            float[] updateValues,
            float[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int axisIndex = validateScatterAddIndex(indexValues[logical], axisSize);
            destination[scatterAddTargetOffset(logical, axisIndex, axisSize, innerSize)] += updateValues[logical];
        }
    }

    private static void scatterAddF64I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            double[] updateValues,
            double[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int axisIndex = validateScatterAddIndex(indexValues[logical], axisSize);
            destination[scatterAddTargetOffset(logical, axisIndex, axisSize, innerSize)] += updateValues[logical];
        }
    }

    private static void scatterAddF64I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            double[] updateValues,
            double[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int axisIndex = validateScatterAddIndex(indexValues[logical], axisSize);
            destination[scatterAddTargetOffset(logical, axisIndex, axisSize, innerSize)] += updateValues[logical];
        }
    }

    private static void scatterAddBf16I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            short[] updateValues,
            short[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues[logical], axisSize),
                    axisSize,
                    innerSize
            );
            float sum = TensorDTypeOps.fromBFloat16Bits(destination[target])
                    + TensorDTypeOps.fromBFloat16Bits(updateValues[logical]);
            destination[target] = TensorDTypeOps.toBFloat16Bits(sum);
        }
    }

    private static void scatterAddBf16I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            short[] updateValues,
            short[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues[logical], axisSize),
                    axisSize,
                    innerSize
            );
            float sum = TensorDTypeOps.fromBFloat16Bits(destination[target])
                    + TensorDTypeOps.fromBFloat16Bits(updateValues[logical]);
            destination[target] = TensorDTypeOps.toBFloat16Bits(sum);
        }
    }

    private static void scatterAddF32I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Float.BYTES;
            float sum = destination.get(JAVA_FLOAT, targetByte)
                    + updateValues.get(JAVA_FLOAT, (long) logical * Float.BYTES);
            destination.set(JAVA_FLOAT, targetByte, sum);
        }
    }

    private static void scatterAddF32I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Float.BYTES;
            float sum = destination.get(JAVA_FLOAT, targetByte)
                    + updateValues.get(JAVA_FLOAT, (long) logical * Float.BYTES);
            destination.set(JAVA_FLOAT, targetByte, sum);
        }
    }

    private static void scatterAddF64I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Double.BYTES;
            double sum = destination.get(JAVA_DOUBLE, targetByte)
                    + updateValues.get(JAVA_DOUBLE, (long) logical * Double.BYTES);
            destination.set(JAVA_DOUBLE, targetByte, sum);
        }
    }

    private static void scatterAddF64I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Double.BYTES;
            double sum = destination.get(JAVA_DOUBLE, targetByte)
                    + updateValues.get(JAVA_DOUBLE, (long) logical * Double.BYTES);
            destination.set(JAVA_DOUBLE, targetByte, sum);
        }
    }

    private static void scatterAddBf16I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Short.BYTES;
            float sum = TensorDTypeOps.fromBFloat16Bits(destination.get(JAVA_SHORT, targetByte))
                    + TensorDTypeOps.fromBFloat16Bits(updateValues.get(JAVA_SHORT, (long) logical * Short.BYTES));
            destination.set(JAVA_SHORT, targetByte, TensorDTypeOps.toBFloat16Bits(sum));
        }
    }

    private static void scatterAddBf16I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int target = scatterAddTargetOffset(
                    logical,
                    validateScatterAddIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize
            );
            long targetByte = (long) target * Short.BYTES;
            float sum = TensorDTypeOps.fromBFloat16Bits(destination.get(JAVA_SHORT, targetByte))
                    + TensorDTypeOps.fromBFloat16Bits(updateValues.get(JAVA_SHORT, (long) logical * Short.BYTES));
            destination.set(JAVA_SHORT, targetByte, TensorDTypeOps.toBFloat16Bits(sum));
        }
    }

    private static void scatterAxisAddF32I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            float[] updateValues,
            float[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            destination[target] += updateValues[logical];
        }
    }

    private static void scatterAxisAddF32I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            float[] updateValues,
            float[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            destination[target] += updateValues[logical];
        }
    }

    private static void scatterAxisAddF64I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            double[] updateValues,
            double[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            destination[target] += updateValues[logical];
        }
    }

    private static void scatterAxisAddF64I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            double[] updateValues,
            double[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            destination[target] += updateValues[logical];
        }
    }

    private static void scatterAxisAddBf16I32Array(
            Cpu1PreparedIndexUnit unit,
            int[] indexValues,
            short[] updateValues,
            short[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            float sum = TensorDTypeOps.fromBFloat16Bits(destination[target])
                    + TensorDTypeOps.fromBFloat16Bits(updateValues[logical]);
            destination[target] = TensorDTypeOps.toBFloat16Bits(sum);
        }
    }

    private static void scatterAxisAddBf16I64Array(
            Cpu1PreparedIndexUnit unit,
            long[] indexValues,
            short[] updateValues,
            short[] destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues[indexLogical], axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            float sum = TensorDTypeOps.fromBFloat16Bits(destination[target])
                    + TensorDTypeOps.fromBFloat16Bits(updateValues[logical]);
            destination[target] = TensorDTypeOps.toBFloat16Bits(sum);
        }
    }

    private static void scatterAxisAddF32I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Float.BYTES;
            float sum = destination.get(JAVA_FLOAT, targetByte)
                    + updateValues.get(JAVA_FLOAT, (long) logical * Float.BYTES);
            destination.set(JAVA_FLOAT, targetByte, sum);
        }
    }

    private static void scatterAxisAddF32I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Float.BYTES;
            float sum = destination.get(JAVA_FLOAT, targetByte)
                    + updateValues.get(JAVA_FLOAT, (long) logical * Float.BYTES);
            destination.set(JAVA_FLOAT, targetByte, sum);
        }
    }

    private static void scatterAxisAddF64I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Double.BYTES;
            double sum = destination.get(JAVA_DOUBLE, targetByte)
                    + updateValues.get(JAVA_DOUBLE, (long) logical * Double.BYTES);
            destination.set(JAVA_DOUBLE, targetByte, sum);
        }
    }

    private static void scatterAxisAddF64I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Double.BYTES;
            double sum = destination.get(JAVA_DOUBLE, targetByte)
                    + updateValues.get(JAVA_DOUBLE, (long) logical * Double.BYTES);
            destination.set(JAVA_DOUBLE, targetByte, sum);
        }
    }

    private static void scatterAxisAddBf16I32Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Short.BYTES;
            float sum = TensorDTypeOps.fromBFloat16Bits(destination.get(JAVA_SHORT, targetByte))
                    + TensorDTypeOps.fromBFloat16Bits(updateValues.get(JAVA_SHORT, (long) logical * Short.BYTES));
            destination.set(JAVA_SHORT, targetByte, TensorDTypeOps.toBFloat16Bits(sum));
        }
    }

    private static void scatterAxisAddBf16I64Segment(
            Cpu1PreparedIndexUnit unit,
            MemorySegment indexValues,
            MemorySegment updateValues,
            MemorySegment destination
    ) {
        int axisSize = unit.axisSize();
        int innerSize = unit.innerSize();
        int indexElementCount = unit.indexElementCount();
        for (int logical = 0; logical < unit.updateElementCount(); logical++) {
            int indexLogical = axisIndexLogical(logical, innerSize, indexElementCount);
            int target = scatterAxisTargetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES), axisSize),
                    axisSize,
                    innerSize,
                    indexElementCount
            );
            long targetByte = (long) target * Short.BYTES;
            float sum = TensorDTypeOps.fromBFloat16Bits(destination.get(JAVA_SHORT, targetByte))
                    + TensorDTypeOps.fromBFloat16Bits(updateValues.get(JAVA_SHORT, (long) logical * Short.BYTES));
            destination.set(JAVA_SHORT, targetByte, TensorDTypeOps.toBFloat16Bits(sum));
        }
    }

    private static void copyF32Array(float[] source, float[] destination, int elementCount) {
        System.arraycopy(source, 0, destination, 0, elementCount);
    }

    private static void copyF64Array(double[] source, double[] destination, int elementCount) {
        System.arraycopy(source, 0, destination, 0, elementCount);
    }

    private static void copyBf16Array(short[] source, short[] destination, int elementCount) {
        System.arraycopy(source, 0, destination, 0, elementCount);
    }

    private static void copyF32Segment(MemorySegment source, MemorySegment destination, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            destination.set(JAVA_FLOAT, (long) i * Float.BYTES, source.get(JAVA_FLOAT, (long) i * Float.BYTES));
        }
    }

    private static void copyF64Segment(MemorySegment source, MemorySegment destination, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            destination.set(JAVA_DOUBLE, (long) i * Double.BYTES, source.get(JAVA_DOUBLE, (long) i * Double.BYTES));
        }
    }

    private static void copyBf16Segment(MemorySegment source, MemorySegment destination, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            destination.set(JAVA_SHORT, (long) i * Short.BYTES, source.get(JAVA_SHORT, (long) i * Short.BYTES));
        }
    }

    private static int scatterAddTargetOffset(int logical, int axisIndex, int axisSize, int innerSize) {
        int inner = logical % innerSize;
        int outer = logical / innerSize;
        return (outer * axisSize + axisIndex) * innerSize + inner;
    }

    private static int axisIndexLogical(int updateLogical, int innerSize, int indexElementCount) {
        return (updateLogical / innerSize) % indexElementCount;
    }

    private static int scatterAxisTargetOffset(
            int updateLogical,
            int axisIndex,
            int axisSize,
            int innerSize,
            int indexElementCount
    ) {
        int inner = updateLogical % innerSize;
        int updateBlock = updateLogical / innerSize;
        int outer = updateBlock / indexElementCount;
        return (outer * axisSize + axisIndex) * innerSize + inner;
    }

    private static int validateScatterAddIndex(long index, int axisSize) {
        if (index < 0L || index >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + index
                    + " for axis size " + axisSize);
        }
        return (int) index;
    }

    private static int normalizeIndex(long index, int axisSize) {
        long normalized = index < 0L ? index + axisSize : index;
        if (normalized < 0L || normalized >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + index
                    + " for axis size " + axisSize);
        }
        return (int) normalized;
    }
}
