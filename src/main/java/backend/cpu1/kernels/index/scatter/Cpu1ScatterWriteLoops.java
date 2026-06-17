package backend.cpu1.kernels.index.scatter;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous deterministic scatter write loops for cpu1.
 */
public final class Cpu1ScatterWriteLoops {
    private Cpu1ScatterWriteLoops() {
    }

    public static void scatterElementsDenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        int[] targets = scatterElementsTargetsArray(unit, indices);
        switch (unit.valueDataType()) {
            case FLOAT32 -> {
                float[] destination = output.float32Array();
                copyF32Array(input.float32Array(), destination, unit.outputElementCount());
                applyF32Array(unit.reduction(), updates.float32Array(), destination, targets);
            }
            case FLOAT64 -> {
                double[] destination = output.float64Array();
                copyF64Array(input.float64Array(), destination, unit.outputElementCount());
                applyF64Array(unit.reduction(), updates.float64Array(), destination, targets);
            }
            case BFLOAT16 -> {
                short[] destination = output.bfloat16Array();
                copyBf16Array(input.bfloat16Array(), destination, unit.outputElementCount());
                applyBf16Array(unit.reduction(), updates.bfloat16Array(), destination, targets);
            }
            case INT32 -> {
                int[] destination = output.int32Array();
                copyI32Array(input.int32Array(), destination, unit.outputElementCount());
                applyI32Array(unit.reduction(), updates.int32Array(), destination, targets);
            }
            case INT64 -> {
                long[] destination = output.int64Array();
                copyI64Array(input.int64Array(), destination, unit.outputElementCount());
                applyI64Array(unit.reduction(), updates.int64Array(), destination, targets);
            }
            case BOOL -> {
                byte[] destination = output.boolArray();
                copyBoolArray(input.boolArray(), destination, unit.outputElementCount());
                applyBoolArray(unit.reduction(), updates.boolArray(), destination, targets);
            }
        }
    }

    public static void scatterElementsDenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        int[] targets = scatterElementsTargetsSegment(unit, indices);
        MemorySegment source = input.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        switch (unit.valueDataType()) {
            case FLOAT32 -> {
                copyF32Segment(source, destination, unit.outputElementCount());
                applyF32Segment(unit.reduction(), updateValues, destination, targets);
            }
            case FLOAT64 -> {
                copyF64Segment(source, destination, unit.outputElementCount());
                applyF64Segment(unit.reduction(), updateValues, destination, targets);
            }
            case BFLOAT16 -> {
                copyBf16Segment(source, destination, unit.outputElementCount());
                applyBf16Segment(unit.reduction(), updateValues, destination, targets);
            }
            case INT32 -> {
                copyI32Segment(source, destination, unit.outputElementCount());
                applyI32Segment(unit.reduction(), updateValues, destination, targets);
            }
            case INT64 -> {
                copyI64Segment(source, destination, unit.outputElementCount());
                applyI64Segment(unit.reduction(), updateValues, destination, targets);
            }
            case BOOL -> {
                copyBoolSegment(source, destination, unit.outputElementCount());
                applyBoolSegment(unit.reduction(), updateValues, destination, targets);
            }
        }
    }

    public static void scatterNdDenseArray(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        int[] targets = scatterNdTargetsArray(unit, indices);
        switch (unit.valueDataType()) {
            case FLOAT32 -> {
                float[] destination = output.float32Array();
                copyF32Array(input.float32Array(), destination, unit.outputElementCount());
                applyF32Array(unit.reduction(), updates.float32Array(), destination, targets);
            }
            case FLOAT64 -> {
                double[] destination = output.float64Array();
                copyF64Array(input.float64Array(), destination, unit.outputElementCount());
                applyF64Array(unit.reduction(), updates.float64Array(), destination, targets);
            }
            case BFLOAT16 -> {
                short[] destination = output.bfloat16Array();
                copyBf16Array(input.bfloat16Array(), destination, unit.outputElementCount());
                applyBf16Array(unit.reduction(), updates.bfloat16Array(), destination, targets);
            }
            case INT32 -> {
                int[] destination = output.int32Array();
                copyI32Array(input.int32Array(), destination, unit.outputElementCount());
                applyI32Array(unit.reduction(), updates.int32Array(), destination, targets);
            }
            case INT64 -> {
                long[] destination = output.int64Array();
                copyI64Array(input.int64Array(), destination, unit.outputElementCount());
                applyI64Array(unit.reduction(), updates.int64Array(), destination, targets);
            }
            case BOOL -> {
                byte[] destination = output.boolArray();
                copyBoolArray(input.boolArray(), destination, unit.outputElementCount());
                applyBoolArray(unit.reduction(), updates.boolArray(), destination, targets);
            }
        }
    }

    public static void scatterNdDenseSegment(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        int[] targets = scatterNdTargetsSegment(unit, indices);
        MemorySegment source = input.segment();
        MemorySegment updateValues = updates.segment();
        MemorySegment destination = output.segment();
        switch (unit.valueDataType()) {
            case FLOAT32 -> {
                copyF32Segment(source, destination, unit.outputElementCount());
                applyF32Segment(unit.reduction(), updateValues, destination, targets);
            }
            case FLOAT64 -> {
                copyF64Segment(source, destination, unit.outputElementCount());
                applyF64Segment(unit.reduction(), updateValues, destination, targets);
            }
            case BFLOAT16 -> {
                copyBf16Segment(source, destination, unit.outputElementCount());
                applyBf16Segment(unit.reduction(), updateValues, destination, targets);
            }
            case INT32 -> {
                copyI32Segment(source, destination, unit.outputElementCount());
                applyI32Segment(unit.reduction(), updateValues, destination, targets);
            }
            case INT64 -> {
                copyI64Segment(source, destination, unit.outputElementCount());
                applyI64Segment(unit.reduction(), updateValues, destination, targets);
            }
            case BOOL -> {
                copyBoolSegment(source, destination, unit.outputElementCount());
                applyBoolSegment(unit.reduction(), updateValues, destination, targets);
            }
        }
    }

    private static int[] scatterElementsTargetsArray(Cpu1PreparedIndexUnit unit, Cpu1TensorView indices) {
        return unit.indexDataType() == DataType.INT32
                ? scatterElementsTargetsI32Array(unit, indices.int32Array())
                : scatterElementsTargetsI64Array(unit, indices.int64Array());
    }

    private static int[] scatterElementsTargetsSegment(Cpu1PreparedIndexUnit unit, Cpu1TensorView indices) {
        return unit.indexDataType() == DataType.INT32
                ? scatterElementsTargetsI32Segment(unit, indices.segment())
                : scatterElementsTargetsI64Segment(unit, indices.segment());
    }

    private static int[] scatterElementsTargetsI32Array(Cpu1PreparedIndexUnit unit, int[] indexValues) {
        ScatterElementsPlan plan = ScatterElementsPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, normalizeIndex(indexValues[logical], plan.axisSize(), "ScatterElements"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterElementsTargetsI64Array(Cpu1PreparedIndexUnit unit, long[] indexValues) {
        ScatterElementsPlan plan = ScatterElementsPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, normalizeIndex(indexValues[logical], plan.axisSize(), "ScatterElements"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterElementsTargetsI32Segment(Cpu1PreparedIndexUnit unit, MemorySegment indexValues) {
        ScatterElementsPlan plan = ScatterElementsPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_INT, (long) logical * Integer.BYTES), plan.axisSize(), "ScatterElements")
            );
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterElementsTargetsI64Segment(Cpu1PreparedIndexUnit unit, MemorySegment indexValues) {
        ScatterElementsPlan plan = ScatterElementsPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(
                    logical,
                    normalizeIndex(indexValues.get(JAVA_LONG, (long) logical * Long.BYTES), plan.axisSize(), "ScatterElements")
            );
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterNdTargetsArray(Cpu1PreparedIndexUnit unit, Cpu1TensorView indices) {
        return unit.indexDataType() == DataType.INT32
                ? scatterNdTargetsI32Array(unit, indices.int32Array())
                : scatterNdTargetsI64Array(unit, indices.int64Array());
    }

    private static int[] scatterNdTargetsSegment(Cpu1PreparedIndexUnit unit, Cpu1TensorView indices) {
        return unit.indexDataType() == DataType.INT32
                ? scatterNdTargetsI32Segment(unit, indices.segment())
                : scatterNdTargetsI64Segment(unit, indices.segment());
    }

    private static int[] scatterNdTargetsI32Array(Cpu1PreparedIndexUnit unit, int[] indexValues) {
        ScatterNdPlan plan = ScatterNdPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        int[] coords = new int[plan.updatesShape().length];
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, coords, (indexLogical, dimensionSize) ->
                    normalizeIndex(indexValues[indexLogical], dimensionSize, "ScatterND"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterNdTargetsI64Array(Cpu1PreparedIndexUnit unit, long[] indexValues) {
        ScatterNdPlan plan = ScatterNdPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        int[] coords = new int[plan.updatesShape().length];
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, coords, (indexLogical, dimensionSize) ->
                    normalizeIndex(indexValues[indexLogical], dimensionSize, "ScatterND"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterNdTargetsI32Segment(Cpu1PreparedIndexUnit unit, MemorySegment indexValues) {
        ScatterNdPlan plan = ScatterNdPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        int[] coords = new int[plan.updatesShape().length];
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, coords, (indexLogical, dimensionSize) ->
                    normalizeIndex(indexValues.get(JAVA_INT, (long) indexLogical * Integer.BYTES), dimensionSize, "ScatterND"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static int[] scatterNdTargetsI64Segment(Cpu1PreparedIndexUnit unit, MemorySegment indexValues) {
        ScatterNdPlan plan = ScatterNdPlan.from(unit);
        int[] targets = new int[unit.updateElementCount()];
        DuplicateState duplicateState = DuplicateState.forReduction(unit);
        int[] coords = new int[plan.updatesShape().length];
        for (int logical = 0; logical < targets.length; logical++) {
            int target = plan.targetOffset(logical, coords, (indexLogical, dimensionSize) ->
                    normalizeIndex(indexValues.get(JAVA_LONG, (long) indexLogical * Long.BYTES), dimensionSize, "ScatterND"));
            duplicateState.mark(target);
            targets[logical] = target;
        }
        return targets;
    }

    private static void applyF32Array(ScatterReduction reduction, float[] updates, float[] destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = updates[i];
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] += updates[i];
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] *= updates[i];
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.max(destination[targets[i]], updates[i]);
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.min(destination[targets[i]], updates[i]);
                }
            }
        }
    }

    private static void applyF64Array(ScatterReduction reduction, double[] updates, double[] destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = updates[i];
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] += updates[i];
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] *= updates[i];
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.max(destination[targets[i]], updates[i]);
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.min(destination[targets[i]], updates[i]);
                }
            }
        }
    }

    private static void applyBf16Array(ScatterReduction reduction, short[] updates, short[] destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = updates[i];
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    float value = TensorDTypeOps.fromBFloat16Bits(destination[targets[i]])
                            + TensorDTypeOps.fromBFloat16Bits(updates[i]);
                    destination[targets[i]] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    float value = TensorDTypeOps.fromBFloat16Bits(destination[targets[i]])
                            * TensorDTypeOps.fromBFloat16Bits(updates[i]);
                    destination[targets[i]] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    float value = Math.max(
                            TensorDTypeOps.fromBFloat16Bits(destination[targets[i]]),
                            TensorDTypeOps.fromBFloat16Bits(updates[i])
                    );
                    destination[targets[i]] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    float value = Math.min(
                            TensorDTypeOps.fromBFloat16Bits(destination[targets[i]]),
                            TensorDTypeOps.fromBFloat16Bits(updates[i])
                    );
                    destination[targets[i]] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
        }
    }

    private static void applyI32Array(ScatterReduction reduction, int[] updates, int[] destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = updates[i];
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.addExact(destination[targets[i]], updates[i]);
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.multiplyExact(destination[targets[i]], updates[i]);
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.max(destination[targets[i]], updates[i]);
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.min(destination[targets[i]], updates[i]);
                }
            }
        }
    }

    private static void applyI64Array(ScatterReduction reduction, long[] updates, long[] destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = updates[i];
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.addExact(destination[targets[i]], updates[i]);
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.multiplyExact(destination[targets[i]], updates[i]);
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.max(destination[targets[i]], updates[i]);
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    destination[targets[i]] = Math.min(destination[targets[i]], updates[i]);
                }
            }
        }
    }

    private static void applyBoolArray(ScatterReduction reduction, byte[] updates, byte[] destination, int[] targets) {
        if (reduction != ScatterReduction.NONE) {
            throw new UnsupportedOperationException("cpu1 BOOL scatter supports only NONE reduction.");
        }
        for (int i = 0; i < targets.length; i++) {
            destination[targets[i]] = updates[i] == 0 ? (byte) 0 : (byte) 1;
        }
    }

    private static void applyF32Segment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination.set(JAVA_FLOAT, (long) targets[i] * Float.BYTES,
                            updates.get(JAVA_FLOAT, (long) i * Float.BYTES));
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Float.BYTES;
                    destination.set(JAVA_FLOAT, targetByte,
                            destination.get(JAVA_FLOAT, targetByte) + updates.get(JAVA_FLOAT, (long) i * Float.BYTES));
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Float.BYTES;
                    destination.set(JAVA_FLOAT, targetByte,
                            destination.get(JAVA_FLOAT, targetByte) * updates.get(JAVA_FLOAT, (long) i * Float.BYTES));
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Float.BYTES;
                    destination.set(JAVA_FLOAT, targetByte,
                            Math.max(destination.get(JAVA_FLOAT, targetByte), updates.get(JAVA_FLOAT, (long) i * Float.BYTES)));
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Float.BYTES;
                    destination.set(JAVA_FLOAT, targetByte,
                            Math.min(destination.get(JAVA_FLOAT, targetByte), updates.get(JAVA_FLOAT, (long) i * Float.BYTES)));
                }
            }
        }
    }

    private static void applyF64Segment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination.set(JAVA_DOUBLE, (long) targets[i] * Double.BYTES,
                            updates.get(JAVA_DOUBLE, (long) i * Double.BYTES));
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Double.BYTES;
                    destination.set(JAVA_DOUBLE, targetByte,
                            destination.get(JAVA_DOUBLE, targetByte) + updates.get(JAVA_DOUBLE, (long) i * Double.BYTES));
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Double.BYTES;
                    destination.set(JAVA_DOUBLE, targetByte,
                            destination.get(JAVA_DOUBLE, targetByte) * updates.get(JAVA_DOUBLE, (long) i * Double.BYTES));
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Double.BYTES;
                    destination.set(JAVA_DOUBLE, targetByte,
                            Math.max(destination.get(JAVA_DOUBLE, targetByte), updates.get(JAVA_DOUBLE, (long) i * Double.BYTES)));
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Double.BYTES;
                    destination.set(JAVA_DOUBLE, targetByte,
                            Math.min(destination.get(JAVA_DOUBLE, targetByte), updates.get(JAVA_DOUBLE, (long) i * Double.BYTES)));
                }
            }
        }
    }

    private static void applyBf16Segment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination.set(JAVA_SHORT, (long) targets[i] * Short.BYTES,
                            updates.get(JAVA_SHORT, (long) i * Short.BYTES));
                }
            }
            case ADD -> applyBf16SegmentReduction(updates, destination, targets, Bf16Reduction.ADD);
            case MUL -> applyBf16SegmentReduction(updates, destination, targets, Bf16Reduction.MUL);
            case MAX -> applyBf16SegmentReduction(updates, destination, targets, Bf16Reduction.MAX);
            case MIN -> applyBf16SegmentReduction(updates, destination, targets, Bf16Reduction.MIN);
        }
    }

    private static void applyBf16SegmentReduction(
            MemorySegment updates,
            MemorySegment destination,
            int[] targets,
            Bf16Reduction reduction
    ) {
        for (int i = 0; i < targets.length; i++) {
            long targetByte = (long) targets[i] * Short.BYTES;
            float current = TensorDTypeOps.fromBFloat16Bits(destination.get(JAVA_SHORT, targetByte));
            float update = TensorDTypeOps.fromBFloat16Bits(updates.get(JAVA_SHORT, (long) i * Short.BYTES));
            float value = switch (reduction) {
                case ADD -> current + update;
                case MUL -> current * update;
                case MAX -> Math.max(current, update);
                case MIN -> Math.min(current, update);
            };
            destination.set(JAVA_SHORT, targetByte, TensorDTypeOps.toBFloat16Bits(value));
        }
    }

    private static void applyI32Segment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination.set(JAVA_INT, (long) targets[i] * Integer.BYTES,
                            updates.get(JAVA_INT, (long) i * Integer.BYTES));
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Integer.BYTES;
                    destination.set(JAVA_INT, targetByte,
                            Math.addExact(destination.get(JAVA_INT, targetByte), updates.get(JAVA_INT, (long) i * Integer.BYTES)));
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Integer.BYTES;
                    destination.set(JAVA_INT, targetByte,
                            Math.multiplyExact(destination.get(JAVA_INT, targetByte), updates.get(JAVA_INT, (long) i * Integer.BYTES)));
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Integer.BYTES;
                    destination.set(JAVA_INT, targetByte,
                            Math.max(destination.get(JAVA_INT, targetByte), updates.get(JAVA_INT, (long) i * Integer.BYTES)));
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Integer.BYTES;
                    destination.set(JAVA_INT, targetByte,
                            Math.min(destination.get(JAVA_INT, targetByte), updates.get(JAVA_INT, (long) i * Integer.BYTES)));
                }
            }
        }
    }

    private static void applyI64Segment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        switch (reduction) {
            case NONE -> {
                for (int i = 0; i < targets.length; i++) {
                    destination.set(JAVA_LONG, (long) targets[i] * Long.BYTES,
                            updates.get(JAVA_LONG, (long) i * Long.BYTES));
                }
            }
            case ADD -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Long.BYTES;
                    destination.set(JAVA_LONG, targetByte,
                            Math.addExact(destination.get(JAVA_LONG, targetByte), updates.get(JAVA_LONG, (long) i * Long.BYTES)));
                }
            }
            case MUL -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Long.BYTES;
                    destination.set(JAVA_LONG, targetByte,
                            Math.multiplyExact(destination.get(JAVA_LONG, targetByte), updates.get(JAVA_LONG, (long) i * Long.BYTES)));
                }
            }
            case MAX -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Long.BYTES;
                    destination.set(JAVA_LONG, targetByte,
                            Math.max(destination.get(JAVA_LONG, targetByte), updates.get(JAVA_LONG, (long) i * Long.BYTES)));
                }
            }
            case MIN -> {
                for (int i = 0; i < targets.length; i++) {
                    long targetByte = (long) targets[i] * Long.BYTES;
                    destination.set(JAVA_LONG, targetByte,
                            Math.min(destination.get(JAVA_LONG, targetByte), updates.get(JAVA_LONG, (long) i * Long.BYTES)));
                }
            }
        }
    }

    private static void applyBoolSegment(ScatterReduction reduction, MemorySegment updates, MemorySegment destination, int[] targets) {
        if (reduction != ScatterReduction.NONE) {
            throw new UnsupportedOperationException("cpu1 BOOL scatter supports only NONE reduction.");
        }
        for (int i = 0; i < targets.length; i++) {
            byte update = updates.get(JAVA_BYTE, i);
            destination.set(JAVA_BYTE, targets[i], update == 0 ? (byte) 0 : (byte) 1);
        }
    }

    private static void copyF32Array(float[] source, float[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyF64Array(double[] source, double[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyBf16Array(short[] source, short[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyI32Array(int[] source, int[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyI64Array(long[] source, long[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyBoolArray(byte[] source, byte[] destination, int elements) {
        System.arraycopy(source, 0, destination, 0, elements);
    }

    private static void copyF32Segment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_FLOAT, (long) i * Float.BYTES, source.get(JAVA_FLOAT, (long) i * Float.BYTES));
        }
    }

    private static void copyF64Segment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_DOUBLE, (long) i * Double.BYTES, source.get(JAVA_DOUBLE, (long) i * Double.BYTES));
        }
    }

    private static void copyBf16Segment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_SHORT, (long) i * Short.BYTES, source.get(JAVA_SHORT, (long) i * Short.BYTES));
        }
    }

    private static void copyI32Segment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_INT, (long) i * Integer.BYTES, source.get(JAVA_INT, (long) i * Integer.BYTES));
        }
    }

    private static void copyI64Segment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_LONG, (long) i * Long.BYTES, source.get(JAVA_LONG, (long) i * Long.BYTES));
        }
    }

    private static void copyBoolSegment(MemorySegment source, MemorySegment destination, int elements) {
        for (int i = 0; i < elements; i++) {
            destination.set(JAVA_BYTE, i, source.get(JAVA_BYTE, i));
        }
    }

    private static int normalizeIndex(long index, int dimensionSize, String opName) {
        long normalized = index < 0L ? index + dimensionSize : index;
        if (normalized < 0L || normalized >= dimensionSize) {
            throw new IllegalArgumentException(opName + " index out of bounds: " + index
                    + " for dimension size " + dimensionSize);
        }
        return (int) normalized;
    }

    @FunctionalInterface
    private interface NdIndexReader {
        int read(int indexLogical, int dimensionSize);
    }

    private enum Bf16Reduction {
        ADD,
        MUL,
        MAX,
        MIN
    }

    private static final class DuplicateState {
        private final boolean[] seen;
        private final String opName;

        private DuplicateState(boolean[] seen, String opName) {
            this.seen = seen;
            this.opName = opName;
        }

        static DuplicateState forReduction(Cpu1PreparedIndexUnit unit) {
            if (unit.reduction() == ScatterReduction.NONE) {
                return new DuplicateState(new boolean[unit.outputElementCount()], unit.opType().name());
            }
            return new DuplicateState(null, unit.opType().name());
        }

        void mark(int target) {
            if (seen == null) {
                return;
            }
            if (seen[target]) {
                throw new IllegalArgumentException("cpu1 " + opName
                        + " NONE reduction does not allow duplicate target indices.");
            }
            seen[target] = true;
        }
    }

    private record ScatterElementsPlan(
            int axis,
            int axisSize,
            int[] dataDense,
            int[] updatesShape,
            int[] updatesDense
    ) {
        static ScatterElementsPlan from(Cpu1PreparedIndexUnit unit) {
            return new ScatterElementsPlan(
                    unit.dimension(),
                    unit.axisSize(),
                    unit.gatherNdInputStrides(),
                    unit.gatherNdOutputShape(),
                    unit.gatherNdOutputDenseStrides()
            );
        }

        int targetOffset(int logical, int axisIndex) {
            int rem = logical;
            int target = 0;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                target += (d == axis ? axisIndex : coord) * dataDense[d];
            }
            return target;
        }
    }

    private record ScatterNdPlan(
            int batchDims,
            int tupleRank,
            int prefixRank,
            int tupleStride,
            int[] dataShape,
            int[] dataDense,
            int[] indicesDense,
            int[] updatesShape,
            int[] updatesDense
    ) {
        static ScatterNdPlan from(Cpu1PreparedIndexUnit unit) {
            return new ScatterNdPlan(
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

        int targetOffset(int logical, int[] coords, NdIndexReader indexReader) {
            int rem = logical;
            for (int d = 0; d < updatesShape.length; d++) {
                coords[d] = rem / updatesDense[d];
                rem %= updatesDense[d];
            }
            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }
            int target = 0;
            for (int d = 0; d < batchDims; d++) {
                target += coords[d] * dataDense[d];
            }
            for (int d = 0; d < tupleRank; d++) {
                int dataDim = batchDims + d;
                int targetCoord = indexReader.read(indexBaseLogical + d * tupleStride, dataShape[dataDim]);
                target += targetCoord * dataDense[dataDim];
            }
            for (int d = batchDims + tupleRank; d < dataShape.length; d++) {
                int coordIndex = prefixRank + d - batchDims - tupleRank;
                target += coords[coordIndex] * dataDense[d];
            }
            return target;
        }
    }
}
