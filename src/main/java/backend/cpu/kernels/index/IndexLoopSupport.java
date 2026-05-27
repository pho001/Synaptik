package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class IndexLoopSupport {
    private IndexLoopSupport() {
    }

    static int[] denseStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    static int offsetForLogical(int logical, int[] shape, int[] dense, int[] strides, int baseOffset) {
        int rem = logical;
        int offset = baseOffset;
        for (int d = 0; d < shape.length; d++) {
            int coord = rem / dense[d];
            rem %= dense[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    static void validateReadStorageViews(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            DataType valueDType
    ) {
        Objects.requireNonNull(inputView, "input storage view cannot be null");
        Objects.requireNonNull(indicesView, "indices storage view cannot be null");
        Objects.requireNonNull(outView, "output storage view cannot be null");
        requireViewMatchesTensor(input, inputView, valueDType, "input");
        requireViewMatchesTensor(indices, indicesView, indices.getDataType(), "indices");
        requireViewMatchesTensor(out, outView, valueDType, "output");
    }

    static void validateScatterStorageViews(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            DataType valueDType
    ) {
        Objects.requireNonNull(dataView, "data storage view cannot be null");
        Objects.requireNonNull(indicesView, "indices storage view cannot be null");
        Objects.requireNonNull(updatesView, "updates storage view cannot be null");
        Objects.requireNonNull(outView, "output storage view cannot be null");
        requireViewMatchesTensor(data, dataView, valueDType, "data");
        requireViewMatchesTensor(indices, indicesView, indices.getDataType(), "indices");
        requireViewMatchesTensor(updates, updatesView, valueDType, "updates");
        requireViewMatchesTensor(out, outView, valueDType, "output");
    }

    static void validateOutputStorageView(Tensor out, CpuStorageView outView, DataType valueDType) {
        Objects.requireNonNull(outView, "output storage view cannot be null");
        requireViewMatchesTensor(out, outView, valueDType, "output");
    }

    static IndexStoragePlan indexStoragePlan(CpuStorageView indices) {
        Objects.requireNonNull(indices, "indices storage view cannot be null");
        int[] shape = indices.shape();
        return new IndexStoragePlan(
                indices.dtype(),
                shape,
                denseStrides(shape),
                indices.strides(),
                indices.storageOffset()
        );
    }

    static int readAxisIndex(
            CpuStorageView indices,
            IndexStoragePlan plan,
            int logicalIndex,
            int axisSize
    ) {
        long integral;
        if (plan.dataType == DataType.INT32) {
            integral = readI32(indices, plan.offset(logicalIndex));
        } else if (plan.dataType == DataType.INT64) {
            integral = readI64(indices, plan.offset(logicalIndex));
        } else {
            double raw = readFloatingIndex(indices, plan, logicalIndex);
            if (!Double.isFinite(raw)) {
                throw new IllegalArgumentException("Gather index must be finite.");
            }
            integral = Math.round(raw);
            if (Math.abs(raw - integral) > 1e-9) {
                throw new IllegalArgumentException("Gather index must be an integer value. got=" + raw);
            }
        }
        if (integral < 0 || integral >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + integral + " for axis size " + axisSize);
        }
        return (int) integral;
    }

    static int readAxisIndexAllowNegative(
            CpuStorageView indices,
            IndexStoragePlan plan,
            int logicalIndex,
            int axisSize
    ) {
        long integral;
        double rawDouble = 0.0d;
        boolean floating = false;
        if (plan.dataType == DataType.INT32) {
            integral = readI32(indices, plan.offset(logicalIndex));
        } else if (plan.dataType == DataType.INT64) {
            integral = readI64(indices, plan.offset(logicalIndex));
        } else {
            rawDouble = readFloatingIndex(indices, plan, logicalIndex);
            floating = true;
            if (!Double.isFinite(rawDouble)) {
                throw new IllegalArgumentException("Gather index must be finite.");
            }
            integral = Math.round(rawDouble);
            if (Math.abs(rawDouble - integral) > 1e-9) {
                throw new IllegalArgumentException("Gather index must be an integer value. got=" + rawDouble);
            }
        }
        long rawIntegral = integral;
        if (integral < 0) {
            integral += axisSize;
        }
        if (integral < 0 || integral >= axisSize) {
            String raw = floating ? Double.toString(rawDouble) : Long.toString(rawIntegral);
            throw new IllegalArgumentException("Gather index out of bounds: " + raw + " for axis size " + axisSize);
        }
        return (int) integral;
    }

    static double readF64(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF64Array()[offset]
                : view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    static float readF32(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF32Array()[offset]
                : view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    static short readBF16Bits(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireBF16Array()[offset]
                : view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    static void writeBF16Bits(CpuStorageView view, int offset, short bits) {
        if (view.isArray()) {
            view.requireBF16Array()[offset] = bits;
        } else {
            view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    static int readI32(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireI32Array()[offset]
                : view.requireSegment().get(JAVA_INT, (long) offset * Integer.BYTES);
    }

    static void writeI32(CpuStorageView view, int offset, int value) {
        if (view.isArray()) {
            view.requireI32Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_INT, (long) offset * Integer.BYTES, value);
        }
    }

    static long readI64(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireI64Array()[offset]
                : view.requireSegment().get(JAVA_LONG, (long) offset * Long.BYTES);
    }

    static void writeI64(CpuStorageView view, int offset, long value) {
        if (view.isArray()) {
            view.requireI64Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_LONG, (long) offset * Long.BYTES, value);
        }
    }

    static byte readBool(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireBoolArray()[offset]
                : view.requireSegment().get(JAVA_BYTE, offset);
    }

    static void writeBool(CpuStorageView view, int offset, byte value) {
        if (view.isArray()) {
            view.requireBoolArray()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_BYTE, offset, value);
        }
    }

    static boolean allArrays(CpuStorageView first, CpuStorageView second, CpuStorageView third) {
        return first.isArray() && second.isArray() && third.isArray();
    }

    static boolean allArrays(CpuStorageView first, CpuStorageView second) {
        return first.isArray() && second.isArray();
    }

    static boolean allArrays(CpuStorageView first, CpuStorageView second, CpuStorageView third, CpuStorageView fourth) {
        return first.isArray() && second.isArray() && third.isArray() && fourth.isArray();
    }

    static void copyStorage(
            Tensor source,
            Tensor destination,
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            DataType dtype
    ) {
        int[] shape = source.getShapeUnsafe();
        int[] dense = denseStrides(shape);
        int total = source.getFlatDataSize();
        int[] sourceStrides = source.getStridesUnsafe();
        int[] destinationStrides = destination.getStridesUnsafe();
        int sourceBaseOffset = source.getStorageOffsetUnsafe();
        int destinationBaseOffset = destination.getStorageOffsetUnsafe();
        switch (dtype) {
            case FLOAT64 -> copyF64(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
            case FLOAT32 -> copyF32(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
            case BFLOAT16 -> copyBF16(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
            case BOOL -> copyBool(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
            case INT32 -> copyI32(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
            case INT64 -> copyI64(sourceView, destinationView, shape, dense, sourceStrides, destinationStrides,
                    sourceBaseOffset, destinationBaseOffset, total);
        }
    }

    static void zeroStorage(Tensor destination, CpuStorageView destinationView, DataType dtype) {
        int[] shape = destination.getShapeUnsafe();
        int[] dense = denseStrides(shape);
        int[] strides = destination.getStridesUnsafe();
        int baseOffset = destination.getStorageOffsetUnsafe();
        int total = destination.getFlatDataSize();
        switch (dtype) {
            case FLOAT64 -> {
                if (destinationView.isArray()) {
                    double[] dst = destinationView.requireF64Array();
                    for (int logical = 0; logical < total; logical++) {
                        dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = 0.0d;
                    }
                } else {
                    for (int logical = 0; logical < total; logical++) {
                        writeF64(destinationView, offsetForLogical(logical, shape, dense, strides, baseOffset), 0.0d);
                    }
                }
            }
            case FLOAT32 -> {
                if (destinationView.isArray()) {
                    float[] dst = destinationView.requireF32Array();
                    for (int logical = 0; logical < total; logical++) {
                        dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = 0.0f;
                    }
                } else {
                    for (int logical = 0; logical < total; logical++) {
                        writeF32(destinationView, offsetForLogical(logical, shape, dense, strides, baseOffset), 0.0f);
                    }
                }
            }
            case BFLOAT16 -> {
                short zero = TensorDTypeOps.toBFloat16Bits(0.0f);
                if (destinationView.isArray()) {
                    short[] dst = destinationView.requireBF16Array();
                    for (int logical = 0; logical < total; logical++) {
                        dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = zero;
                    }
                } else {
                    for (int logical = 0; logical < total; logical++) {
                        writeBF16Bits(destinationView, offsetForLogical(logical, shape, dense, strides, baseOffset), zero);
                    }
                }
            }
            case BOOL, INT32, INT64 -> throw new IllegalArgumentException("zeroStorage requires floating output dtype.");
        }
    }

    static double reduce(double current, double update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> current + update;
            case MUL -> current * update;
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static int reduceInt(int current, int update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> Math.addExact(current, update);
            case MUL -> Math.multiplyExact(current, update);
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static long reduceLong(long current, long update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> Math.addExact(current, update);
            case MUL -> Math.multiplyExact(current, update);
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static DuplicateState duplicateState(Tensor out, ScatterReduction reduction, String operationName) {
        if (reduction == ScatterReduction.NONE) {
            return new DuplicateState(new boolean[out.getFlatDataSize()], operationName);
        }
        return DuplicateState.NOOP;
    }

    static final class DuplicateState {
        static final DuplicateState NOOP = new DuplicateState(null, "scatter");

        private final boolean[] seen;
        private final String operationName;

        private DuplicateState(boolean[] seen, String operationName) {
            this.seen = seen;
            this.operationName = operationName;
        }

        void mark(int targetLogical) {
            if (seen == null) {
                return;
            }
            if (seen[targetLogical]) {
                throw new IllegalArgumentException(operationName + " NONE reduction does not allow duplicate target indices.");
            }
            seen[targetLogical] = true;
        }
    }

    static final class IndexStoragePlan {
        private final DataType dataType;
        private final int[] shape;
        private final int[] dense;
        private final int[] strides;
        private final int baseOffset;

        private IndexStoragePlan(DataType dataType, int[] shape, int[] dense, int[] strides, int baseOffset) {
            this.dataType = dataType;
            this.shape = shape;
            this.dense = dense;
            this.strides = strides;
            this.baseOffset = baseOffset;
        }

        private int offset(int logicalIndex) {
            return offsetForLogical(logicalIndex, shape, dense, strides, baseOffset);
        }
    }

    private static void requireViewMatchesTensor(
            Tensor tensor,
            CpuStorageView view,
            DataType expectedDType,
            String label
    ) {
        if (view.dtype() != expectedDType || view.dtype() != tensor.getDataType()) {
            throw new IllegalStateException(label + " storage dtype mismatch. tensor="
                    + tensor.getDataType() + ", view=" + view.dtype() + ", expected=" + expectedDType);
        }
        if (view.logicalSize() != tensor.getFlatDataSize()) {
            throw new IllegalStateException(label + " storage logical size mismatch. tensor="
                    + tensor.getFlatDataSize() + ", view=" + view.logicalSize());
        }
    }

    private static double readFloatingIndex(CpuStorageView indices, IndexStoragePlan plan, int logicalIndex) {
        int offset = plan.offset(logicalIndex);
        return switch (plan.dataType) {
            case FLOAT64 -> readF64(indices, offset);
            case FLOAT32 -> readF32(indices, offset);
            case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(readBF16Bits(indices, offset));
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Gather indices must be numeric integral values.");
        };
    }

    private static void copyF64(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            double[] src = sourceView.requireF64Array();
            double[] dst = destinationView.requireF64Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeF64(destinationView, destinationOffset, readF64(sourceView, sourceOffset));
        }
    }

    private static void copyF32(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            float[] src = sourceView.requireF32Array();
            float[] dst = destinationView.requireF32Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeF32(destinationView, destinationOffset, readF32(sourceView, sourceOffset));
        }
    }

    private static void copyBF16(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            short[] src = sourceView.requireBF16Array();
            short[] dst = destinationView.requireBF16Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeBF16Bits(destinationView, destinationOffset, readBF16Bits(sourceView, sourceOffset));
        }
    }

    private static void copyBool(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            byte[] src = sourceView.requireBoolArray();
            byte[] dst = destinationView.requireBoolArray();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeBool(destinationView, destinationOffset, readBool(sourceView, sourceOffset));
        }
    }

    private static void copyI32(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            int[] src = sourceView.requireI32Array();
            int[] dst = destinationView.requireI32Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeI32(destinationView, destinationOffset, readI32(sourceView, sourceOffset));
        }
    }

    private static void copyI64(
            CpuStorageView sourceView,
            CpuStorageView destinationView,
            int[] shape,
            int[] dense,
            int[] sourceStrides,
            int[] destinationStrides,
            int sourceBaseOffset,
            int destinationBaseOffset,
            int total
    ) {
        if (allArrays(sourceView, destinationView)) {
            long[] src = sourceView.requireI64Array();
            long[] dst = destinationView.requireI64Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset)] =
                        src[offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset)];
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            int sourceOffset = offsetForLogical(logical, shape, dense, sourceStrides, sourceBaseOffset);
            int destinationOffset = offsetForLogical(logical, shape, dense, destinationStrides, destinationBaseOffset);
            writeI64(destinationView, destinationOffset, readI64(sourceView, sourceOffset));
        }
    }
}
