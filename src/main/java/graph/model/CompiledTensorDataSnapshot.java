package graph.model;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.Arrays;

/**
 * Immutable compile-time data snapshot for small static tensor payloads needed by planning/lowering.
 */
public record CompiledTensorDataSnapshot(
        DataType dataType,
        int[] int32Values
) {
    public static final CompiledTensorDataSnapshot EMPTY = new CompiledTensorDataSnapshot(null, null);

    public CompiledTensorDataSnapshot {
        int32Values = int32Values == null ? null : int32Values.clone();
    }

    public static CompiledTensorDataSnapshot captureStaticLeaf(Tensor tensor) {
        if (tensor == null || tensor.getOperation() != null || tensor.getDataType() != DataType.INT32) {
            return EMPTY;
        }
        int[] storage = TensorInternalAccess.int32Data(tensor);
        if (storage == null) {
            return EMPTY;
        }
        int[] shape = tensor.getShapeUnsafe();
        int[] strides = tensor.getStridesUnsafe();
        int size = tensor.getFlatDataSize();
        if (shape == null || strides == null || shape.length != strides.length || size < 0) {
            return EMPTY;
        }
        int[] values = new int[size];
        if (size == 0) {
            return new CompiledTensorDataSnapshot(DataType.INT32, values);
        }
        copyInt32LogicalValues(storage, values, shape, strides, tensor.getStorageOffsetUnsafe(), 0, 0);
        return new CompiledTensorDataSnapshot(DataType.INT32, values);
    }

    public boolean hasInt32Values() {
        return dataType == DataType.INT32 && int32Values != null;
    }

    @Override
    public int[] int32Values() {
        return int32Values == null ? null : int32Values.clone();
    }

    @Override
    public String toString() {
        return hasInt32Values()
                ? "CompiledTensorDataSnapshot[INT32," + int32Values.length + "]"
                : "CompiledTensorDataSnapshot[empty]";
    }

    private static int copyInt32LogicalValues(
            int[] storage,
            int[] out,
            int[] shape,
            int[] strides,
            int storageOffset,
            int dimension,
            int outOffset
    ) {
        if (dimension == shape.length) {
            if (storageOffset < 0 || storageOffset >= storage.length) {
                throw new IllegalArgumentException("Static INT32 tensor snapshot reads outside storage");
            }
            out[outOffset] = storage[storageOffset];
            return outOffset + 1;
        }
        int current = outOffset;
        for (int i = 0; i < shape[dimension]; i++) {
            current = copyInt32LogicalValues(
                    storage,
                    out,
                    shape,
                    strides,
                    storageOffset + i * strides[dimension],
                    dimension + 1,
                    current
            );
        }
        return current;
    }
}
