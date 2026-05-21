package tensor.ops.loss;

import tensor.DataType;
import tensor.Tensor;

final class IndexTargetMasking {
    private IndexTargetMasking() {
    }

    static Tensor buildIgnoreMask(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        byte[] mask = new byte[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            mask[i] = value == ignoreIndex ? (byte) 0 : (byte) 1;
        }
        return new Tensor(mask, targetIndices.getShape().clone(), null, "index_valid_mask", DataType.BOOL);
    }

    static Tensor buildSafeIndices(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        if (targetIndices.getDataType() == DataType.INT32) {
            int[] safe = new int[size];
            for (int i = 0; i < size; i++) {
                long value = readIntegralIndex(targetIndices, i);
                safe[i] = value == ignoreIndex ? 0 : (int) value;
            }
            return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", DataType.INT32);
        }
        if (targetIndices.getDataType() == DataType.INT64) {
            long[] safe = new long[size];
            for (int i = 0; i < size; i++) {
                long value = readIntegralIndex(targetIndices, i);
                safe[i] = value == ignoreIndex ? 0L : value;
            }
            return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", DataType.INT64);
        }
        double[] safe = new double[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            safe[i] = value == ignoreIndex ? 0.0 : (double) value;
        }
        return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", targetIndices.getDataType());
    }

    private static long readIntegralIndex(Tensor indices, int flatIndex) {
        if (indices.getDataType() == DataType.INT32 || indices.getDataType() == DataType.INT64) {
            return indices.getIntegralByFlatIndex(flatIndex);
        }
        double raw = indices.getByFlatIndex(flatIndex);
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value.");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return integral;
    }
}
