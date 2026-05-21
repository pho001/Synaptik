package tensor;

import tensor.storage.TensorStorageAccess;

import java.util.Arrays;

public final class TensorInspection {
    private TensorInspection() {
    }

    public static String toStructString(Tensor tensor) {
        int[] shape = tensor.getShape();
        int[] strides = tensor.getStrides();
        double[] data = toDoubleArrayCopy(tensor);
        StringBuilder sb = new StringBuilder();
        buildTensorString(shape, strides, data, new int[shape.length], 0, sb);
        return "tensor.Tensor{" +
                "shape=" + Arrays.toString(shape) +
                ", strides=" + Arrays.toString(strides) +
                ", data=" + sb +
                '}';
    }

    public static double[] toDoubleArrayCopy(Tensor tensor) {
        int n = tensor.getFlatDataSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = tensor.getByFlatIndex(i);
        }
        return out;
    }

    public static boolean[] toBooleanArrayCopy(Tensor tensor) {
        if (tensor.getDataType() != DataType.BOOL) {
            throw new UnsupportedOperationException("toBooleanArrayCopy() is only supported for BOOL tensors.");
        }
        int n = tensor.getFlatDataSize();
        boolean[] out = new boolean[n];
        for (int i = 0; i < n; i++) {
            out[i] = tensor.getByFlatIndex(i) != 0.0d;
        }
        return out;
    }

    public static float[] toFloat32ArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.FLOAT32, "toFloat32ArrayCopy()");
        float[] storage = tensor.float32DataInternal();
        int n = tensor.getFlatDataSize();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static double[] toFloat64ArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.FLOAT64, "toFloat64ArrayCopy()");
        double[] storage = tensor.float64DataInternal();
        int n = tensor.getFlatDataSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static short[] toBFloat16BitsArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.BFLOAT16, "toBFloat16BitsArrayCopy()");
        short[] storage = tensor.bfloat16DataInternal();
        int n = tensor.getFlatDataSize();
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static int[] toInt32ArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.INT32, "toInt32ArrayCopy()");
        int[] storage = tensor.int32DataInternal();
        int n = tensor.getFlatDataSize();
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static long[] toInt64ArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.INT64, "toInt64ArrayCopy()");
        long[] storage = tensor.int64DataInternal();
        int n = tensor.getFlatDataSize();
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static byte[] toBoolByteArrayCopy(Tensor tensor) {
        requireDataType(tensor, DataType.BOOL, "toBoolByteArrayCopy()");
        byte[] storage = tensor.boolDataInternal();
        int n = tensor.getFlatDataSize();
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = storage[TensorStorageAccess.logicalFlatIndexToStorageOffset(tensor.metadataInternal(), i)];
        }
        return out;
    }

    public static double[] toDoubleStorageOrderCopy(Tensor tensor) {
        int n = tensor.getStorageSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = tensor.getByStorageOffset(i);
        }
        return out;
    }

    public static double scalarAsDouble(Tensor tensor) {
        if (tensor.getFlatDataSize() != 1) {
            throw new IllegalStateException("Tensor is not scalar.");
        }
        if (tensor.getDataType() == DataType.BOOL) {
            throw new UnsupportedOperationException("scalarAsDouble() is not supported for BOOL tensors.");
        }
        return tensor.getByFlatIndex(0);
    }

    private static void requireDataType(Tensor tensor, DataType expected, String operationName) {
        if (tensor.getDataType() != expected) {
            throw new UnsupportedOperationException(operationName + " is only supported for " + expected + " tensors.");
        }
    }

    private static void buildTensorString(int[] shape, int[] strides, double[] data, int[] indices, int dim, StringBuilder sb) {
        if (dim == shape.length) {
            int index = 0;
            for (int i = 0; i < indices.length; i++) {
                index += indices[i] * strides[i];
            }
            sb.append(data[index]);
            return;
        }
        sb.append("[");
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            buildTensorString(shape, strides, data, indices, dim + 1, sb);
            if (i < shape[dim] - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
    }
}
