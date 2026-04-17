package tensor;

import java.util.Arrays;

final class TensorDebugSupport {
    private TensorDebugSupport() {
    }

    static String toStructString(Tensor tensor) {
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

    static double[] toDoubleArrayCopy(Tensor tensor) {
        int n = tensor.getFlatDataSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = tensor.getByFlatIndex(i);
        }
        return out;
    }

    static boolean[] toBooleanArrayCopy(Tensor tensor) {
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

    static double[] toDoubleStorageOrderCopy(Tensor tensor) {
        int n = tensor.getStorageSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = tensor.getByStorageOffset(i);
        }
        return out;
    }

    static double scalarAsDouble(Tensor tensor) {
        if (tensor.getFlatDataSize() != 1) {
            throw new IllegalStateException("Tensor is not scalar.");
        }
        if (tensor.getDataType() == DataType.BOOL) {
            throw new UnsupportedOperationException("scalarAsDouble() is not supported for BOOL tensors.");
        }
        return tensor.getByFlatIndex(0);
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
