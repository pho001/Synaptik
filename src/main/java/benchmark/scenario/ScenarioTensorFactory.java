package benchmark.scenario;

import tensor.DataType;
import tensor.Tensor;

public final class ScenarioTensorFactory {
    private ScenarioTensorFactory() {}

    public static Tensor flatTensor(String label, double[] data, boolean requiresGrad, DataType dataType) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        return shapedTensor(label, data, requiresGrad, dataType, new int[]{data.length});
    }

    public static Tensor shapedTensor(String label, double[] data, boolean requiresGrad, DataType dataType, int[] shape) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("shape cannot be null/empty");
        }
        Tensor t = new Tensor(shape, null, label, dataType);
        t.setData(data.clone());
        t.setRequiresGrad(requiresGrad);
        return t;
    }

    public static Tensor prefixTensorStrict(
            String label,
            double[] data,
            boolean requiresGrad,
            DataType dataType,
            int... shape
    ) {
        int size = flatSize(label, shape);
        if (data == null || data.length < size) {
            throw new IllegalArgumentException(
                    "Input data too small for " + label + ": required=" + size + ", available=" + (data == null ? 0 : data.length)
            );
        }
        double[] sliced = new double[size];
        System.arraycopy(data, 0, sliced, 0, size);
        return shapedTensor(label, sliced, requiresGrad, dataType, shape);
    }

    public static Tensor prefixTensorWrap(
            String label,
            double[] data,
            boolean requiresGrad,
            DataType dataType,
            int... shape
    ) {
        int size = flatSize(label, shape);
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null/empty for " + label);
        }
        double[] sliced = new double[size];
        if (data.length >= size) {
            System.arraycopy(data, 0, sliced, 0, size);
        } else {
            for (int i = 0; i < size; i++) {
                sliced[i] = data[i % data.length];
            }
        }
        return shapedTensor(label, sliced, requiresGrad, dataType, shape);
    }

    private static int flatSize(String label, int... shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("shape cannot be null/empty for " + label);
        }
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Invalid shape size for " + label);
        }
        return size;
    }
}
