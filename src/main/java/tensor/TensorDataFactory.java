package tensor;

public final class TensorDataFactory {
    private TensorDataFactory() {
    }

    public static Tensor scalar(double value, DataType dataType) {
        if (dataType == DataType.INT32) {
            long integral = Math.round(value);
            if (Math.abs(value - integral) > 1e-9) {
                throw new IllegalArgumentException("INT32 scalar requires an integral value. got=" + value);
            }
            return new Tensor(new int[]{(int) integral}, new int[]{1}, null, "scalar_const", DataType.INT32);
        }
        return new Tensor(new double[]{value}, new int[]{1}, new int[]{1}, null, "scalar_const", dataType);
    }

    public static Tensor onesLike(Tensor other) {
        int size = other.getFlatDataSize();
        int[] shape = other.getShape().clone();
        if (other.getDataType() == DataType.INT32) {
            int[] data = new int[size];
            java.util.Arrays.fill(data, 1);
            return new Tensor(data, shape, null, "ones_like", DataType.INT32);
        }
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0d);
        return new Tensor(data, shape, null, "ones_like", other.getDataType());
    }

    public static Tensor zerosLike(Tensor other) {
        int size = other.getFlatDataSize();
        int[] shape = other.getShape().clone();
        if (other.getDataType() == DataType.INT32) {
            return new Tensor(new int[size], shape, null, "zeros_like", DataType.INT32);
        }
        return new Tensor(new double[size], shape, null, "zeros_like", other.getDataType());
    }

    public static Tensor flatTensor(String label, double[] data, boolean requiresGrad, DataType dataType) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        return shapedTensor(label, data, requiresGrad, dataType, data.length);
    }

    public static Tensor shapedTensor(String label, double[] data, boolean requiresGrad, DataType dataType, int... shape) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("shape cannot be null/empty");
        }
        Tensor tensor = new Tensor(shape, null, label, dataType);
        tensor.setData(data.clone());
        tensor.setRequiresGrad(requiresGrad);
        return tensor;
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
