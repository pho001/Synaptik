package tensor;

/**
 * Convenience constructors for common tensor data patterns.
 *
 * <p>Factory methods allocate new tensors and do not retain mutable input arrays
 * unless documented otherwise. Returned tensors are not synchronized.</p>
 */
public final class TensorDataFactory {
    private TensorDataFactory() {
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * <p>All public dtypes are supported. Numeric dtypes receive exact zero;
     * {@link DataType#BOOL} receives {@code false}. The supplied shape is
     * defensively copied before the tensor is constructed.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType dtype of the returned tensor; must be non-null
     * @param label tensor label, may be null
     * @return newly allocated tensor
     */
    public static Tensor zeros(int[] shape, DataType dataType, String label) {
        int size = flatSize(label, shape);
        int[] copiedShape = shape.clone();
        return switch (requireDataType(dataType)) {
            case FLOAT64 -> new Tensor(new double[size], copiedShape, null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(new float[size], copiedShape, null, label, DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(new short[size], copiedShape, null, label, DataType.BFLOAT16);
            case INT32 -> new Tensor(new int[size], copiedShape, null, label, DataType.INT32);
            case INT64 -> new Tensor(new long[size], copiedShape, null, label, DataType.INT64);
            case BOOL -> new Tensor(new byte[size], copiedShape, null, label, DataType.BOOL);
        };
    }

    /**
     * Creates a tensor filled with ones.
     *
     * <p>All public dtypes are supported. Numeric dtypes receive exact one;
     * {@link DataType#BOOL} receives {@code true}. The supplied shape is
     * defensively copied before the tensor is constructed.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType dtype of the returned tensor; must be non-null
     * @param label tensor label, may be null
     * @return newly allocated tensor
     */
    public static Tensor ones(int[] shape, DataType dataType, String label) {
        int size = flatSize(label, shape);
        int[] copiedShape = shape.clone();
        return switch (requireDataType(dataType)) {
            case FLOAT64 -> {
                double[] data = new double[size];
                java.util.Arrays.fill(data, 1.0d);
                yield new Tensor(data, copiedShape, null, label, DataType.FLOAT64);
            }
            case FLOAT32 -> {
                float[] data = new float[size];
                java.util.Arrays.fill(data, 1.0f);
                yield new Tensor(data, copiedShape, null, label, DataType.FLOAT32);
            }
            case BFLOAT16 -> {
                double[] data = new double[size];
                java.util.Arrays.fill(data, 1.0d);
                yield new Tensor(data, copiedShape, null, label, DataType.BFLOAT16);
            }
            case INT32 -> {
                int[] data = new int[size];
                java.util.Arrays.fill(data, 1);
                yield new Tensor(data, copiedShape, null, label, DataType.INT32);
            }
            case INT64 -> {
                long[] data = new long[size];
                java.util.Arrays.fill(data, 1L);
                yield new Tensor(data, copiedShape, null, label, DataType.INT64);
            }
            case BOOL -> {
                byte[] data = new byte[size];
                java.util.Arrays.fill(data, (byte) 1);
                yield new Tensor(data, copiedShape, null, label, DataType.BOOL);
            }
        };
    }

    /**
     * Creates a floating tensor with normally distributed values.
     *
     * <p>The generator samples {@code mean + Random.nextGaussian() * stdDev}.
     * Integer and boolean dtypes are rejected because they would require an
     * additional rounding/truthiness policy that this factory does not own.</p>
     *
     * @param shape output shape
     * @param mean normal distribution mean
     * @param stdDev normal distribution standard deviation; must be non-negative
     * @param dataType floating dtype of the returned tensor
     * @param label tensor label, may be null
     * @return newly allocated tensor
     * @throws IllegalArgumentException if {@code stdDev} is negative/non-finite or {@code dataType} is non-floating
     */
    public static Tensor randn(int[] shape, double mean, double stdDev, DataType dataType, String label) {
        if (stdDev < 0.0d || !Double.isFinite(stdDev)) {
            throw new IllegalArgumentException("randn stdDev must be finite and non-negative.");
        }
        DataType type = requireDataType(dataType);
        if (type == DataType.BOOL || type == DataType.INT32 || type == DataType.INT64) {
            throw new IllegalArgumentException("randn requires a floating dtype.");
        }
        int size = flatSize(label, shape);
        double[] data = new double[size];
        java.util.Random random = new java.util.Random();
        for (int i = 0; i < size; i++) {
            data[i] = mean + random.nextGaussian() * stdDev;
        }
        return new Tensor(data, shape.clone(), null, label, type);
    }

    /**
     * Creates a rank-1 tensor containing an arithmetic integer range.
     *
     * <p>The output values are {@code start, start + step, ...} up to but not
     * including {@code end}. Positive and negative steps are supported when they
     * can reach {@code end}; empty ranges are rejected.</p>
     *
     * @param start inclusive start
     * @param end exclusive end
     * @param step non-zero step
     * @param dataType output dtype; supports numeric non-BOOL dtypes
     * @return rank-1 range tensor
     * @throws IllegalArgumentException if step is zero, direction cannot reach end,
     *                                  the range is empty, or dtype is BOOL
     */
    public static Tensor arange(int start, int end, int step, DataType dataType) {
        if (step == 0) {
            throw new IllegalArgumentException("arange step cannot be zero.");
        }
        long distance = (long) end - start;
        if ((step > 0 && distance < 0) || (step < 0 && distance > 0)) {
            throw new IllegalArgumentException("arange step direction cannot reach end.");
        }
        int size = (int) Math.max(0L, (Math.abs(distance) + Math.abs((long) step) - 1L) / Math.abs((long) step));
        if (size == 0) {
            throw new IllegalArgumentException("arange cannot produce an empty tensor.");
        }
        DataType type = requireDataType(dataType);
        if (type == DataType.BOOL) {
            throw new IllegalArgumentException("arange does not support BOOL dtype.");
        }
        int[] shape = new int[]{size};
        if (type == DataType.INT32) {
            int[] data = new int[shape[0]];
            for (int i = 0, value = start; i < size; i++, value += step) {
                data[i] = value;
            }
            return new Tensor(data, shape, null, "arange", DataType.INT32);
        }
        if (type == DataType.INT64) {
            long[] data = new long[shape[0]];
            for (int i = 0, value = start; i < size; i++, value += step) {
                data[i] = value;
            }
            return new Tensor(data, shape, null, "arange", DataType.INT64);
        }
        double[] data = new double[shape[0]];
        for (int i = 0, value = start; i < size; i++, value += step) {
            data[i] = value;
        }
        return new Tensor(data, shape, null, "arange", type);
    }

    /**
     * Creates a rank-1 scalar tensor with one element.
     *
     * @param value scalar value; must be integral when {@code dataType} is {@link DataType#INT32} or {@link DataType#INT64}
     * @param dataType dtype of the returned tensor; must be non-null
     * @return tensor with shape {@code [1]} and label {@code scalar_const}
     * @throws IllegalArgumentException if an INT32/INT64 scalar is requested with a non-integral value
     */
    public static Tensor scalar(double value, DataType dataType) {
        if (dataType == DataType.INT32) {
            long integral = Math.round(value);
            if (Math.abs(value - integral) > 1e-9) {
                throw new IllegalArgumentException("INT32 scalar requires an integral value. got=" + value);
            }
            return new Tensor(new int[]{(int) integral}, new int[]{1}, null, "scalar_const", DataType.INT32);
        }
        if (dataType == DataType.INT64) {
            long integral = Math.round(value);
            if (Math.abs(value - integral) > 1e-9) {
                throw new IllegalArgumentException("INT64 scalar requires an integral value. got=" + value);
            }
            return new Tensor(new long[]{integral}, new int[]{1}, null, "scalar_const", DataType.INT64);
        }
        return new Tensor(new double[]{value}, new int[]{1}, new int[]{1}, null, "scalar_const", dataType);
    }

    /**
     * Creates a tensor filled with ones matching another tensor's shape and dtype.
     *
     * <p>The returned tensor is independent leaf storage. For BOOL tensors, one
     * means {@code true}; for INT64 tensors, one is stored as {@code 1L}.</p>
     *
     * @param other tensor whose shape and dtype are copied; must be non-null
     * @return new tensor labeled {@code ones_like}
     */
    public static Tensor onesLike(Tensor other) {
        int size = other.getFlatDataSize();
        int[] shape = other.getShape().clone();
        if (other.getDataType() == DataType.BOOL) {
            byte[] data = new byte[size];
            java.util.Arrays.fill(data, (byte) 1);
            return new Tensor(data, shape, null, "ones_like", DataType.BOOL);
        }
        if (other.getDataType() == DataType.INT32) {
            int[] data = new int[size];
            java.util.Arrays.fill(data, 1);
            return new Tensor(data, shape, null, "ones_like", DataType.INT32);
        }
        if (other.getDataType() == DataType.INT64) {
            long[] data = new long[size];
            java.util.Arrays.fill(data, 1L);
            return new Tensor(data, shape, null, "ones_like", DataType.INT64);
        }
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0d);
        return new Tensor(data, shape, null, "ones_like", other.getDataType());
    }

    /**
     * Creates a tensor filled with zeros matching another tensor's shape and dtype.
     *
     * <p>The returned tensor is independent leaf storage. For BOOL tensors, zero
     * means {@code false}; for INT64 tensors, zero is stored as {@code 0L}.</p>
     *
     * @param other tensor whose shape and dtype are copied; must be non-null
     * @return new tensor labeled {@code zeros_like}
     */
    public static Tensor zerosLike(Tensor other) {
        int size = other.getFlatDataSize();
        int[] shape = other.getShape().clone();
        if (other.getDataType() == DataType.BOOL) {
            return new Tensor(new byte[size], shape, null, "zeros_like", DataType.BOOL);
        }
        if (other.getDataType() == DataType.INT32) {
            return new Tensor(new int[size], shape, null, "zeros_like", DataType.INT32);
        }
        if (other.getDataType() == DataType.INT64) {
            return new Tensor(new long[size], shape, null, "zeros_like", DataType.INT64);
        }
        return new Tensor(new double[size], shape, null, "zeros_like", other.getDataType());
    }

    /**
     * Creates a rank-1 tensor from a copied double array.
     *
     * @param label tensor label, may be null
     * @param data source values; must be non-null
     * @param requiresGrad whether the result should participate in gradient accumulation
     * @param dataType dtype used to store the values; must be a numeric floating dtype
     * @return tensor with shape {@code [data.length]}
     * @throws IllegalArgumentException if {@code data} is null
     */
    public static Tensor flatTensor(String label, double[] data, boolean requiresGrad, DataType dataType) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        return shapedTensor(label, data, requiresGrad, dataType, data.length);
    }

    /**
     * Creates a shaped tensor from a copied double array.
     *
     * @param label tensor label, may be null
     * @param data source values; must be non-null and match the product of {@code shape}
     * @param requiresGrad whether the result should participate in gradient accumulation
     * @param dataType dtype used to store the values
     * @param shape non-empty tensor shape
     * @return tensor with the requested shape
     * @throws IllegalArgumentException if {@code data} is null, {@code shape} is null/empty,
     *                                  or the data length does not match the shape size
     */
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

    /**
     * Creates a shaped tensor from the prefix of a source array.
     *
     * @param label tensor label used in diagnostics, may be null
     * @param data source values; must contain at least the requested flat size
     * @param requiresGrad whether the result should participate in gradient accumulation
     * @param dataType dtype used to store the values
     * @param shape non-empty tensor shape
     * @return tensor filled from {@code data[0..flatSize)}
     * @throws IllegalArgumentException if data is null/too small or shape is invalid
     */
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

    /**
     * Creates a shaped tensor by taking or repeating source values as needed.
     *
     * @param label tensor label used in diagnostics, may be null
     * @param data source values; must be non-null and non-empty
     * @param requiresGrad whether the result should participate in gradient accumulation
     * @param dataType dtype used to store the values
     * @param shape non-empty tensor shape
     * @return tensor filled from the source prefix, wrapping cyclically when the source is shorter
     * @throws IllegalArgumentException if data is null/empty or shape is invalid
     */
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
        try {
            return TensorShape.checkedFlatSize(shape);
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("Invalid shape size for " + label + ": " + ex.getMessage(), ex);
        }
    }

    private static DataType requireDataType(DataType dataType) {
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        return dataType;
    }
}
