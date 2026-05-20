package tensor;

/**
 * Element storage type for a {@link Tensor}.
 *
 * <p>Floating types participate in numeric operations and gradient propagation.
 * {@link #INT32} and {@link #INT64} are intended for integral index tensors and
 * ONNX-compatible integer values, and {@link #BOOL} is intended for masks and
 * logical reductions. Implicit conversion between integral/boolean tensors and
 * floating tensors is intentionally not supported by the public tensor API.</p>
 */
public enum DataType {
    /** IEEE-754 double precision values backed by {@code double[]}. */
    FLOAT64,
    /** IEEE-754 single precision values backed by {@code float[]}. */
    FLOAT32,
    /** bfloat16 bit patterns backed by {@code short[]}. */
    BFLOAT16,
    /** 32-bit signed integer values, mainly for gather/scatter target indices. */
    INT32,
    /** 64-bit signed integer values, mainly for ONNX runtime index tensors. */
    INT64,
    /** Boolean mask values backed by normalized {@code byte} values 0 or 1. */
    BOOL;

    /**
     * Returns whether this dtype stores floating-point numeric values.
     *
     * @return true for FLOAT64, FLOAT32, and BFLOAT16
     */
    public boolean isFloating() {
        return switch (this) {
            case FLOAT64, FLOAT32, BFLOAT16 -> true;
            case INT32, INT64, BOOL -> false;
        };
    }

    /**
     * Returns whether this dtype stores signed integer values.
     *
     * @return true for INT32 and INT64
     */
    public boolean isIntegral() {
        return switch (this) {
            case INT32, INT64 -> true;
            case FLOAT64, FLOAT32, BFLOAT16, BOOL -> false;
        };
    }

    /**
     * Returns whether this dtype stores boolean mask values.
     *
     * @return true for BOOL
     */
    public boolean isBoolean() {
        return this == BOOL;
    }
}
