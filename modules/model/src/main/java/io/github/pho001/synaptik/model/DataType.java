package io.github.pho001.synaptik.model;

/**
 * Identifies the logical element type of a Synaptik tensor.
 *
 * <p>Each constant exposes backend-independent semantic metadata: its value category, logical bit
 * width, storage byte width, and whether values of the type may participate in automatic
 * differentiation. The enum deliberately contains no backend support, device format, storage
 * carrier, or kernel-selection information.</p>
 */
public enum DataType {
    /** IEEE-754 binary64 floating-point values. */
    FLOAT64(DataTypeCategory.FLOATING, 64, true),

    /** IEEE-754 binary32 floating-point values and the default floating data type. */
    FLOAT32(DataTypeCategory.FLOATING, 32, true),

    /** Brain floating-point values stored as the most significant 16 bits of binary32. */
    BFLOAT16(DataTypeCategory.FLOATING, 16, true),

    /** 32-bit signed integral values. */
    INT32(DataTypeCategory.INTEGRAL, 32, false),

    /** 64-bit signed integral values. */
    INT64(DataTypeCategory.INTEGRAL, 64, false),

    /** Logical false-or-true values with one byte of logical storage width. */
    BOOL(DataTypeCategory.BOOLEAN, 8, false);

    private final DataTypeCategory category;
    private final int bitWidth;
    private final int byteWidth;
    private final boolean differentiable;

    /**
     * Creates one immutable data type constant.
     *
     * @param category non-null mathematical value category of the data type
     * @param bitWidth positive logical width in bits; must be divisible by eight
     * @param differentiable whether the data type may participate in automatic differentiation
     */
    DataType(DataTypeCategory category, int bitWidth, boolean differentiable) {
        this.category = category;
        this.bitWidth = bitWidth;
        this.byteWidth = bitWidth / Byte.SIZE;
        this.differentiable = differentiable;
    }

    /**
     * Returns the mathematical value category of this data type.
     *
     * @return non-null category; the result is stable for the lifetime of the enum constant
     */
    public DataTypeCategory category() {
        return category;
    }

    /**
     * Returns the logical width of one value in bits.
     *
     * <p>This metadata describes the portable model format. It does not guarantee that a concrete
     * backend uses the same physical allocation granularity.</p>
     *
     * @return positive logical width in bits
     */
    public int bitWidth() {
        return bitWidth;
    }

    /**
     * Returns the logical storage width of one value in bytes.
     *
     * <p>The value is derived from {@link #bitWidth()} and is exact because every initial Synaptik
     * data type is byte-aligned. Backend-specific alignment and padding are outside this contract.</p>
     *
     * @return positive logical storage width in bytes
     */
    public int byteWidth() {
        return byteWidth;
    }

    /**
     * Reports whether this data type represents floating-point values.
     *
     * @return {@code true} for {@link #FLOAT64}, {@link #FLOAT32}, and {@link #BFLOAT16}; otherwise
     *     {@code false}
     */
    public boolean isFloating() {
        return category == DataTypeCategory.FLOATING;
    }

    /**
     * Reports whether this data type represents signed integral values.
     *
     * @return {@code true} for {@link #INT32} and {@link #INT64}; otherwise {@code false}
     */
    public boolean isIntegral() {
        return category == DataTypeCategory.INTEGRAL;
    }

    /**
     * Reports whether this data type represents logical boolean values.
     *
     * @return {@code true} only for {@link #BOOL}
     */
    public boolean isBoolean() {
        return category == DataTypeCategory.BOOLEAN;
    }

    /**
     * Reports whether values of this data type may participate in automatic differentiation.
     *
     * <p>This is semantic model metadata. It does not indicate whether any particular operation or
     * backend supports differentiation.</p>
     *
     * @return {@code true} for floating data types; {@code false} for integral and boolean data
     *     types
     */
    public boolean isDifferentiable() {
        return differentiable;
    }

    /**
     * Returns the floating data type used when an API does not request another floating precision.
     *
     * @return {@link #FLOAT32}; never {@code null}
     */
    public static DataType defaultFloating() {
        return FLOAT32;
    }
}
