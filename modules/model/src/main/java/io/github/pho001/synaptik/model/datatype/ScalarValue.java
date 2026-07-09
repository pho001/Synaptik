package io.github.pho001.synaptik.model.datatype;

/**
 * Stores one immutable scalar semantic value as an exact {@link DataType} and primitive bits.
 *
 * <p>The named factories preserve the representation of each current data type. Floating-point
 * signed zeros and raw NaN payloads remain distinct, integral values never pass through a
 * floating representation, raw BFLOAT16 patterns remain unchanged, and booleans use canonical
 * zero-or-one bits. This value is an operation parameter, not a scalar Tensor, storage value,
 * conversion request, serialization, or executable constant. Inspection is deliberately
 * type-specific: an inspector for a different data type fails instead of converting the value.</p>
 *
 * <p>Equality and hashing use both the data type and exact stored bits. Consequently signed
 * floating zeros and distinct raw NaN payloads remain unequal, while BOOL is always represented
 * canonically by zero or one. The class has no mutable state, returns no {@code null} value, and
 * owns no Tensor, storage, graph, compiler, backend, or execution state.</p>
 */
public final class ScalarValue {
    private final DataType dataType;
    private final long bits;

    private ScalarValue(DataType dataType, long bits) {
        this.dataType = dataType;
        this.bits = bits;
    }

    /**
     * Creates an exact IEEE-754 binary64 value.
     *
     * @param value value whose raw bits are retained
     * @return non-null exact {@link DataType#FLOAT64} scalar value
     */
    public static ScalarValue float64(double value) {
        return new ScalarValue(DataType.FLOAT64, Double.doubleToRawLongBits(value));
    }

    /**
     * Creates an exact IEEE-754 binary32 value.
     *
     * @param value value whose raw bits are retained
     * @return non-null exact {@link DataType#FLOAT32} scalar value
     */
    public static ScalarValue float32(float value) {
        return new ScalarValue(DataType.FLOAT32, Float.floatToRawIntBits(value) & 0xFFFF_FFFFL);
    }

    /**
     * Converts one binary32 value to BFLOAT16 using round-to-nearest, ties-to-even.
     *
     * @param value binary32 value to convert through {@link BFloat16Bits#fromFloat(float)}
     * @return non-null converted {@link DataType#BFLOAT16} scalar value
     */
    public static ScalarValue bfloat16(float value) {
        return bfloat16Bits(BFloat16Bits.fromFloat(value));
    }

    /**
     * Creates an exact BFLOAT16 value without conversion or NaN canonicalization.
     *
     * @param bits raw BFLOAT16 bit pattern; every pattern is accepted
     * @return non-null exact {@link DataType#BFLOAT16} scalar value
     */
    public static ScalarValue bfloat16Bits(short bits) {
        return new ScalarValue(DataType.BFLOAT16, bits & 0xFFFFL);
    }

    /**
     * Creates an exact signed 32-bit integral value.
     *
     * @param value value retained without conversion
     * @return non-null exact {@link DataType#INT32} scalar value
     */
    public static ScalarValue int32(int value) {
        return new ScalarValue(DataType.INT32, value & 0xFFFF_FFFFL);
    }

    /**
     * Creates an exact signed 64-bit integral value.
     *
     * @param value value retained without conversion, including values beyond binary64's exact
     *     integer range
     * @return non-null exact {@link DataType#INT64} scalar value
     */
    public static ScalarValue int64(long value) {
        return new ScalarValue(DataType.INT64, value);
    }

    /**
     * Creates a canonical logical value.
     *
     * @param value logical value to retain
     * @return non-null {@link DataType#BOOL} scalar value with bits zero or one
     */
    public static ScalarValue bool(boolean value) {
        return new ScalarValue(DataType.BOOL, value ? 1L : 0L);
    }

    /**
     * Returns the exact logical data type of this scalar value.
     *
     * @return non-null exact data type
     */
    public DataType dataType() {
        return dataType;
    }

    /**
     * Returns the exact binary64 value.
     *
     * @return value reconstructed from its raw bits
     * @throws IllegalStateException if this value is not {@link DataType#FLOAT64}
     */
    public double float64Value() {
        requireType(DataType.FLOAT64);
        return Double.longBitsToDouble(bits);
    }

    /**
     * Returns the exact binary32 value.
     *
     * @return value reconstructed from its raw bits
     * @throws IllegalStateException if this value is not {@link DataType#FLOAT32}
     */
    public float float32Value() {
        requireType(DataType.FLOAT32);
        return Float.intBitsToFloat((int) bits);
    }

    /**
     * Returns the exact raw BFLOAT16 pattern.
     *
     * @return all 16 stored bits in a Java {@code short}
     * @throws IllegalStateException if this value is not {@link DataType#BFLOAT16}
     */
    public short bfloat16Bits() {
        requireType(DataType.BFLOAT16);
        return (short) bits;
    }

    /**
     * Returns the exact signed 32-bit integral value.
     *
     * @return stored signed value
     * @throws IllegalStateException if this value is not {@link DataType#INT32}
     */
    public int int32Value() {
        requireType(DataType.INT32);
        return (int) bits;
    }

    /**
     * Returns the exact signed 64-bit integral value.
     *
     * @return stored signed value
     * @throws IllegalStateException if this value is not {@link DataType#INT64}
     */
    public long int64Value() {
        requireType(DataType.INT64);
        return bits;
    }

    /**
     * Returns the canonical logical value.
     *
     * @return {@code true} exactly when the stored bits are one
     * @throws IllegalStateException if this value is not {@link DataType#BOOL}
     */
    public boolean booleanValue() {
        requireType(DataType.BOOL);
        return bits == 1L;
    }

    private void requireType(DataType expected) {
        if (dataType != expected) {
            throw new IllegalStateException(
                    "scalar value has data type " + dataType + ", not " + expected);
        }
    }

    /**
     * Compares the exact data type and stored bit pattern.
     *
     * @param other value to compare with this scalar value; may be {@code null}
     * @return {@code true} exactly when {@code other} is a {@code ScalarValue} with the same data
     *     type and exact stored bits
     */
    @Override
    public boolean equals(Object other) {
        return this == other
                || other instanceof ScalarValue that
                        && dataType == that.dataType
                        && bits == that.bits;
    }

    /**
     * Returns a hash derived from the exact data type and stored bit pattern.
     *
     * @return hash consistent with exact typed-bit equality
     */
    @Override
    public int hashCode() {
        return 31 * dataType.hashCode() + Long.hashCode(bits);
    }

    /**
     * Returns fixed-width hexadecimal diagnostic text for the exact typed bits.
     *
     * @return non-null diagnostic text; not a serialization format
     */
    @Override
    public String toString() {
        int width = switch (dataType) {
            case FLOAT64, INT64 -> 16;
            case FLOAT32, INT32 -> 8;
            case BFLOAT16 -> 4;
            case BOOL -> 2;
        };
        return "ScalarValue[dataType="
                + dataType
                + ", bits=0x"
                + String.format("%0" + width + "X", bits)
                + "]";
    }
}
