package io.github.pho001.synaptik.model.operation.index;

/**
 * Selects the replacement or mathematical reduction used to combine functional-scatter updates
 * at one target coordinate.
 *
 * <p>The base value is the value initially present in the scatter's {@code data} input. An update
 * is a value from the ordered {@code updates} input, and a target coordinate is the result
 * coordinate selected through {@code indices}. For one coordinate {@code c}, let {@code U(c)} be
 * the logical multiset of update scalar values addressed to {@code c}. Equal values from
 * different update positions remain different multiset members. A non-replacement reduction
 * combines exactly one occurrence of {@code data[c]} with exactly one occurrence of every member
 * of {@code U(c)}. If {@code U(c)} is empty, the result is the exact unchanged representation of
 * {@code data[c]}; an implementation must not evaluate an identity operation or otherwise
 * canonicalize that value.</p>
 *
 * <p>For {@link #MUL}, {@link #MIN}, and {@link #MAX}, the abstract result is independent of
 * update encounter order, physical layout, strides, atomic scheduling, tree shape, and backend
 * traversal. This fixes a represented-value target, not an execution algorithm. Data and updates
 * have the same exact type: there is no model-visible promotion, widening, conversion, or
 * saturation. These arithmetic reductions accept floating and signed-integral values and do not
 * accept BOOL.</p>
 *
 * <p>For floating {@code MUL} when {@code U(c)} is non-empty, the exact represented factors are
 * interpreted as one abstract product in the unchanged result format. Any NaN factor produces
 * NaN. A group
 * containing both any zero and any infinity produces NaN. Otherwise, any infinity produces
 * infinity and any zero produces zero; in either case the sign is the parity of all negative
 * factors, including negative zero and negative infinity. A finite non-zero exact product is
 * rounded to the unchanged result format with round-to-nearest, ties-to-even. Finite overflow
 * produces signed infinity; subnormal, underflow, and signed-zero results follow that rounding
 * and the exact product sign. FLOAT32 and FLOAT64 use their represented IEEE-754 values, while
 * BFLOAT16 uses the value represented by its exact current 16-bit storage and rounds the final
 * abstract product to BFLOAT16. The contract does not select a NaN payload, sign, signaling
 * behavior, or source. Reassociation and equal-or-wider intermediates are permitted only when
 * they conform to this target and any future conformance tolerance; narrower accumulation,
 * saturation, a fixed factor sequence, payload preservation, and bitwise reproducibility are not
 * promised.</p>
 *
 * <p>When {@code U(c)} is non-empty, floating {@code MIN} and {@code MAX} propagate NaN, with no
 * payload, sign, signaling, source, or bitwise promise. Otherwise they use ordinary numeric order,
 * including infinities. When both zero signs occur, {@code MIN} produces negative zero and
 * {@code MAX} produces positive zero, independently of encounter order. Equal non-zero values
 * identify only the represented numeric result, not a source occurrence. Integral {@code MUL} is
 * exact-width two's-complement modular multiplication, modulo {@code 2^32} for INT32 or
 * {@code 2^64} for INT64. Integral {@code MIN} and {@code MAX} use ordinary signed order.</p>
 *
 * <p>This vocabulary is explicit: {@code null} never means {@link #NONE}. Value-aware validation
 * occurs after model metadata construction. In particular, {@code NONE} requires target
 * coordinates to be unique within one operation; duplicate targets are invalid rather than
 * resolved according to an unspecified update order. The existing {@link #ADD} meaning is
 * unchanged: it combines the base and every addressed update, including duplicates, using
 * fixed-width modular integral addition or reassociable floating addition without a bitwise-order
 * guarantee. This enum defines no derivative or subgradient policy, operand validation, graph or
 * compiler behavior, backend capability, or execution route.</p>
 */
public enum ScatterReduction {
    /**
     * Replaces each addressed base value with its single update value.
     *
     * <p>Target coordinates within one operation must be unique. Multiple updates addressing the
     * same target are invalid and are not resolved by first-write, last-write, or any other update
     * order. Detection requires index values and therefore occurs after semantic metadata
     * construction.</p>
     */
    NONE,

    /** Combines the base value and all updates addressed to a target by addition. */
    ADD,

    /**
     * Produces the order-independent abstract product of the base and all addressed updates.
     *
     * <p>Floating special values, final result-format rounding, and integral modular arithmetic
     * follow the type-level contract above. An unaddressed target preserves the exact base
     * representation.</p>
     */
    MUL,

    /**
     * Produces the order-independent maximum of the base and all addressed updates.
     *
     * <p>Floating NaN propagates and opposite signed zeros select positive zero. Integral values
     * use signed order. An unaddressed target preserves the exact base representation.</p>
     */
    MAX,

    /**
     * Produces the order-independent minimum of the base and all addressed updates.
     *
     * <p>Floating NaN propagates and opposite signed zeros select negative zero. Integral values
     * use signed order. An unaddressed target preserves the exact base representation.</p>
     */
    MIN
}
