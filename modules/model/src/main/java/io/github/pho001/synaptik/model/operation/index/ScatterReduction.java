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
 * <p>Target membership is independent of update encounter order, physical layout, strides,
 * atomic scheduling, and backend traversal. Floating reduction arithmetic may depend on the
 * rounded evaluation tree; extrema retain their exact winner rules. Data and updates have the
 * same declared type, with no model-visible promotion or saturation. These reductions accept
 * floating and signed-integral values, not BOOL.</p>
 *
 * <p>For floating {@code MUL} with nonempty {@code U(c)}, the base and every addressed update
 * participate exactly once in a rounded multiplication tree. A NaN factor or a zero/infinity
 * pair requires NaN class. Intermediate overflow can also produce NaN: with factors
 * {@code maxFinite}, {@code 2}, and {@code +0}, an overflow-first tree can produce NaN while a
 * different grouping can produce positive zero. NaN payload, sign, and signaling behavior are
 * unspecified. FLOAT32 multiplication rounds at named FLOAT32 sites; FLOAT64 sites do not permit
 * denormals-are-zero (DAZ) or flush-to-zero (FTZ). Homogeneous BFLOAT16 and FLOAT16 factors use
 * FLOAT32 working and accumulator values, then one final round-to-nearest, ties-to-even narrowing
 * to the declared low output. Named FLOAT32, BFLOAT16, and FLOAT16 multiplication inputs may use
 * same-signed-zero DAZ and finite subnormal primitive results may use FTZ under the family's
 * zero-sign rule, including low multiplication. Unaddressed cells retain their original bits.
 * No factor may be dropped, duplicated, or pre-narrowed; no once-rounded exact-product or bitwise
 * reproducibility promise applies.</p>
 *
 * <p>When {@code U(c)} is non-empty, floating {@code MIN} and {@code MAX} compare stored
 * represented values, including subnormals, without DAZ or FTZ. They propagate NaN, with no
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
     * Combines the base and all addressed updates once in a rounded multiplication tree.
     *
     * <p>Floating special values and intermediate overflow follow the type-level contract above;
     * integral multiplication is modular. An unaddressed target preserves the exact base
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
