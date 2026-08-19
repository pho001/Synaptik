package io.github.pho001.synaptik.nn.initialization;

import java.util.Objects;

/**
 * Selects one closed eager parameter-initialization algorithm without retaining layer state.
 *
 * <p>The value contains only an algorithm choice and, for configured normal or uniform
 * initialization, its finite binary64 arguments. Shapes, data types, random generators, seeds,
 * parameter order, and bias policy remain the responsibility of the applying layer. Glorot and
 * Kaiming presets therefore do not carry a fan value: the selected initializer derives fan-in
 * and fan-out from each complete positive rank-two Shape supplied when the policy is applied.</p>
 *
 * <p>Configured values use structural equality over the selected algorithm and the exact
 * {@link Double#doubleToLongBits(double)} representation of each argument. Equality distinguishes
 * positive and negative zero. The diagnostic string is deterministic but is not a parser input,
 * checkpoint encoding, or compatibility format. This type has no public constructor or
 * extension point and retains no callback or mutable state.</p>
 */
public final class ParameterInitialization {
    enum Kind {
        GLOROT_NORMAL,
        GLOROT_UNIFORM,
        KAIMING_RELU_NORMAL,
        KAIMING_RELU_UNIFORM,
        NORMAL,
        UNIFORM,
        ZEROS,
        ONES
    }

    private static final ParameterInitialization GLOROT_NORMAL = preset(Kind.GLOROT_NORMAL);
    private static final ParameterInitialization GLOROT_UNIFORM = preset(Kind.GLOROT_UNIFORM);
    private static final ParameterInitialization KAIMING_RELU_NORMAL =
            preset(Kind.KAIMING_RELU_NORMAL);
    private static final ParameterInitialization KAIMING_RELU_UNIFORM =
            preset(Kind.KAIMING_RELU_UNIFORM);
    private static final ParameterInitialization ZEROS = preset(Kind.ZEROS);
    private static final ParameterInitialization ONES = preset(Kind.ONES);

    private final Kind kind;
    private final double first;
    private final double second;

    private ParameterInitialization(Kind kind, double first, double second) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.first = first;
        this.second = second;
    }

    /**
     * Selects Glorot normal initialization.
     *
     * @return the immutable Glorot normal preset; never {@code null}
     */
    public static ParameterInitialization glorotNormal() {
        return GLOROT_NORMAL;
    }

    /**
     * Selects Glorot uniform initialization.
     *
     * @return the immutable Glorot uniform preset; never {@code null}
     */
    public static ParameterInitialization glorotUniform() {
        return GLOROT_UNIFORM;
    }

    /**
     * Selects fan-in Kaiming normal initialization with fixed ReLU gain.
     *
     * @return the immutable preset; never {@code null}
     */
    public static ParameterInitialization kaimingReluNormal() {
        return KAIMING_RELU_NORMAL;
    }

    /**
     * Selects fan-in Kaiming uniform initialization with fixed ReLU gain.
     *
     * @return the immutable preset; never {@code null}
     */
    public static ParameterInitialization kaimingReluUniform() {
        return KAIMING_RELU_UNIFORM;
    }

    /**
     * Creates a configured normal policy.
     *
     * @param mean finite binary64 mean
     * @param standardDeviation finite non-negative binary64 standard deviation; either signed
     *     zero is accepted
     * @return a new immutable value selecting the configured normal algorithm
     * @throws IllegalArgumentException if {@code mean} is non-finite, or if
     *     {@code standardDeviation} is non-finite or negative, checked in that order
     */
    public static ParameterInitialization normal(double mean, double standardDeviation) {
        requireFinite(mean, "mean");
        requireFinite(standardDeviation, "standardDeviation");
        if (standardDeviation < 0.0d) {
            throw new IllegalArgumentException(
                    "standardDeviation must be non-negative: " + standardDeviation);
        }
        return new ParameterInitialization(Kind.NORMAL, mean, standardDeviation);
    }

    /**
     * Creates a configured continuous-uniform policy.
     *
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound, strictly greater than
     *     {@code lowerBoundInclusive}
     * @return a new immutable value selecting the configured uniform algorithm
     * @throws IllegalArgumentException if the lower bound is non-finite, the upper bound is
     *     non-finite, or the interval is not increasing, checked in that order
     */
    public static ParameterInitialization uniform(
            double lowerBoundInclusive, double upperBoundExclusive) {
        requireFinite(lowerBoundInclusive, "lowerBoundInclusive");
        requireFinite(upperBoundExclusive, "upperBoundExclusive");
        if (!(lowerBoundInclusive < upperBoundExclusive)) {
            throw new IllegalArgumentException(
                    "uniform bounds must satisfy lower < upper: lower="
                            + lowerBoundInclusive + ", upper=" + upperBoundExclusive);
        }
        return new ParameterInitialization(
                Kind.UNIFORM, lowerBoundInclusive, upperBoundExclusive);
    }

    /**
     * Selects exact typed-zero initialization.
     *
     * @return the immutable exact-zero preset; never {@code null}
     */
    public static ParameterInitialization zeros() {
        return ZEROS;
    }

    /**
     * Selects exact typed-one initialization.
     *
     * @return the immutable exact-one preset; never {@code null}
     */
    public static ParameterInitialization ones() {
        return ONES;
    }

    /**
     * Reports whether applying this policy requires a transient random generator.
     *
     * @return {@code false} only for {@link #zeros()} and {@link #ones()}; {@code true} for the
     *     six sampling policies
     */
    public boolean requiresRandomGenerator() {
        return kind != Kind.ZEROS && kind != Kind.ONES;
    }

    Kind kind() {
        return kind;
    }

    double first() {
        return first;
    }

    double second() {
        return second;
    }

    /**
     * Compares the selected algorithm and exact configured binary64 argument bits.
     *
     * @param object the value to compare; may be {@code null}
     * @return {@code true} exactly when {@code object} is a parameter-initialization value with
     *     the same algorithm and argument bits
     */
    @Override
    public boolean equals(Object object) {
        return this == object || object instanceof ParameterInitialization other
                && kind == other.kind
                && Double.doubleToLongBits(first) == Double.doubleToLongBits(other.first)
                && Double.doubleToLongBits(second) == Double.doubleToLongBits(other.second);
    }

    /**
     * Returns a hash derived from the same algorithm and argument bits used by
     * {@link #equals(Object)}.
     *
     * @return the structural value hash
     */
    @Override
    public int hashCode() {
        return Objects.hash(
                kind, Double.doubleToLongBits(first), Double.doubleToLongBits(second));
    }

    /**
     * Returns a deterministic descriptive rendering of this policy.
     *
     * @return a non-null diagnostic string; never a persistence or parsing format
     */
    @Override
    public String toString() {
        return switch (kind) {
            case NORMAL -> "ParameterInitialization.normal(mean=" + first
                    + ", standardDeviation=" + second + ")";
            case UNIFORM -> "ParameterInitialization.uniform(lowerBoundInclusive=" + first
                    + ", upperBoundExclusive=" + second + ")";
            case GLOROT_NORMAL -> "ParameterInitialization.glorotNormal()";
            case GLOROT_UNIFORM -> "ParameterInitialization.glorotUniform()";
            case KAIMING_RELU_NORMAL -> "ParameterInitialization.kaimingReluNormal()";
            case KAIMING_RELU_UNIFORM -> "ParameterInitialization.kaimingReluUniform()";
            case ZEROS -> "ParameterInitialization.zeros()";
            case ONES -> "ParameterInitialization.ones()";
        };
    }

    private static ParameterInitialization preset(Kind kind) {
        return new ParameterInitialization(kind, 0.0d, 0.0d);
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite: " + value);
        }
    }
}
