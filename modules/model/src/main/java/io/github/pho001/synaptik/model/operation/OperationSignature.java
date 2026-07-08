package io.github.pho001.synaptik.model.operation;

import java.util.Objects;

/**
 * Defines one structurally valid attribute and occurrence-count variant for an operation kind.
 *
 * <p>A signature identifies the exact immutable {@link OperationAttrs} implementation accepted
 * by one variant and the inclusive numbers of logical input and output positions that one
 * occurrence may expose. Counts describe tensor or graph-value positions, not the rank, shape, or
 * element count of any individual value.</p>
 *
 * <p>The upper bound {@link Integer#MAX_VALUE} accepts every count representable by a Java
 * collection's {@code size()} result and therefore expresses an effectively variadic position
 * without a negative sentinel. A signature performs structural validation only. It does not
 * inspect operands, infer descriptors, define numerical behavior, report backend support, or
 * select execution behavior.</p>
 *
 * @param attributesType the exact non-null concrete attributes class accepted by this variant
 * @param minimumInputs the inclusive minimum logical input count; must be non-negative
 * @param maximumInputs the inclusive maximum logical input count; must be at least
 *     {@code minimumInputs}
 * @param minimumOutputs the inclusive minimum logical output count; must be positive
 * @param maximumOutputs the inclusive maximum logical output count; must be at least
 *     {@code minimumOutputs}
 */
public record OperationSignature(
        Class<? extends OperationAttrs> attributesType,
        int minimumInputs,
        int maximumInputs,
        int minimumOutputs,
        int maximumOutputs) {
    /**
     * Creates a validated immutable occurrence signature.
     *
     * @param attributesType the exact non-null concrete attributes class accepted by this variant
     * @param minimumInputs the inclusive minimum logical input count; must be non-negative
     * @param maximumInputs the inclusive maximum logical input count; must be at least
     *     {@code minimumInputs}
     * @param minimumOutputs the inclusive minimum logical output count; must be positive
     * @param maximumOutputs the inclusive maximum logical output count; must be at least
     *     {@code minimumOutputs}
     * @throws NullPointerException if {@code attributesType} is {@code null}
     * @throws IllegalArgumentException if an input bound is negative, the minimum input bound
     *     exceeds the maximum, the minimum output bound is not positive, or the minimum output
     *     bound exceeds the maximum
     */
    public OperationSignature {
        attributesType = Objects.requireNonNull(attributesType, "attributesType");
        if (minimumInputs < 0) {
            throw new IllegalArgumentException(
                    "minimumInputs must be non-negative: " + minimumInputs);
        }
        if (maximumInputs < minimumInputs) {
            throw new IllegalArgumentException(
                    "maximumInputs must be greater than or equal to minimumInputs: "
                            + maximumInputs + " < " + minimumInputs);
        }
        if (minimumOutputs < 1) {
            throw new IllegalArgumentException(
                    "minimumOutputs must be positive: " + minimumOutputs);
        }
        if (maximumOutputs < minimumOutputs) {
            throw new IllegalArgumentException(
                    "maximumOutputs must be greater than or equal to minimumOutputs: "
                            + maximumOutputs + " < " + minimumOutputs);
        }
    }

    /**
     * Creates a signature with exact input and output counts.
     *
     * @param attributesType the exact non-null concrete attributes class accepted by the variant
     * @param inputCount the exact non-negative logical input count
     * @param outputCount the exact positive logical output count
     * @return a new immutable signature whose minimum and maximum counts are equal
     * @throws NullPointerException if {@code attributesType} is {@code null}
     * @throws IllegalArgumentException if {@code inputCount} is negative or
     *     {@code outputCount} is not positive
     */
    public static OperationSignature fixed(
            Class<? extends OperationAttrs> attributesType, int inputCount, int outputCount) {
        return new OperationSignature(
                attributesType, inputCount, inputCount, outputCount, outputCount);
    }

    /**
     * Creates a signature with an inclusive input range and an exact output count.
     *
     * @param attributesType the exact non-null concrete attributes class accepted by the variant
     * @param minimumInputs the inclusive non-negative minimum logical input count
     * @param maximumInputs the inclusive maximum logical input count; use
     *     {@link Integer#MAX_VALUE} to accept every representable collection size
     * @param outputCount the exact positive logical output count
     * @return a new immutable signature with the supplied input range and exact output count
     * @throws NullPointerException if {@code attributesType} is {@code null}
     * @throws IllegalArgumentException if the bounds do not satisfy the canonical constructor
     *     requirements
     */
    public static OperationSignature inputRange(
            Class<? extends OperationAttrs> attributesType,
            int minimumInputs,
            int maximumInputs,
            int outputCount) {
        return new OperationSignature(
                attributesType,
                minimumInputs,
                maximumInputs,
                outputCount,
                outputCount);
    }

    /**
     * Returns the exact concrete attributes class accepted by this signature variant.
     *
     * @return the exact stored non-null attributes class token
     */
    @Override
    public Class<? extends OperationAttrs> attributesType() {
        return attributesType;
    }

    /**
     * Returns the inclusive minimum number of logical input positions.
     *
     * @return the stored non-negative minimum input count
     */
    @Override
    public int minimumInputs() {
        return minimumInputs;
    }

    /**
     * Returns the inclusive maximum number of logical input positions.
     *
     * @return the stored maximum input count, which is at least {@link #minimumInputs()}
     */
    @Override
    public int maximumInputs() {
        return maximumInputs;
    }

    /**
     * Returns the inclusive minimum number of logical output positions.
     *
     * @return the stored positive minimum output count
     */
    @Override
    public int minimumOutputs() {
        return minimumOutputs;
    }

    /**
     * Returns the inclusive maximum number of logical output positions.
     *
     * @return the stored maximum output count, which is at least {@link #minimumOutputs()}
     */
    @Override
    public int maximumOutputs() {
        return maximumOutputs;
    }

    /**
     * Reports whether an attributes value has the exact concrete class required by this variant.
     *
     * <p>Matching deliberately uses class identity rather than assignability or class-name text.
     * Operation attribute contracts are intended to be final immutable records or enum values,
     * and an explicitly different implementation represents a different semantic variant.</p>
     *
     * @param attrs the non-null attributes value to inspect without retaining
     * @return {@code true} exactly when {@code attrs.getClass()} is {@link #attributesType()}
     * @throws NullPointerException if {@code attrs} is {@code null}
     */
    public boolean acceptsAttributes(OperationAttrs attrs) {
        return Objects.requireNonNull(attrs, "attrs").getClass() == attributesType;
    }

    /**
     * Reports whether a logical input count lies inside this signature's inclusive input range.
     *
     * @param inputCount the count to test; negative values are outside every valid range
     * @return {@code true} when the count is between the stored input bounds, inclusive
     */
    public boolean acceptsInputCount(int inputCount) {
        return inputCount >= minimumInputs && inputCount <= maximumInputs;
    }

    /**
     * Reports whether a logical output count lies inside this signature's inclusive output range.
     *
     * @param outputCount the count to test; non-positive values are outside every valid range
     * @return {@code true} when the count is between the stored output bounds, inclusive
     */
    public boolean acceptsOutputCount(int outputCount) {
        return outputCount >= minimumOutputs && outputCount <= maximumOutputs;
    }

    /**
     * Validates the logical input and output counts of one operation occurrence.
     *
     * <p>Input count is checked first. The method allocates no collection and performs no operand,
     * descriptor, graph, compiler, backend, or execution validation.</p>
     *
     * @param inputCount the non-negative number of ordered logical input positions
     * @param outputCount the positive number of ordered logical output positions
     * @throws IllegalArgumentException if {@code inputCount} is outside the inclusive input range
     * @throws IllegalArgumentException if {@code outputCount} is outside the inclusive output
     *     range
     */
    public void validateOccurrence(int inputCount, int outputCount) {
        if (!acceptsInputCount(inputCount)) {
            throw new IllegalArgumentException(
                    "input count " + inputCount + " is outside accepted range ["
                            + minimumInputs + ", " + maximumInputs + "]");
        }
        if (!acceptsOutputCount(outputCount)) {
            throw new IllegalArgumentException(
                    "output count " + outputCount + " is outside accepted range ["
                            + minimumOutputs + ", " + maximumOutputs + "]");
        }
    }
}
