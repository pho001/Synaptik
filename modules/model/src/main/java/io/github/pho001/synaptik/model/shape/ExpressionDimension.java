package io.github.pho001.synaptik.model.shape;

import java.util.Objects;

/**
 * A dynamic dimension represented by an exact symbolic formula or constrained unknown extent.
 *
 * <p>Instances are created through {@link DimensionExpressions}, which validates and canonicalizes
 * every public construction request. Exact formulas have structural value equality. Unknown
 * formulas retain the identity semantics of {@link DimensionExpression.Unknown}, so separately
 * generated unknowns remain distinct even when their bounds match.</p>
 */
public final class ExpressionDimension implements Dimension {
    private final DimensionExpression expression;

    /**
     * Creates a dimension from a validated expression.
     *
     * @param expression non-null immutable expression
     * @throws NullPointerException if {@code expression} is {@code null}
     */
    ExpressionDimension(DimensionExpression expression) {
        this.expression = Objects.requireNonNull(expression, "expression");
    }

    /**
     * Returns the retained expression for typed inspection.
     *
     * @return non-null immutable symbolic formula or constrained unknown
     */
    public DimensionExpression expression() {
        return expression;
    }

    /**
     * Compares the retained expression according to that expression form's equality contract.
     *
     * <p>Exact formulas compare structurally. Constrained unknowns compare only when they retain
     * the same {@link DimensionExpression.Unknown} object.</p>
     *
     * @param other candidate object, which may be {@code null}
     * @return {@code true} when {@code other} is an expression dimension whose retained expression
     *     is equal to this one
     */
    @Override
    public boolean equals(Object other) {
        return this == other
                || (other instanceof ExpressionDimension dimension
                && expression.equals(dimension.expression));
    }

    /**
     * Returns the hash of the retained expression.
     *
     * @return hash code consistent with {@link #equals(Object)}
     */
    @Override
    public int hashCode() {
        return expression.hashCode();
    }

    /**
     * Returns the retained expression's readable diagnostic text.
     *
     * @return non-null expression diagnostic; the format is not a serialization contract
     */
    @Override
    public String toString() {
        return expression.toString();
    }
}
