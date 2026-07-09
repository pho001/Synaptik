package io.github.pho001.synaptik.model.shape;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Validated construction and canonicalization boundary for symbolic dimension arithmetic.
 *
 * <p>Every operation returns the simplest truthful {@link Dimension}: static arithmetic is folded,
 * neutral operations preserve the input reference, and linear terms are flattened and combined.
 * This class creates model values only. It does not bind or evaluate runtime dimensions.</p>
 */
public final class DimensionExpressions {
    /** Prevents instantiation of this field-free construction boundary. */
    private DimensionExpressions() {
    }

    /**
     * Adds two dimensions using checked arithmetic and canonical linear-term combination.
     *
     * @param left non-null left dimension
     * @param right non-null right dimension
     * @return non-null canonical sum; fully static sums are {@link StaticDimension} values, and
     *     adding static zero returns the exact opposing input reference
     * @throws NullPointerException if {@code left} or {@code right} is {@code null}
     * @throws ArithmeticException if a static value, coefficient, or offset overflows {@code long}
     */
    public static Dimension add(Dimension left, Dimension right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        if (isStaticZero(right)) {
            return left;
        }
        if (isStaticZero(left)) {
            return right;
        }

        Map<Dimension, Long> coefficients = new LinkedHashMap<>();
        long offset = addTerms(coefficients, 0, left, 1);
        offset = addTerms(coefficients, offset, right, 1);
        return canonicalLinearCombination(coefficients, offset);
    }

    /**
     * Adds a signed constant offset to a dimension using checked arithmetic.
     *
     * @param input non-null input dimension
     * @param offset signed constant offset
     * @return non-null canonical result; the exact input reference when {@code offset} is zero
     * @throws NullPointerException if {@code input} is {@code null}
     * @throws ArithmeticException if a static result or accumulated offset overflows {@code long}
     * @throws IllegalArgumentException if a fully static result is negative
     */
    public static Dimension addConstant(Dimension input, long offset) {
        Objects.requireNonNull(input, "input");
        if (offset == 0) {
            return input;
        }

        Map<Dimension, Long> coefficients = new LinkedHashMap<>();
        long combinedOffset = addTerms(coefficients, 0, input, 1);
        combinedOffset = Math.addExact(combinedOffset, offset);
        return canonicalLinearCombination(coefficients, combinedOffset);
    }

    /**
     * Multiplies a dimension by a non-negative constant using checked arithmetic.
     *
     * @param input non-null input dimension
     * @param factor non-negative constant factor
     * @return non-null canonical product; a static zero for factor zero and the exact input
     *     reference for factor one
     * @throws NullPointerException if {@code input} is {@code null}
     * @throws IllegalArgumentException if {@code factor} is negative
     * @throws ArithmeticException if a static value, coefficient, or offset overflows {@code long}
     */
    public static Dimension multiply(Dimension input, long factor) {
        Objects.requireNonNull(input, "input");
        if (factor < 0) {
            throw new IllegalArgumentException("factor must be non-negative: " + factor);
        }
        if (factor == 0) {
            return new StaticDimension(0);
        }
        if (factor == 1) {
            return input;
        }

        Map<Dimension, Long> coefficients = new LinkedHashMap<>();
        long offset = addTerms(coefficients, 0, input, factor);
        return canonicalLinearCombination(coefficients, offset);
    }

    /**
     * Divides a dimension by a positive constant with non-negative integer floor semantics.
     *
     * @param input non-null dividend
     * @param divisor positive constant divisor
     * @return non-null canonical quotient; static when the input is static and the exact input
     *     reference when {@code divisor} is one
     * @throws NullPointerException if {@code input} is {@code null}
     * @throws IllegalArgumentException if {@code divisor} is not positive
     */
    public static Dimension floorDivide(Dimension input, long divisor) {
        Objects.requireNonNull(input, "input");
        requirePositiveDivisor(divisor);
        if (divisor == 1) {
            return input;
        }
        if (input instanceof StaticDimension staticDimension) {
            return new StaticDimension(staticDimension.size() / divisor);
        }
        return new ExpressionDimension(new DimensionExpression.FloorDivision(input, divisor));
    }

    /**
     * Divides a dimension by a positive constant with non-negative integer ceiling semantics.
     *
     * @param input non-null dividend
     * @param divisor positive constant divisor
     * @return non-null canonical quotient; static when the input is static and the exact input
     *     reference when {@code divisor} is one
     * @throws NullPointerException if {@code input} is {@code null}
     * @throws IllegalArgumentException if {@code divisor} is not positive
     */
    public static Dimension ceilingDivide(Dimension input, long divisor) {
        Objects.requireNonNull(input, "input");
        requirePositiveDivisor(divisor);
        if (divisor == 1) {
            return input;
        }
        if (input instanceof StaticDimension staticDimension) {
            long value = staticDimension.size();
            long quotient = value / divisor;
            return new StaticDimension(value % divisor == 0 ? quotient : quotient + 1);
        }
        return new ExpressionDimension(new DimensionExpression.CeilingDivision(input, divisor));
    }

    /**
     * Creates a distinct unknown extent with inclusive lower and optional upper bounds.
     *
     * @param minimum inclusive non-negative lower bound
     * @param maximum non-null optional inclusive upper-bound dimension; its exact reference is
     *     retained
     * @return non-null expression dimension that is unequal to every result of a separate unknown
     *     construction call, even when the bounds match
     * @throws NullPointerException if {@code maximum} is {@code null}
     * @throws IllegalArgumentException if {@code minimum} is negative or a static maximum is less
     *     than {@code minimum}
     */
    public static Dimension unknown(long minimum, Optional<Dimension> maximum) {
        if (minimum < 0) {
            throw new IllegalArgumentException("minimum must be non-negative: " + minimum);
        }
        Objects.requireNonNull(maximum, "maximum");
        if (maximum.isPresent() && maximum.orElseThrow() instanceof StaticDimension staticMaximum
                && staticMaximum.size() < minimum) {
            throw new IllegalArgumentException(
                    "maximum static size must be greater than or equal to minimum: minimum="
                            + minimum + ", maximum=" + staticMaximum.size());
        }
        return new ExpressionDimension(new DimensionExpression.Unknown(minimum, maximum));
    }

    private static long addTerms(
            Map<Dimension, Long> coefficients,
            long offset,
            Dimension dimension,
            long factor) {
        if (dimension instanceof StaticDimension staticDimension) {
            return Math.addExact(offset, Math.multiplyExact(staticDimension.size(), factor));
        }
        if (dimension instanceof ExpressionDimension expressionDimension
                && expressionDimension.expression()
                instanceof DimensionExpression.LinearCombination combination) {
            long combinedOffset = Math.addExact(
                    offset, Math.multiplyExact(combination.offset(), factor));
            for (Map.Entry<Dimension, Long> entry : combination.coefficients().entrySet()) {
                addCoefficient(
                        coefficients,
                        entry.getKey(),
                        Math.multiplyExact(entry.getValue(), factor));
            }
            return combinedOffset;
        }
        addCoefficient(coefficients, dimension, factor);
        return offset;
    }

    private static void addCoefficient(
            Map<Dimension, Long> coefficients, Dimension dimension, long coefficient) {
        Long existing = coefficients.get(dimension);
        coefficients.put(
                dimension,
                existing == null ? coefficient : Math.addExact(existing, coefficient));
    }

    private static Dimension canonicalLinearCombination(
            Map<Dimension, Long> coefficients, long offset) {
        if (coefficients.isEmpty()) {
            if (offset < 0) {
                throw new IllegalArgumentException(
                        "dimension expression must not produce a negative static size: " + offset);
            }
            return new StaticDimension(offset);
        }
        if (coefficients.size() == 1 && offset == 0) {
            Map.Entry<Dimension, Long> term = coefficients.entrySet().iterator().next();
            if (term.getValue() == 1) {
                return term.getKey();
            }
        }
        return new ExpressionDimension(
                new DimensionExpression.LinearCombination(coefficients, offset));
    }

    private static void requirePositiveDivisor(long divisor) {
        if (divisor <= 0) {
            throw new IllegalArgumentException("divisor must be positive: " + divisor);
        }
    }

    private static boolean isStaticZero(Dimension dimension) {
        return dimension instanceof StaticDimension staticDimension
                && staticDimension.size() == 0;
    }
}
