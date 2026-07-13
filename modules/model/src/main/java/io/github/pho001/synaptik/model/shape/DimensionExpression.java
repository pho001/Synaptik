package io.github.pho001.synaptik.model.shape;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.StringJoiner;

/**
 * Read-only symbolic formula or constraint retained by an {@link ExpressionDimension}.
 *
 * <p>The permitted forms describe canonical linear arithmetic, canonical symbolic products,
 * floor or ceiling division, and an identity-based constrained unknown. They contain model
 * values only: no form binds a runtime size, evaluates a graph, or carries compiler, storage, or
 * backend state.</p>
 */
public sealed interface DimensionExpression permits DimensionExpression.LinearCombination,
        DimensionExpression.Product, DimensionExpression.FloorDivision,
        DimensionExpression.CeilingDivision, DimensionExpression.Unknown {
    /**
     * A canonical positive-coefficient sum of dimensions plus a signed constant offset.
     *
     * <p>The coefficient map is immutable, non-empty, and independent of iteration order. Every
     * coefficient is positive. The signed offset permits formulas such as {@code N - 3}; later
     * binding remains responsible for satisfying the non-negative dimension invariant.</p>
     */
    final class LinearCombination implements DimensionExpression {
        private final Map<Dimension, Long> coefficients;
        private final long offset;

        /**
         * Creates an already canonical linear combination.
         *
         * @param coefficients non-null, non-empty dimension coefficients with non-null keys,
         *     non-null values, and strictly positive values
         * @param offset signed constant offset
         * @throws NullPointerException if the map, a key, or a value is {@code null}
         * @throws IllegalArgumentException if the map is empty or a coefficient is not positive
         */
        LinearCombination(Map<Dimension, Long> coefficients, long offset) {
            Objects.requireNonNull(coefficients, "coefficients");
            if (coefficients.isEmpty()) {
                throw new IllegalArgumentException("coefficients must not be empty");
            }
            for (Map.Entry<Dimension, Long> entry : coefficients.entrySet()) {
                Objects.requireNonNull(entry.getKey(), "coefficients key");
                Long coefficient = Objects.requireNonNull(
                        entry.getValue(), "coefficients value");
                if (coefficient <= 0) {
                    throw new IllegalArgumentException(
                            "coefficient must be positive: " + coefficient);
                }
            }
            this.coefficients = Map.copyOf(coefficients);
            this.offset = offset;
        }

        /**
         * Returns the immutable positive coefficient map.
         *
         * @return non-null, non-empty immutable map whose iteration order is not semantic
         */
        public Map<Dimension, Long> coefficients() {
            return coefficients;
        }

        /**
         * Returns the signed constant added to the symbolic terms.
         *
         * @return signed constant offset
         */
        public long offset() {
            return offset;
        }

        /**
         * Compares the complete coefficient map and signed offset structurally.
         *
         * @param other candidate object, which may be {@code null}
         * @return {@code true} when {@code other} is a linear combination with equal coefficients
         *     and offset, independently of coefficient-map iteration order
         */
        @Override
        public boolean equals(Object other) {
            return this == other
                    || (other instanceof LinearCombination combination
                    && offset == combination.offset
                    && coefficients.equals(combination.coefficients));
        }

        /**
         * Returns the structural hash of the coefficient map and signed offset.
         *
         * @return hash code consistent with {@link #equals(Object)}
         */
        @Override
        public int hashCode() {
            return 31 * coefficients.hashCode() + Long.hashCode(offset);
        }

        /**
         * Returns a readable mathematical-style diagnostic for this combination.
         *
         * @return non-null text containing every term and the signed offset; the format is not a
         *     serialization contract
         */
        @Override
        public String toString() {
            StringJoiner terms = new StringJoiner(" + ");
            coefficients.entrySet().stream()
                    .sorted((left, right) -> dimensionText(left.getKey())
                            .compareTo(dimensionText(right.getKey())))
                    .forEach(entry -> terms.add(entry.getValue() == 1
                            ? dimensionText(entry.getKey())
                            : entry.getValue() + " * " + dimensionText(entry.getKey())));
            String result = terms.toString();
            if (offset > 0) {
                return result + " + " + offset;
            }
            if (offset < 0) {
                return result + " - " + Long.toUnsignedString(-offset);
            }
            return result;
        }
    }

    /**
     * A canonical product of symbolic dimensions and one positive constant coefficient.
     *
     * <p>The factor map is immutable, non-empty, and independent of iteration order. Each key is
     * one complete symbolic factor and each strictly positive value is its exponent. Public
     * construction through {@link DimensionExpressions} folds static factors, flattens nested
     * products, combines structurally equal factors, applies the zero and one identities, and
     * reports coefficient or exponent overflow rather than wrapping. Equality and hashing compare
     * the coefficient and complete factor map structurally; diagnostic text is mathematical but
     * is not a serialization format. The expression retains only structural metadata and performs
     * no binding or evaluation.</p>
     */
    final class Product implements DimensionExpression {
        private final Map<Dimension, Long> factors;
        private final long coefficient;

        /**
         * Creates an already canonical symbolic product.
         *
         * @param factors non-null, non-empty factor map with non-null dimensions, non-null
         *     exponents, and strictly positive exponents
         * @param coefficient strictly positive constant coefficient
         * @throws NullPointerException if the map, a key, or a value is {@code null}
         * @throws IllegalArgumentException if the map is empty, an exponent is not positive, or
         *     {@code coefficient} is not positive
         */
        Product(Map<Dimension, Long> factors, long coefficient) {
            Objects.requireNonNull(factors, "factors");
            if (factors.isEmpty()) {
                throw new IllegalArgumentException("factors must not be empty");
            }
            for (Map.Entry<Dimension, Long> entry : factors.entrySet()) {
                Objects.requireNonNull(entry.getKey(), "factors key");
                Long exponent = Objects.requireNonNull(entry.getValue(), "factors value");
                if (exponent <= 0) {
                    throw new IllegalArgumentException(
                            "exponent must be positive: " + exponent);
                }
            }
            if (coefficient <= 0) {
                throw new IllegalArgumentException(
                        "coefficient must be positive: " + coefficient);
            }
            this.factors = Map.copyOf(factors);
            this.coefficient = coefficient;
        }

        /**
         * Returns the immutable map from symbolic factors to positive exponents.
         *
         * @return non-null, non-empty immutable factor map whose iteration order is not semantic
         */
        public Map<Dimension, Long> factors() {
            return factors;
        }

        /**
         * Returns the positive constant coefficient.
         *
         * @return strictly positive constant coefficient
         */
        public long coefficient() {
            return coefficient;
        }

        /**
         * Compares the complete factor map and coefficient structurally.
         *
         * @param other candidate object, which may be {@code null}
         * @return {@code true} when {@code other} has equal factors and coefficient, independently
         *     of factor-map iteration order
         */
        @Override
        public boolean equals(Object other) {
            return this == other
                    || (other instanceof Product product
                    && coefficient == product.coefficient
                    && factors.equals(product.factors));
        }

        /**
         * Returns the structural hash of the factor map and coefficient.
         *
         * @return hash code consistent with {@link #equals(Object)}
         */
        @Override
        public int hashCode() {
            return 31 * factors.hashCode() + Long.hashCode(coefficient);
        }

        /**
         * Returns a readable mathematical-style product diagnostic.
         *
         * @return non-null text containing the coefficient, factors, and exponents; the format is
         *     not a serialization contract
         */
        @Override
        public String toString() {
            StringJoiner terms = new StringJoiner(" * ");
            if (coefficient != 1) {
                terms.add(Long.toString(coefficient));
            }
            factors.entrySet().stream()
                    .sorted((left, right) -> dimensionText(left.getKey())
                            .compareTo(dimensionText(right.getKey())))
                    .forEach(entry -> terms.add(entry.getValue() == 1
                            ? dimensionText(entry.getKey())
                            : dimensionText(entry.getKey()) + "^" + entry.getValue()));
            return terms.toString();
        }
    }

    /** Describes non-negative integer floor division by a positive constant. */
    final class FloorDivision implements DimensionExpression {
        private final Dimension dividend;
        private final long divisor;

        /**
         * Creates a floor-division expression.
         *
         * @param dividend non-null dimension to divide
         * @param divisor positive constant divisor
         * @throws NullPointerException if {@code dividend} is {@code null}
         * @throws IllegalArgumentException if {@code divisor} is not positive
         */
        FloorDivision(Dimension dividend, long divisor) {
            this.dividend = Objects.requireNonNull(dividend, "dividend");
            if (divisor <= 0) {
                throw new IllegalArgumentException("divisor must be positive: " + divisor);
            }
            this.divisor = divisor;
        }

        /**
         * Returns the symbolic dividend.
         *
         * @return non-null immutable dimension
         */
        public Dimension dividend() {
            return dividend;
        }

        /**
         * Returns the positive divisor.
         *
         * @return positive constant divisor
         */
        public long divisor() {
            return divisor;
        }

        /**
         * Compares the dividend structurally and the divisor numerically.
         *
         * @param other candidate object, which may be {@code null}
         * @return {@code true} when {@code other} is an equal floor-division expression
         */
        @Override
        public boolean equals(Object other) {
            return this == other
                    || (other instanceof FloorDivision division
                    && divisor == division.divisor
                    && dividend.equals(division.dividend));
        }

        /**
         * Returns the structural hash of the dividend and divisor.
         *
         * @return hash code consistent with {@link #equals(Object)}
         */
        @Override
        public int hashCode() {
            return 31 * dividend.hashCode() + Long.hashCode(divisor);
        }

        /**
         * Returns a readable floor-division diagnostic.
         *
         * @return non-null text in {@code floorDiv(dividend, divisor)} form; the format is not a
         *     serialization contract
         */
        @Override
        public String toString() {
            return "floorDiv(" + dimensionText(dividend) + ", " + divisor + ")";
        }
    }

    /** Describes non-negative integer ceiling division by a positive constant. */
    final class CeilingDivision implements DimensionExpression {
        private final Dimension dividend;
        private final long divisor;

        /**
         * Creates a ceiling-division expression.
         *
         * @param dividend non-null dimension to divide
         * @param divisor positive constant divisor
         * @throws NullPointerException if {@code dividend} is {@code null}
         * @throws IllegalArgumentException if {@code divisor} is not positive
         */
        CeilingDivision(Dimension dividend, long divisor) {
            this.dividend = Objects.requireNonNull(dividend, "dividend");
            if (divisor <= 0) {
                throw new IllegalArgumentException("divisor must be positive: " + divisor);
            }
            this.divisor = divisor;
        }

        /**
         * Returns the symbolic dividend.
         *
         * @return non-null immutable dimension
         */
        public Dimension dividend() {
            return dividend;
        }

        /**
         * Returns the positive divisor.
         *
         * @return positive constant divisor
         */
        public long divisor() {
            return divisor;
        }

        /**
         * Compares the dividend structurally and the divisor numerically.
         *
         * @param other candidate object, which may be {@code null}
         * @return {@code true} when {@code other} is an equal ceiling-division expression
         */
        @Override
        public boolean equals(Object other) {
            return this == other
                    || (other instanceof CeilingDivision division
                    && divisor == division.divisor
                    && dividend.equals(division.dividend));
        }

        /**
         * Returns the structural hash of the dividend and divisor.
         *
         * @return hash code consistent with {@link #equals(Object)}
         */
        @Override
        public int hashCode() {
            return 31 * dividend.hashCode() + Long.hashCode(divisor);
        }

        /**
         * Returns a readable ceiling-division diagnostic.
         *
         * @return non-null text in {@code ceilDiv(dividend, divisor)} form; the format is not a
         *     serialization contract
         */
        @Override
        public String toString() {
            return "ceilDiv(" + dimensionText(dividend) + ", " + divisor + ")";
        }
    }

    /**
     * A distinct unknown extent with an inclusive non-negative minimum and optional upper bound.
     *
     * <p>This class deliberately retains object-identity equality. Equal-looking bounds do not
     * imply that two independently generated unknown extents are the same value.</p>
     */
    final class Unknown implements DimensionExpression {
        private final long minimum;
        private final Optional<Dimension> maximum;

        /**
         * Creates a constrained unknown expression.
         *
         * @param minimum inclusive non-negative lower bound
         * @param maximum non-null optional inclusive upper-bound dimension
         * @throws NullPointerException if {@code maximum} is {@code null}
         * @throws IllegalArgumentException if {@code minimum} is negative
         */
        Unknown(long minimum, Optional<Dimension> maximum) {
            if (minimum < 0) {
                throw new IllegalArgumentException(
                        "minimum must be non-negative: " + minimum);
            }
            this.minimum = minimum;
            this.maximum = Objects.requireNonNull(maximum, "maximum");
        }

        /**
         * Returns the inclusive lower bound.
         *
         * @return non-negative minimum extent
         */
        public long minimum() {
            return minimum;
        }

        /**
         * Returns the optional inclusive upper-bound dimension.
         *
         * @return non-null optional retaining the exact supplied maximum dimension reference
         */
        public Optional<Dimension> maximum() {
            return maximum;
        }

        /**
         * Returns a readable constraint diagnostic without defining value equality.
         *
         * @return non-null text containing the inclusive minimum and present maximum; the format
         *     is not a serialization contract
         */
        @Override
        public String toString() {
            return maximum
                    .map(dimension -> "unknown(min=" + minimum + ", max="
                            + dimensionText(dimension) + ")")
                    .orElseGet(() -> "unknown(min=" + minimum + ")");
        }
    }

    private static String dimensionText(Dimension dimension) {
        if (dimension instanceof StaticDimension staticDimension) {
            return Long.toString(staticDimension.size());
        }
        if (dimension instanceof DynamicDimension dynamicDimension) {
            return dynamicDimension.symbol();
        }
        return "(" + ((ExpressionDimension) dimension).expression() + ")";
    }
}
