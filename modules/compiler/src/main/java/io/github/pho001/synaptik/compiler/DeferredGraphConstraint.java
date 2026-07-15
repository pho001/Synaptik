package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpression;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.ExpressionDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.math.BigInteger;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * One unresolved, occurrence-owned compile-time Shape obligation.
 *
 * <p>The constraint records why verification could not decide a predicate from compile-visible
 * descriptor facts. It is internal pass state, not a concrete-dimension binding, a serialization
 * contract, or a public execution check.
 *
 * @param nodeId non-null identity of the operation occurrence that owns the obligation
 * @param subject non-null, nonblank semantic role used in deterministic diagnostics
 * @param predicate non-null immutable predicate that remained undecidable
 */
record DeferredGraphConstraint(NodeId nodeId, String subject, GraphPredicate predicate) {
    /**
     * Creates one occurrence-owned unresolved obligation.
     *
     * @param nodeId non-null owning node identity
     * @param subject non-null, nonblank semantic role
     * @param predicate non-null undecidable predicate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if {@code subject} is blank
     */
    DeferredGraphConstraint {
        Objects.requireNonNull(nodeId, "nodeId");
        Objects.requireNonNull(subject, "subject");
        Objects.requireNonNull(predicate, "predicate");
        if (subject.isBlank()) {
            throw new IllegalArgumentException("subject must not be blank");
        }
    }
}

/**
 * Closed, binding-free predicate vocabulary used by captured-graph verification.
 *
 * <p>Predicates contain immutable model dimensions and Shapes only. The vocabulary deliberately
 * exposes no symbol assignment, substitution, serialization, or public evaluation surface.
 */
sealed interface GraphPredicate permits DimensionEqual, DimensionAtLeast, DimensionDivisible,
        ShapeElementCountEqual, ShapeElementCountValue, FitsWithin, AllOf, AnyOf {}

/**
 * Requires two dimensions to denote the same non-negative extent.
 *
 * @param left non-null left dimension
 * @param right non-null right dimension
 */
record DimensionEqual(Dimension left, Dimension right) implements GraphPredicate {
    /**
     * Creates a dimension-equality predicate.
     *
     * @param left non-null left dimension
     * @param right non-null right dimension
     * @throws NullPointerException if either argument is {@code null}
     */
    DimensionEqual { Objects.requireNonNull(left, "left"); Objects.requireNonNull(right, "right"); }
}

/**
 * Requires a dimension to be at least a non-negative constant.
 *
 * @param dimension non-null dimension to inspect
 * @param minimum non-negative inclusive minimum
 */
record DimensionAtLeast(Dimension dimension, long minimum) implements GraphPredicate {
    /**
     * Creates an inclusive minimum predicate.
     *
     * @param dimension non-null dimension to inspect
     * @param minimum non-negative inclusive minimum
     * @throws NullPointerException if {@code dimension} is {@code null}
     * @throws IllegalArgumentException if {@code minimum} is negative
     */
    DimensionAtLeast {
        Objects.requireNonNull(dimension, "dimension");
        if (minimum < 0) throw new IllegalArgumentException("minimum must be non-negative");
    }
}

/**
 * Requires a dimension to be evenly divisible by a positive constant.
 *
 * @param dimension non-null dimension to inspect
 * @param divisor positive divisor
 */
record DimensionDivisible(Dimension dimension, long divisor) implements GraphPredicate {
    /**
     * Creates a positive-divisor predicate.
     *
     * @param dimension non-null dimension to inspect
     * @param divisor positive divisor
     * @throws NullPointerException if {@code dimension} is {@code null}
     * @throws IllegalArgumentException if {@code divisor} is not positive
     */
    DimensionDivisible {
        Objects.requireNonNull(dimension, "dimension");
        if (divisor <= 0) throw new IllegalArgumentException("divisor must be positive");
    }
}

/**
 * Requires two Shapes to have the same logical element count.
 *
 * @param left non-null left Shape
 * @param right non-null right Shape
 */
record ShapeElementCountEqual(Shape left, Shape right) implements GraphPredicate {
    /**
     * Creates a Shape element-count-equality predicate.
     *
     * @param left non-null left Shape
     * @param right non-null right Shape
     * @throws NullPointerException if either argument is {@code null}
     */
    ShapeElementCountEqual { Objects.requireNonNull(left, "left"); Objects.requireNonNull(right, "right"); }
}

/**
 * Compares a Shape's logical element count with a non-negative constant.
 *
 * @param shape non-null Shape whose logical count is inspected
 * @param value non-negative comparison value
 * @param comparison non-null equality or inclusive-lower-bound relation
 */
record ShapeElementCountValue(Shape shape, long value, Comparison comparison)
        implements GraphPredicate {
    /** Logical element-count relations supported by {@link ShapeElementCountValue}. */
    enum Comparison {
        /** Requires exact equality with the supplied value. */
        EQUAL,
        /** Requires the logical count to be at least the supplied value. */
        AT_LEAST
    }

    /**
     * Creates a Shape-to-constant count predicate.
     *
     * @param shape non-null Shape to inspect
     * @param value non-negative comparison value
     * @param comparison non-null relation
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if {@code value} is negative
     */
    ShapeElementCountValue {
        Objects.requireNonNull(shape, "shape"); Objects.requireNonNull(comparison, "comparison");
        if (value < 0) throw new IllegalArgumentException("value must be non-negative");
    }
}

/**
 * Requires a non-negative start plus exactly one extent form to fit in a containing dimension.
 *
 * @param start non-negative start coordinate
 * @param extentShape nullable Shape extent; exactly one extent component must be non-null
 * @param extentDimension nullable dimension extent; exactly one extent component must be non-null
 * @param containing non-null containing dimension
 */
record FitsWithin(long start, Shape extentShape, Dimension extentDimension, Dimension containing)
        implements GraphPredicate {
    /**
     * Creates a fit predicate with exactly one extent representation.
     *
     * @param start non-negative start coordinate
     * @param extentShape nullable Shape extent
     * @param extentDimension nullable dimension extent
     * @param containing non-null containing dimension
     * @throws NullPointerException if {@code containing} is {@code null}
     * @throws IllegalArgumentException if {@code start} is negative or both/neither extent
     *     components are present
     */
    FitsWithin {
        if (start < 0) throw new IllegalArgumentException("start must be non-negative");
        if ((extentShape == null) == (extentDimension == null)) {
            throw new IllegalArgumentException("exactly one extent must be present");
        }
        Objects.requireNonNull(containing, "containing");
    }
    /**
     * Creates a fit predicate with a dimension extent.
     *
     * @param start non-negative start coordinate
     * @param extent non-null dimension extent
     * @param containing non-null containing dimension
     * @return the immutable fit predicate; never {@code null}
     * @throws NullPointerException if an extent or containing dimension is {@code null}
     * @throws IllegalArgumentException if {@code start} is negative
     */
    static FitsWithin dimension(long start, Dimension extent, Dimension containing) {
        return new FitsWithin(start, null, Objects.requireNonNull(extent, "extent"), containing);
    }
    /**
     * Creates a fit predicate with a Shape element-count extent.
     *
     * @param start non-negative start coordinate
     * @param extent non-null Shape extent
     * @param containing non-null containing dimension
     * @return the immutable fit predicate; never {@code null}
     * @throws NullPointerException if an extent or containing dimension is {@code null}
     * @throws IllegalArgumentException if {@code start} is negative
     */
    static FitsWithin shape(long start, Shape extent, Dimension containing) {
        return new FitsWithin(start, Objects.requireNonNull(extent, "extent"), null, containing);
    }
}

/**
 * Requires every ordered child predicate to hold.
 *
 * @param predicates non-null child list, snapshot on construction
 */
record AllOf(List<GraphPredicate> predicates) implements GraphPredicate {
    /**
     * Creates a conjunction from an immutable snapshot of ordered children.
     *
     * @param predicates non-null child predicates without null elements
     * @throws NullPointerException if the list or a contained element is {@code null}
     */
    AllOf { predicates = List.copyOf(Objects.requireNonNull(predicates, "predicates")); }
}

/**
 * Requires at least one ordered child predicate to hold.
 *
 * @param predicates non-null child list, snapshot on construction
 */
record AnyOf(List<GraphPredicate> predicates) implements GraphPredicate {
    /**
     * Creates a disjunction from an immutable snapshot of ordered children.
     *
     * @param predicates non-null child predicates without null elements
     * @throws NullPointerException if the list or a contained element is {@code null}
     */
    AnyOf { predicates = List.copyOf(Objects.requireNonNull(predicates, "predicates")); }
}

/** Outcome of conservative proof from current immutable compile-time facts. */
enum ProofStatus {
    /** The predicate follows from available facts. */
    PROVEN,
    /** Available facts contradict the predicate. */
    DISPROVEN,
    /** Available facts cannot prove or disprove the predicate. */
    DEFERRED
}

/**
 * Conservatively evaluates the closed predicate vocabulary without binding or unifying symbols.
 *
 * <p>Unavailable bounds and checked-arithmetic overflow make a proof undecidable; they are never
 * treated as contradiction.
 */
final class GraphPredicateProof {
    private GraphPredicateProof() {}

    /**
     * Evaluates one predicate from static, structural, and available bounded facts.
     *
     * @param predicate non-null predicate to evaluate
     * @return the three-valued proof outcome; never {@code null}
     * @throws NullPointerException if {@code predicate} is {@code null}
     */
    static ProofStatus evaluate(GraphPredicate predicate) {
        return switch (predicate) {
            case DimensionEqual p -> equal(p.left(), p.right());
            case DimensionAtLeast p -> atLeast(p.dimension(), p.minimum());
            case DimensionDivisible p -> divisible(p.dimension(), p.divisor());
            case ShapeElementCountEqual p -> countEqual(p.left(), p.right());
            case ShapeElementCountValue p -> countValue(p);
            case FitsWithin p -> fits(p);
            case AllOf p -> all(p.predicates());
            case AnyOf p -> any(p.predicates());
        };
    }

    private static ProofStatus equal(Dimension left, Dimension right) {
        if (left.equals(right)) return ProofStatus.PROVEN;
        OptionalLong ls = left.staticSize(); OptionalLong rs = right.staticSize();
        if (ls.isPresent() && rs.isPresent()) return ProofStatus.DISPROVEN;
        Bounds lb = bounds(left); Bounds rb = bounds(right);
        if (lb != null && rb != null && lb.maximum != null && rb.maximum != null
                && (lb.maximum < rb.minimum || rb.maximum < lb.minimum)) return ProofStatus.DISPROVEN;
        return ProofStatus.DEFERRED;
    }

    private static ProofStatus atLeast(Dimension d, long minimum) {
        Bounds b = bounds(d);
        if (b == null) return ProofStatus.DEFERRED;
        if (b.minimum >= minimum) return ProofStatus.PROVEN;
        return b.maximum != null && b.maximum < minimum ? ProofStatus.DISPROVEN : ProofStatus.DEFERRED;
    }

    private static ProofStatus divisible(Dimension d, long divisor) {
        if (divisor == 1) return ProofStatus.PROVEN;
        return d.staticSize().isPresent()
                ? (d.staticSize().getAsLong() % divisor == 0 ? ProofStatus.PROVEN : ProofStatus.DISPROVEN)
                : ProofStatus.DEFERRED;
    }

    private static ProofStatus countEqual(Shape left, Shape right) {
        if (left.equals(right)) return ProofStatus.PROVEN;
        BigInteger l = exactCount(left), r = exactCount(right);
        if (l != null && r != null) return l.equals(r) ? ProofStatus.PROVEN : ProofStatus.DISPROVEN;
        Bounds lb = countBounds(left), rb = countBounds(right);
        if (lb != null && rb != null && lb.maximum != null && rb.maximum != null
                && (lb.maximum < rb.minimum || rb.maximum < lb.minimum)) return ProofStatus.DISPROVEN;
        return ProofStatus.DEFERRED;
    }

    private static ProofStatus countValue(ShapeElementCountValue p) {
        BigInteger exact = exactCount(p.shape());
        if (exact != null) {
            int cmp = exact.compareTo(BigInteger.valueOf(p.value()));
            return (p.comparison() == ShapeElementCountValue.Comparison.EQUAL ? cmp == 0 : cmp >= 0)
                    ? ProofStatus.PROVEN : ProofStatus.DISPROVEN;
        }
        Bounds b = countBounds(p.shape());
        if (b == null) return ProofStatus.DEFERRED;
        if (p.comparison() == ShapeElementCountValue.Comparison.AT_LEAST) {
            if (b.minimum >= p.value()) return ProofStatus.PROVEN;
            if (b.maximum != null && b.maximum < p.value()) return ProofStatus.DISPROVEN;
        } else {
            if (b.minimum == p.value() && b.maximum != null && b.maximum == p.value()) return ProofStatus.PROVEN;
            if (b.minimum > p.value() || b.maximum != null && b.maximum < p.value()) return ProofStatus.DISPROVEN;
        }
        return ProofStatus.DEFERRED;
    }

    private static ProofStatus fits(FitsWithin p) {
        Bounds e = p.extentDimension() == null ? countBounds(p.extentShape()) : bounds(p.extentDimension());
        Bounds c = bounds(p.containing());
        if (e == null || c == null) return ProofStatus.DEFERRED;
        Long minEnd = add(p.start(), e.minimum), maxEnd = e.maximum == null ? null : add(p.start(), e.maximum);
        if (minEnd == null) return ProofStatus.DEFERRED;
        if (c.maximum != null && minEnd > c.maximum) return ProofStatus.DISPROVEN;
        if (maxEnd != null && maxEnd <= c.minimum) return ProofStatus.PROVEN;
        return ProofStatus.DEFERRED;
    }

    private static ProofStatus all(List<GraphPredicate> ps) {
        boolean deferred = false;
        for (GraphPredicate p : ps) { ProofStatus s = evaluate(p); if (s == ProofStatus.DISPROVEN) return s; deferred |= s == ProofStatus.DEFERRED; }
        return deferred ? ProofStatus.DEFERRED : ProofStatus.PROVEN;
    }

    private static ProofStatus any(List<GraphPredicate> ps) {
        boolean deferred = false;
        for (GraphPredicate p : ps) { ProofStatus s = evaluate(p); if (s == ProofStatus.PROVEN) return s; deferred |= s == ProofStatus.DEFERRED; }
        return deferred ? ProofStatus.DEFERRED : ProofStatus.DISPROVEN;
    }

    private static BigInteger exactCount(Shape shape) {
        BigInteger result = BigInteger.ONE;
        for (Dimension d : shape.dimensions()) {
            if (!(d instanceof StaticDimension s)) return null;
            result = result.multiply(BigInteger.valueOf(s.size()));
        }
        return result;
    }

    private static Bounds countBounds(Shape shape) {
        long min = 1; Long max = 1L;
        for (Dimension d : shape.dimensions()) {
            Bounds b = bounds(d); if (b == null) return null;
            Long nextMin = multiply(min, b.minimum); if (nextMin == null) return null; min = nextMin;
            max = max == null || b.maximum == null ? null : multiply(max, b.maximum);
        }
        return new Bounds(min, max);
    }

    private static Bounds bounds(Dimension d) {
        if (d instanceof StaticDimension s) return new Bounds(s.size(), s.size());
        if (d instanceof DynamicDimension) return new Bounds(0, null);
        DimensionExpression e = ((ExpressionDimension) d).expression();
        try {
            return switch (e) {
                case DimensionExpression.Unknown u -> new Bounds(u.minimum(), u.maximum().map(GraphPredicateProof::bounds).map(b -> b.maximum).orElse(null));
                case DimensionExpression.LinearCombination l -> linearBounds(l);
                case DimensionExpression.Product p -> productBounds(p);
                case DimensionExpression.FloorDivision f -> divideBounds(bounds(f.dividend()), f.divisor(), false);
                case DimensionExpression.CeilingDivision c -> divideBounds(bounds(c.dividend()), c.divisor(), true);
            };
        } catch (ArithmeticException ignored) { return null; }
    }

    private static Bounds linearBounds(DimensionExpression.LinearCombination e) {
        long min = e.offset(); Long max = e.offset();
        for (var entry : e.coefficients().entrySet()) {
            Bounds b = bounds(entry.getKey()); if (b == null) return null;
            min = Math.addExact(min, Math.multiplyExact(b.minimum, entry.getValue()));
            max = max == null || b.maximum == null ? null : Math.addExact(max, Math.multiplyExact(b.maximum, entry.getValue()));
        }
        return new Bounds(min, max);
    }

    private static Bounds productBounds(DimensionExpression.Product e) {
        long min = e.coefficient(); Long max = e.coefficient();
        for (var entry : e.factors().entrySet()) {
            Bounds b = bounds(entry.getKey()); if (b == null || b.minimum < 0) return null;
            for (long i = 0; i < entry.getValue(); i++) {
                min = Math.multiplyExact(min, b.minimum);
                max = max == null || b.maximum == null ? null : Math.multiplyExact(max, b.maximum);
            }
        }
        return new Bounds(min, max);
    }

    private static Bounds divideBounds(Bounds b, long divisor, boolean ceiling) {
        if (b == null || b.minimum < 0) return null;
        long min = ceiling ? ceilDiv(b.minimum, divisor) : b.minimum / divisor;
        Long max = b.maximum == null ? null : (ceiling ? ceilDiv(b.maximum, divisor) : b.maximum / divisor);
        return new Bounds(min, max);
    }

    private static long ceilDiv(long value, long divisor) { return value / divisor + (value % divisor == 0 ? 0 : 1); }
    private static Long add(long a, long b) { try { return Math.addExact(a, b); } catch (ArithmeticException e) { return null; } }
    private static Long multiply(long a, long b) { try { return Math.multiplyExact(a, b); } catch (ArithmeticException e) { return null; } }
    private record Bounds(long minimum, Long maximum) {}
}
