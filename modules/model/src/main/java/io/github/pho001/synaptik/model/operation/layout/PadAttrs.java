package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.List;
import java.util.Objects;

/**
 * Carries normalized per-axis widths and one exact typed value for constant padding.
 *
 * <p>At index {@code i}, {@code before[i]} and {@code after[i]} give the non-negative numbers of
 * logical positions inserted before and after the input extent at the same normalized axis
 * position. Input {@code [10, 20]} with before {@code [1]}, after {@code [2]}, and constant
 * {@code -1} semantically requests {@code [-1, 10, 20, -1, -1]}. The record expresses this
 * request without constructing a Tensor, deriving a Shape, or populating values.</p>
 *
 * <p>The caller-owned lists are validated in ascending index order and copied only after all
 * validation succeeds. The stored values are immutable snapshots that preserve list order and
 * contents but not caller list identity. Two empty lists describe rank-zero scalar identity
 * parameters. {@link Long#MAX_VALUE} is structurally valid because rank matching and result-Shape
 * arithmetic are deferred.</p>
 *
 * <p>{@code constantValue} is retained by exact immutable reference. Its data type and bits are
 * not converted, ranged, rounded, saturated, or normalized. This record has no input descriptor,
 * so receiver compatibility is validated later by the public Tensor construction boundary.
 * Record-generated equality and hashing compose the value's exact typed-bit semantics.</p>
 *
 * <p>These attributes contain no Tensor, result Shape or DataType, layout, storage,
 * materialization, provenance, gradient, compiler, backend, ONNX, or execution behavior.</p>
 *
 * @param before the non-null ordered before widths; elements must be non-null and non-negative,
 *     and the stored value is an immutable snapshot
 * @param after the non-null ordered after widths paired by index with {@code before}; elements
 *     must be non-null and non-negative, and the stored value is an immutable snapshot
 * @param constantValue the non-null exact typed padding constant retained by reference without
 *     conversion or receiver compatibility validation
 */
public record PadAttrs(
        List<Long> before,
        List<Long> after,
        ScalarValue constantValue) implements OperationAttrs {
    /**
     * Creates immutable normalized constant-padding parameters.
     *
     * <p>Validation first null-checks the two list references and constant in component order,
     * then checks list sizes. Entries are inspected from index zero upward. At each index the
     * before and after elements are null-checked in that order, followed by the non-negative
     * before and after width checks. Only after every entry succeeds are immutable list snapshots
     * stored.</p>
     *
     * <p>The constant receives no type conversion or input compatibility check. Construction
     * performs no rank lookup, output-extent arithmetic, overflow check, or result inference.
     * Empty lists and {@link Long#MAX_VALUE} widths are structurally valid.</p>
     *
     * @param before the ordered before widths; must be non-null, match {@code after} in size, and
     *     contain only non-null non-negative values
     * @param after the ordered after widths; must be non-null, match {@code before} in size, and
     *     contain only non-null non-negative values
     * @param constantValue non-null exact typed padding constant retained by reference
     * @throws NullPointerException if {@code before}, {@code after}, or {@code constantValue} is
     *     {@code null}, checked in that order with its component name as the message; or if an
     *     element is {@code null}, with the exact indexed message {@code before[i]} or
     *     {@code after[i]}
     * @throws IllegalArgumentException if list sizes differ, with message
     *     {@code before and after must have matching sizes}; or if a width is negative, with the
     *     exact indexed message {@code before[i] must be non-negative: value} or
     *     {@code after[i] must be non-negative: value}
     */
    public PadAttrs {
        Objects.requireNonNull(before, "before");
        Objects.requireNonNull(after, "after");
        Objects.requireNonNull(constantValue, "constantValue");
        if (before.size() != after.size()) {
            throw new IllegalArgumentException("before and after must have matching sizes");
        }

        for (int index = 0; index < before.size(); index++) {
            Long beforeWidth =
                    Objects.requireNonNull(before.get(index), "before[" + index + "]");
            Long afterWidth = Objects.requireNonNull(after.get(index), "after[" + index + "]");
            if (beforeWidth < 0) {
                throw new IllegalArgumentException(
                        "before[" + index + "] must be non-negative: " + beforeWidth);
            }
            if (afterWidth < 0) {
                throw new IllegalArgumentException(
                        "after[" + index + "] must be non-negative: " + afterWidth);
            }
        }

        before = List.copyOf(before);
        after = List.copyOf(after);
    }

    /**
     * Returns the immutable ordered before-padding widths.
     *
     * <p>Entry {@code i} is paired with {@link #after()} at the same axis position. The returned
     * list is the stored immutable snapshot; no identity relationship with the caller's original
     * list is promised.</p>
     *
     * @return the non-null immutable before-width snapshot; an empty list denotes scalar identity
     */
    @Override
    public List<Long> before() {
        return before;
    }

    /**
     * Returns the immutable ordered after-padding widths.
     *
     * <p>Entry {@code i} is paired with {@link #before()} at the same axis position. The returned
     * list is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable after-width snapshot; an empty list denotes scalar identity
     */
    @Override
    public List<Long> after() {
        return after;
    }

    /**
     * Returns the exact typed padding constant supplied at construction.
     *
     * <p>No receiver compatibility or conversion policy is implied by this stored semantic
     * value.</p>
     *
     * @return the non-null retained padding constant by its original reference
     */
    @Override
    public ScalarValue constantValue() {
        return constantValue;
    }
}
