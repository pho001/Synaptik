package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.List;
import java.util.Objects;

/**
 * Carries a normalized reduction axis and an explicit ordered mapping from mask dimensions to
 * input axes for a masked sum or mean.
 *
 * <p>Element {@code maskInputAxes[i]} is the zero-based input-axis position to which mask
 * dimension {@code i} aligns. For example, mapping {@code [0, 1]} aligns a mask shaped like
 * {@code [batch, time]} to the first two axes of an input shaped like
 * {@code [batch, time, features]}; the omitted input axis two is an implicit broadcast dimension.
 * A strictly increasing mapping preserves mask-dimension order and prevents two mask dimensions
 * from claiming the same input axis. An empty mapping represents a rank-zero scalar mask.</p>
 *
 * <p>{@link AggregateReductionKind#SUM} and {@link AggregateReductionKind#MEAN} pair with this
 * value to describe masked, axis-removing reductions. Their later provenance order is
 * {@code [input, mask]}. A false aligned mask position excludes the corresponding input value.
 * Masked sum produces zero when no value is selected. Masked mean divides by the number of true
 * mask positions contributing to each output and also produces zero when that count is zero.</p>
 *
 * <p>This immutable semantic value stores neither the input nor the mask. It does not resolve a
 * mapping from shapes, prove axis bounds or dimension compatibility, derive a result descriptor,
 * construct provenance, inspect values, or execute a reduction. The supplied mapping is copied
 * after validation. Record-generated equality and hashing use both components; generated text is
 * diagnostic only and is not a serialization or execution contract.</p>
 *
 * @param axis the already normalized, non-negative reduction-axis index that is removed from the
 *     eventual result
 * @param maskInputAxes the non-null, strictly increasing input-axis positions for successive mask
 *     dimensions; an empty list represents a scalar mask, and the stored value is an immutable
 *     snapshot
 */
public record MaskedReductionAttrs(
        int axis,
        List<Integer> maskInputAxes) implements OperationAttrs {
    /**
     * Creates immutable attributes for a masked, axis-removing sum or mean.
     *
     * <p>Validation follows component and element order. The reduction axis is checked first. The
     * mapping reference is then checked for null, followed by each element from index zero: null
     * check, non-negative check, and, from index one, strict-order check. Only after every value
     * passes is the caller-owned list copied with {@link List#copyOf(java.util.Collection)}. The
     * constructor does not normalize an axis, inspect shapes, or verify that mapped axes exist.</p>
     *
     * @param axis the already normalized reduction-axis index; must be non-negative
     * @param maskInputAxes the mask-dimension-to-input-axis mapping; must be non-null, contain no
     *     null or negative values, and be strictly increasing; an empty list is valid
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}; if the element at index {@code i} is
     *     negative, with message
     *     {@code maskInputAxes[<i>] must be non-negative: <value>}; or if the element at index
     *     {@code i} is not greater than its predecessor, with this message:
     *     <pre>{@code
     *     "maskInputAxes must be strictly increasing at index <i>:"
     *             + " previous=<previous>, current=<current>"
     *     }</pre>
     * @throws NullPointerException if {@code maskInputAxes} is {@code null}, with message
     *     {@code maskInputAxes}, or if the element at index {@code i} is {@code null}, with
     *     message {@code maskInputAxes[<i>]}
     */
    public MaskedReductionAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }

        Objects.requireNonNull(maskInputAxes, "maskInputAxes");
        for (int index = 0; index < maskInputAxes.size(); index++) {
            Integer mappedAxis = Objects.requireNonNull(
                    maskInputAxes.get(index), "maskInputAxes[" + index + "]");
            if (mappedAxis < 0) {
                throw new IllegalArgumentException(
                        "maskInputAxes[" + index + "] must be non-negative: " + mappedAxis);
            }
            if (index > 0) {
                int previous = maskInputAxes.get(index - 1);
                if (mappedAxis <= previous) {
                    throw new IllegalArgumentException(
                            "maskInputAxes must be strictly increasing at index "
                                    + index
                                    + ": previous="
                                    + previous
                                    + ", current="
                                    + mappedAxis);
                }
            }
        }
        maskInputAxes = List.copyOf(maskInputAxes);
    }

    /**
     * Returns the already normalized reduction-axis index.
     *
     * @return the exact non-negative axis supplied at construction; the eventual masked reduction
     *     removes this axis from its result
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the immutable ordered mapping from mask dimensions to input axes.
     *
     * @return the non-null immutable snapshot whose element at index {@code i} identifies the
     *     input axis aligned with mask dimension {@code i}; no list-object identity is promised
     */
    @Override
    public List<Integer> maskInputAxes() {
        return maskInputAxes;
    }
}
