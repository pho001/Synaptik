package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Carries normalized finite coordinate sequences for a logical slice as four parallel lists.
 *
 * <p>Entry {@code i} selects {@code lengths[i]} coordinates on normalized input axis
 * {@code axes[i]}. Coordinate {@code k} is {@code starts[i] + k * steps[i]} for
 * {@code 0 <= k < lengths[i]}. Steps are signed and non-zero: a positive step traverses toward
 * larger coordinates, while a negative step traverses toward smaller coordinates. For example,
 * start {@code 1}, length {@code 2}, and step {@code 2} selects {@code [1, 3]}; start {@code 4},
 * length {@code 5}, and step {@code -1} selects {@code [4, 3, 2, 1, 0]}.</p>
 *
 * <p>Starts, lengths, and axes are non-negative, axes are distinct, and a non-empty sequence must
 * end at a representable non-negative coordinate. An empty entry has canonical start zero and
 * retains its signed non-zero step. Four empty lists describe an identity slice that constrains
 * no axes. The normalized representation stores no exclusive-end sentinel, raw request bound,
 * input Shape, or rank; caller-facing bound normalization, clamping, and axis bounds therefore
 * belong to the Shape-aware Tensor expression boundary.</p>
 *
 * <p>Coordinates, lengths, and steps use {@code long}; axes use {@code int}. The four lists are
 * validated in ascending entry order and copied only after every entry succeeds. Each accessor
 * returns the stored immutable snapshot, and entry order plus all four list values participate in
 * record equality and hashing. A {@link Long#MIN_VALUE} step is valid when the declared finite
 * sequence has a representable non-negative final coordinate, such as a one-coordinate sequence.</p>
 *
 * <p>These attributes describe logical semantics only. They calculate no result Shape or layout,
 * attach no Tensor or storage, create no provenance, and define no gradient, compiler, backend,
 * materialization, serialization, or execution behavior.</p>
 *
 * @param starts the non-null ordered normalized first coordinates; elements must be non-null and
 *     non-negative, empty entries must use zero, and the stored value is an immutable snapshot
 * @param lengths the non-null ordered selected-coordinate counts; elements must be non-null and
 *     non-negative, and the stored value is an immutable snapshot
 * @param axes the non-null ordered normalized input axes; elements must be non-null,
 *     non-negative, and distinct, and the stored value is an immutable snapshot
 * @param steps the non-null ordered signed coordinate increments; elements must be non-null and
 *     non-zero, and the stored value is an immutable snapshot
 */
public record SliceAttrs(
        List<Long> starts,
        List<Long> lengths,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs {
    /**
     * Creates immutable normalized slice parameters from four exactly paired lists.
     *
     * <p>Validation null-checks the four list references in component order, checks their sizes,
     * then inspects entries from index zero upward. At each index it null-checks start, length,
     * axis, and step before checking non-negative start and length, non-negative distinct axis,
     * non-zero step, canonical empty start, and the checked final coordinate for a non-empty
     * sequence. Only after all entries pass are immutable snapshots stored in component order.</p>
     *
     * <p>Construction performs no raw-bound normalization, rank or dimension lookup, clamping,
     * result-Shape derivation, or layout decision. It does verify that its intrinsic normalized
     * sequence is finite and representable.</p>
     *
     * @param starts the ordered inclusive normalized starts; must be non-null, match every other
     *     list size, and contain only non-null non-negative values
     * @param lengths the ordered selected-coordinate counts; must be non-null, match every other
     *     list size, and contain only non-null non-negative values
     * @param axes the ordered normalized input axes; must be non-null, match every other list size,
     *     and contain only non-null, non-negative, unique values
     * @param steps the ordered signed coordinate increments; must be non-null, match every other
     *     list size, and contain only non-null non-zero values
     * @throws NullPointerException if a list is {@code null}, with its component name as the
     *     message, or if an element is {@code null}, with its component name and index as the
     *     message
     * @throws IllegalArgumentException if list sizes differ; if a start, length, or axis is
     *     negative; if an axis is repeated; if a step is zero; if an empty entry has a non-zero
     *     start; or if a non-empty entry's final coordinate is negative
     * @throws ArithmeticException if calculating a non-empty entry's final coordinate overflows
     */
    public SliceAttrs {
        Objects.requireNonNull(starts, "starts");
        Objects.requireNonNull(lengths, "lengths");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(steps, "steps");
        if (starts.size() != lengths.size()
                || starts.size() != axes.size()
                || starts.size() != steps.size()) {
            throw new IllegalArgumentException(
                    "starts, lengths, axes, and steps must have matching sizes");
        }

        HashSet<Integer> seenAxes = new HashSet<>();
        for (int index = 0; index < starts.size(); index++) {
            Long start = Objects.requireNonNull(starts.get(index), "starts[" + index + "]");
            Long length = Objects.requireNonNull(lengths.get(index), "lengths[" + index + "]");
            Integer axis = Objects.requireNonNull(axes.get(index), "axes[" + index + "]");
            Long step = Objects.requireNonNull(steps.get(index), "steps[" + index + "]");
            if (start < 0) {
                throw new IllegalArgumentException(
                        "starts[" + index + "] must be non-negative: " + start);
            }
            if (length < 0) {
                throw new IllegalArgumentException(
                        "lengths[" + index + "] must be non-negative: " + length);
            }
            if (axis < 0) {
                throw new IllegalArgumentException(
                        "axes[" + index + "] must be non-negative: " + axis);
            }
            if (!seenAxes.add(axis)) {
                throw new IllegalArgumentException(
                        "axes contains duplicate axis " + axis + " at index " + index);
            }
            if (step == 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be non-zero: " + step);
            }
            if (length == 0) {
                if (start != 0) {
                    throw new IllegalArgumentException(
                            "starts[" + index + "] must be zero when lengths[" + index
                                    + "] is zero: " + start);
                }
            } else {
                long last = Math.addExact(
                        start, Math.multiplyExact(Math.subtractExact(length, 1L), step));
                if (last < 0) {
                    throw new IllegalArgumentException(
                            "last slice coordinate at index " + index
                                    + " must be non-negative: " + last);
                }
            }
        }

        starts = List.copyOf(starts);
        lengths = List.copyOf(lengths);
        axes = List.copyOf(axes);
        steps = List.copyOf(steps);
    }

    /**
     * Returns the immutable ordered inclusive normalized starts.
     *
     * <p>Entry {@code i} is paired exactly with {@link #lengths()}, {@link #axes()}, and
     * {@link #steps()} at the same index. The returned list is the stored immutable snapshot; no
     * identity relationship with the caller's original list is promised.</p>
     *
     * @return the non-null immutable inclusive-start snapshot; an empty list constrains no axes
     */
    @Override
    public List<Long> starts() {
        return starts;
    }

    /**
     * Returns the immutable ordered selected-coordinate counts.
     *
     * <p>Entry {@code i} is the exact finite length of the sequence beginning at
     * {@link #starts()} entry {@code i}. Zero denotes a canonical empty entry whose start is zero.
     * The returned list is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable non-negative length snapshot; an empty list constrains no axes
     */
    @Override
    public List<Long> lengths() {
        return lengths;
    }

    /**
     * Returns the immutable ordered distinct normalized input axes.
     *
     * <p>The values are structurally non-negative and unique but are not checked against an input
     * rank. The returned list is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable normalized-axis snapshot; an empty list constrains no axes
     */
    @Override
    public List<Integer> axes() {
        return axes;
    }

    /**
     * Returns the immutable ordered signed non-zero coordinate increments.
     *
     * <p>Entry {@code i} advances the finite sequence at the same index; its sign determines the
     * direction. The returned list is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable signed non-zero step snapshot; an empty list constrains no axes
     */
    @Override
    public List<Long> steps() {
        return steps;
    }
}
