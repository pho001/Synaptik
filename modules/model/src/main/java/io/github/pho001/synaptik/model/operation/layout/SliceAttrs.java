package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Carries normalized parameters for a positive-step logical slice as four parallel lists.
 *
 * <p>At each index {@code i}, {@code starts[i]} is the inclusive first coordinate,
 * {@code ends[i]} is the exclusive bound, {@code axes[i]} is the distinct normalized input axis,
 * and {@code steps[i]} is the positive coordinate increment. For a conceptual Shape
 * {@code [3, 6]}, starts {@code [0, 1]}, ends {@code [3, 6]}, axes {@code [0, 1]}, and steps
 * {@code [1, 2]} select rows beginning at zero and columns {@code 1, 3, 5}. This example defines
 * logical selection only; no Tensor construction or execution exists here.</p>
 *
 * <p>Coordinates and steps use {@code long} to share the numeric width of Shape dimensions and
 * layout geometry, while axes use {@code int} because Java rank and axis positions are indexed by
 * {@code int}. Values are already normalized: starts, ends, and axes are non-negative, axes are
 * unique, and steps are positive. Raw negative request syntax, rank and dimension bounds,
 * clamping, result extents, empty-result policy, and arithmetic overflow require input Shape
 * context and are deliberately deferred.</p>
 *
 * <p>The four caller-owned lists are validated in ascending entry order and copied only after all
 * validation succeeds. Each stored list is an immutable snapshot, so later caller mutation cannot
 * alter this value and accessor mutation fails. Entry order and all four list values participate
 * in record equality and hashing. Four empty lists describe a normalized identity slice that
 * constrains no axes. Starts may equal or exceed their paired ends because this record does not
 * calculate an extent.</p>
 *
 * <p>A future single-axis convenience is represented by one entry with step one, for example
 * {@code new SliceAttrs(List.of(fromInclusive), List.of(toExclusive),
 * List.of(normalizedAxis), List.of(1L))}; it does not require another kind. Generated text is
 * diagnostic only, not request syntax, serialization, compiler canonical form, ONNX mapping, or
 * backend dispatch. These attributes contain no Tensor, Shape calculation, layout or storage
 * view, provenance, materialization, gradient, compiler, backend, ONNX, or execution behavior.</p>
 *
 * @param starts the non-null ordered inclusive normalized starts; elements must be non-null and
 *     non-negative, and the stored value is an immutable snapshot
 * @param ends the non-null ordered exclusive normalized ends paired with {@code starts}; elements
 *     must be non-null and non-negative, and the stored value is an immutable snapshot
 * @param axes the non-null ordered normalized input axes paired with the bounds; elements must be
 *     non-null, non-negative, and unique, and the stored value is an immutable snapshot
 * @param steps the non-null ordered positive increments paired with the bounds and axes; elements
 *     must be non-null and positive, and the stored value is an immutable snapshot
 */
public record SliceAttrs(
        List<Long> starts,
        List<Long> ends,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs {
    /**
     * Creates immutable normalized slice parameters from four exactly paired lists.
     *
     * <p>Validation null-checks the four list references in component order, checks their sizes,
     * then inspects entries from index zero upward. At each index it null-checks start, end, axis,
     * and step before checking non-negative start, non-negative end, non-negative and unique axis,
     * and positive step. Only after every entry succeeds are the four immutable snapshots stored
     * in component order.</p>
     *
     * <p>Construction performs no raw-coordinate normalization, rank or dimension lookup,
     * start/end comparison, clamping, extent calculation, Shape derivation, or layout decision.
     * Empty lists and every non-negative start/end relationship are structurally valid.</p>
     *
     * @param starts the ordered inclusive normalized starts; must be non-null, match every other
     *     list size, and contain only non-null non-negative values
     * @param ends the ordered exclusive normalized ends; must be non-null, match every other list
     *     size, and contain only non-null non-negative values
     * @param axes the ordered normalized input axes; must be non-null, match every other list size,
     *     and contain only non-null, non-negative, unique values
     * @param steps the ordered coordinate increments; must be non-null, match every other list
     *     size, and contain only non-null positive values
     * @throws NullPointerException if a list is {@code null}, with its component name as the
     *     message, or if an element is {@code null}, with its component name and index as the
     *     message
     * @throws IllegalArgumentException if list sizes differ; if a start, end, or axis is negative;
     *     if an axis is repeated; or if a step is zero or negative
     */
    public SliceAttrs {
        Objects.requireNonNull(starts, "starts");
        Objects.requireNonNull(ends, "ends");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(steps, "steps");
        if (starts.size() != ends.size()
                || starts.size() != axes.size()
                || starts.size() != steps.size()) {
            throw new IllegalArgumentException(
                    "starts, ends, axes, and steps must have matching sizes");
        }

        HashSet<Integer> seenAxes = new HashSet<>();
        for (int index = 0; index < starts.size(); index++) {
            Long start = Objects.requireNonNull(starts.get(index), "starts[" + index + "]");
            Long end = Objects.requireNonNull(ends.get(index), "ends[" + index + "]");
            Integer axis = Objects.requireNonNull(axes.get(index), "axes[" + index + "]");
            Long step = Objects.requireNonNull(steps.get(index), "steps[" + index + "]");
            if (start < 0) {
                throw new IllegalArgumentException(
                        "starts[" + index + "] must be non-negative: " + start);
            }
            if (end < 0) {
                throw new IllegalArgumentException(
                        "ends[" + index + "] must be non-negative: " + end);
            }
            if (axis < 0) {
                throw new IllegalArgumentException(
                        "axes[" + index + "] must be non-negative: " + axis);
            }
            if (!seenAxes.add(axis)) {
                throw new IllegalArgumentException(
                        "axes contains duplicate axis " + axis + " at index " + index);
            }
            if (step <= 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be positive: " + step);
            }
        }

        starts = List.copyOf(starts);
        ends = List.copyOf(ends);
        axes = List.copyOf(axes);
        steps = List.copyOf(steps);
    }

    /**
     * Returns the immutable ordered inclusive normalized starts.
     *
     * <p>Entry {@code i} is paired exactly with {@link #ends()}, {@link #axes()}, and
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
     * Returns the immutable ordered exclusive normalized ends.
     *
     * <p>An end may be less than or equal to its paired start because Shape-aware extent and
     * empty-slice policy are deferred. The returned list is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable exclusive-end snapshot; an empty list constrains no axes
     */
    @Override
    public List<Long> ends() {
        return ends;
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
     * Returns the immutable ordered positive coordinate increments.
     *
     * <p>Entry {@code i} advances within the half-open bounds at the same index. The returned list
     * is the stored immutable snapshot.</p>
     *
     * @return the non-null immutable positive-step snapshot; an empty list constrains no axes
     */
    @Override
    public List<Long> steps() {
        return steps;
    }
}
