package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.List;
import java.util.Objects;

/**
 * Carries normalized positive complete-pattern repeat counts for per-axis tiling.
 *
 * <p>Entry {@code repeats[i]} is the positive number of times the complete input pattern is
 * repeated along axis position {@code i}. For conceptual input {@code [[1, 2], [3, 4]]} and
 * repeats {@code [2, 3]}, the complete row pattern is tiled three times along axis one and the
 * complete two-row pattern is tiled twice along axis zero. This differs from repeating each
 * scalar into an adjacent run and defines no value execution.</p>
 *
 * <p>The caller-owned list is validated in ascending index order and copied exactly once after
 * validation. The stored value is an immutable snapshot that preserves order and contents but
 * not caller list identity. An empty list describes rank-zero scalar identity parameters.
 * {@link Long#MAX_VALUE} is structurally valid because input-rank matching, result-Shape
 * multiplication, and overflow checking are deferred.</p>
 *
 * <p>These attributes contain no Tensor, result Shape or DataType, layout, storage,
 * materialization, provenance, gradient, compiler, backend, ONNX, or execution behavior.</p>
 *
 * @param repeats the non-null ordered positive complete-pattern repeat counts; elements must be
 *     non-null and positive, and the stored value is an immutable snapshot
 */
public record TileAttrs(List<Long> repeats) implements OperationAttrs {
    /**
     * Creates immutable normalized per-axis tiling parameters.
     *
     * <p>Validation first null-checks the list reference, then inspects entries from index zero
     * upward. Each element is null-checked before its positive-value check. Only after every
     * entry succeeds is one immutable snapshot stored.</p>
     *
     * <p>Construction performs no rank lookup, result-extent multiplication, overflow check,
     * Shape derivation, layout decision, or value repetition. An empty list and
     * {@link Long#MAX_VALUE} are structurally valid.</p>
     *
     * @param repeats the ordered complete-pattern repeat counts; must be non-null and contain only
     *     non-null values greater than zero
     * @throws NullPointerException if {@code repeats} is {@code null}, with message
     *     {@code repeats}, or if element {@code i} is {@code null}, with the exact indexed message
     *     {@code repeats[i]}
     * @throws IllegalArgumentException if an element is zero or negative, with the exact indexed
     *     message {@code repeats[i] must be positive: value}
     */
    public TileAttrs {
        Objects.requireNonNull(repeats, "repeats");
        for (int index = 0; index < repeats.size(); index++) {
            Long repeat = Objects.requireNonNull(repeats.get(index), "repeats[" + index + "]");
            if (repeat <= 0) {
                throw new IllegalArgumentException(
                        "repeats[" + index + "] must be positive: " + repeat);
            }
        }
        repeats = List.copyOf(repeats);
    }

    /**
     * Returns the immutable ordered complete-pattern repeat counts.
     *
     * <p>Entry {@code i} applies to axis position {@code i}. The returned list is the stored
     * immutable snapshot; no identity relationship with the caller's original list is promised.</p>
     *
     * @return the non-null immutable positive-repeat snapshot; an empty list denotes scalar
     *     identity
     */
    @Override
    public List<Long> repeats() {
        return repeats;
    }
}
