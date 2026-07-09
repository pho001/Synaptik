package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies the backend-independent meanings for indexing tensor data along one axis with
 * a tensor of indices.
 *
 * <p>Every kind has the ordered logical inputs {@code [data, indices]}: {@code data} supplies the
 * values being indexed, and {@code indices} supplies logical coordinates on the normalized axis
 * carried by {@link IndexAxisAttrs}. The shape relationships below explain each meaning; neither
 * this enum nor the attributes inspect inputs, validate shapes, or construct a result.</p>
 *
 * <ul>
 *   <li>{@link #GATHER} replaces the selected data axis with the complete indices shape. For
 *       data shape {@code [2, 3, 4]}, axis {@code 1}, and indices shape {@code [5, 6]}, the
 *       conceptual result shape is {@code [2, 5, 6, 4]}.</li>
 *   <li>{@link #GATHER_ELEMENTS} aligns same-rank indices with data coordinates away from the
 *       selected axis and has the exact indices shape as its conceptual result. For data shape
 *       {@code [2, 3, 4]}, axis {@code 1}, and indices shape {@code [2, 7, 4]}, the conceptual
 *       result shape is {@code [2, 7, 4]}, subject to public non-axis compatibility checks.</li>
 * </ul>
 *
 * <p>Both kinds pair explicitly with {@link IndexAxisAttrs}; their family-owned signatures
 * enforce that exact pairing and declare the ordered two-input, one-output occurrence. Axis
 * gather is
 * distinct from scalar {@link SelectKind#SELECT}, whose coordinate is an intrinsic scalar and
 * removes an axis, and from gather-ND, which uses multi-axis index tuples. It is also distinct
 * from functional scatter, which writes or combines updates rather than reading indexed data.</p>
 *
 * <p>This enum stores no input, shape, data type, bounds, result descriptor, provenance, gradient,
 * graph or compiler policy, backend support, or execution state. The public Tensor-expression
 * contract validates that indices use {@code INT32} or {@code INT64}, normalizes a caller-facing
 * axis, checks input-dependent shape relationships, and constructs result metadata and
 * provenance. It reads no index values and therefore performs no index-value bounds check. The
 * inherited enum name is diagnostic text rather than a serialization, dispatch, registry, route,
 * or kernel identifier.</p>
 */
public enum AxisGatherKind implements OperationKind {
    /**
     * Replaces the selected data axis with the complete indices shape in canonical gather
     * semantics.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}. For data {@code [2, 3, 4]}, axis
     * {@code 1}, and indices {@code [5, 6]}, the conceptual result is {@code [2, 5, 6, 4]}.
     * This enum performs no shape validation or result construction.</p>
     */
    GATHER,

    /**
     * Reads indices aligned with same-rank data coordinates away from the selected axis and
     * produces the exact indices shape conceptually.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}. For data {@code [2, 3, 4]}, axis
     * {@code 1}, and indices {@code [2, 7, 4]}, the conceptual result is {@code [2, 7, 4]}, subject
     * to public non-axis compatibility checks. This kind performs neither those checks nor value
     * selection.</p>
     */
    GATHER_ELEMENTS;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(IndexAxisAttrs.class, 2, 1));

    /**
     * Returns the shared two-input, one-output axis-indexing signature.
     *
     * @return the stable immutable singleton signature list accepting {@link IndexAxisAttrs}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
