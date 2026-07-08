package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies the backend-independent meaning of indexing tensor data with tuples of coordinates.
 *
 * <p>{@link #GATHER_ND} has ordered logical inputs {@code [data, indices]}: {@code data} supplies
 * the values being indexed, and {@code indices} supplies coordinate tuples. If the data rank is
 * {@code R}, the indices rank is {@code Q}, and {@link GatherNdAttrs#batchDimensions()} is
 * {@code B}, the leading {@code B} Dimensions of both inputs form a shared batch prefix. The
 * final indices Dimension has extent {@code K} and is the tuple depth. Each tuple indexes the
 * {@code K} data axes {@code [B, B + K)}; the data axes {@code [B + K, R)} remain as the
 * untouched suffix of each selected value.</p>
 *
 * <p>The conceptual result Shape is
 * {@code indices.shape[0:Q-1] + data.shape[B+K:R]}. For example:</p>
 *
 * <ul>
 *   <li>data {@code [2, 3, 4]}, indices {@code [5, 2]}, {@code B=0}, and {@code K=2} give result
 *       {@code [5, 4]};</li>
 *   <li>data {@code [2, 3, 4]}, indices {@code [2, 5, 1]}, {@code B=1}, and {@code K=1} give
 *       result {@code [2, 5, 4]}; and</li>
 *   <li>data {@code [2, 3]}, indices {@code [2]}, {@code B=0}, and {@code K=2} give the canonical
 *       rank-zero scalar result {@code []}, not {@code [1]}.</li>
 * </ul>
 *
 * <p>The kind pairs explicitly with {@link GatherNdAttrs}. The final indices Dimension supplies
 * tuple depth for each operation occurrence, so that input-specific value is not duplicated in
 * the attributes. The public input-aware Tensor expression boundary validates ranks, the shared
 * batch prefix, tuple depth, index data type, and the result Shape. This enum itself performs no
 * input inspection, validation, result construction, bounds checking, or execution.</p>
 *
 * <p>Gather-ND differs from scalar {@link SelectKind#SELECT}, whose one intrinsic coordinate
 * removes one axis, and from {@link AxisGatherKind}, whose index values address one selected data
 * axis. It also differs from scatter-ND, which writes or combines updates rather than reading
 * selected data. This enum stores no inputs, Shape, data type, result, provenance, gradient,
 * graph or compiler policy, backend support, or execution state. Its inherited enum name is
 * diagnostic text rather than a serialization, dispatch, registry, route, or kernel identifier.</p>
 */
public enum GatherNdKind implements OperationKind {
    /**
     * Reads values or suffix slices selected by coordinate tuples after a shared batch prefix.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}, tuple depth comes from the final
     * indices Dimension, and the normalized batch count comes from {@link GatherNdAttrs}. The
     * public zero-batch convenience uses {@code new GatherNdAttrs(0)} rather than another
     * operation kind. This constant does not validate inputs, calculate a result Shape, inspect
     * index values, or execute work.</p>
     */
    GATHER_ND;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(GatherNdAttrs.class, 2, 1));

    /**
     * Returns the tuple-index two-input, one-output structural signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
