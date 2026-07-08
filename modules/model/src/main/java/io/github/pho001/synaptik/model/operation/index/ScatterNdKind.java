package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies the backend-independent meaning of functionally scattering updates through tuples
 * of coordinates.
 *
 * <p>{@link #SCATTER_ND} has ordered logical inputs {@code [data, indices, updates]}.
 * {@code data} supplies the base values and the exact result Shape, {@code indices} supplies
 * coordinate tuples, and {@code updates} supplies the values applied at the selected targets. The
 * conceptual result starts from {@code data}, applies the {@link ScatterReduction} selected by
 * {@link ScatterNdAttrs#reduction()}, and is a new value with exactly the data Shape; it does not
 * mutate {@code data} in place. A target is the result coordinate or suffix slice addressed by
 * one tuple. Duplicate target tuples are multiple tuples that address that same target.</p>
 *
 * <p>If the data rank is {@code R}, the indices rank is {@code Q}, and
 * {@link ScatterNdAttrs#batchDimensions()} is {@code B}, the leading {@code B} Dimensions of data
 * and indices form a shared batch prefix. The final indices Dimension has extent {@code K} and is
 * the tuple depth. Each tuple indexes data axes {@code [B, B + K)}. Data axes
 * {@code [B + K, R)} form the untouched suffix applied as one update slice. The required updates
 * Shape is therefore:</p>
 *
 * <pre>{@code
 * indices.shape[0:Q-1] + data.shape[B+K:R]
 * }</pre>
 *
 * <p>Equivalently, {@code updates.shape == indices.shape[:-1] +
 * data.shape[batchDimensions + K:]}. For example:</p>
 *
 * <ul>
 *   <li>data {@code [2, 3, 4]}, indices {@code [5, 2]}, {@code B=0}, and {@code K=2} require
 *       updates {@code [5, 4]} and produce result {@code [2, 3, 4]};</li>
 *   <li>data {@code [2, 3, 4]}, indices {@code [2, 5, 1]}, {@code B=1}, and {@code K=1} require
 *       updates {@code [2, 5, 4]} and produce result {@code [2, 3, 4]}; and</li>
 *   <li>data {@code [2, 3]}, indices {@code [2]}, {@code B=0}, and {@code K=2} require canonical
 *       rank-zero scalar updates {@code []} and produce result {@code [2, 3]}.</li>
 * </ul>
 *
 * <p>{@link ScatterReduction#NONE} replaces a base value and requires unique target tuples;
 * duplicate targets are invalid rather than resolved by update order. The other reduction values
 * combine a base value and all updates for a target by addition, multiplication, maximum, or
 * minimum. These are mathematical meanings, not numerical algorithms or execution-order
 * promises.</p>
 *
 * <p>The kind pairs explicitly with {@link ScatterNdAttrs}. Tuple depth remains the final indices
 * Dimension because it is specific to one operation occurrence and is not intrinsic attribute
 * state. The public {@code Tensor.scatterNd} overloads now own input-aware rank, shared-batch-
 * prefix, tuple-depth, Shape, data-type, result, and provenance validation because those facts
 * depend on concrete operands and are not stored here. This enum performs none of those checks
 * and does not read values, check bounds or duplicates, define gradients, capture a graph, select
 * backend support, or execute work.</p>
 *
 * <p>Scatter-ND differs from {@link GatherNdKind#GATHER_ND Gather-ND}, which reads selected data,
 * and from {@link AxisScatterKind axis scatter}, whose indices address one selected axis rather
 * than multi-axis coordinate tuples. Its inherited enum name is diagnostic text rather than a
 * serialization, dispatch, registry, route, or kernel identifier.</p>
 */
public enum ScatterNdKind implements OperationKind {
    /**
     * Functionally applies tuple-indexed updates to a data-shaped result.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; tuple depth comes from
     * the final indices Dimension, while normalized batch count and explicit reduction come from
     * {@link ScatterNdAttrs}. This constant stores no operands or occurrence-specific Shape facts
     * and performs no input validation, duplicate detection, mutation, reduction, or execution.</p>
     */
    SCATTER_ND;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(ScatterNdAttrs.class, 3, 1));

    /**
     * Returns the tuple-index three-input, one-output structural signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
