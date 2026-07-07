package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized shared-batch count and explicit reduction for
 * {@link ScatterNdKind#SCATTER_ND}.
 *
 * <p>Scatter-ND has ordered logical inputs {@code [data, indices, updates]}. The non-negative
 * {@link #batchDimensions()} value {@code B} means that axes {@code [0, B)} of data and indices
 * form a shared batch prefix. The final indices Dimension has extent {@code K}, the coordinate-
 * tuple depth. Each tuple indexes data axes {@code [B, B + K)}, while data axes after those
 * indexed axes form the untouched suffix represented by each update slice.</p>
 *
 * <p>For data rank {@code R} and indices rank {@code Q}, updates must conceptually satisfy:</p>
 *
 * <pre>{@code
 * updates.shape == indices.shape[0:Q-1] + data.shape[B+K:R]
 * }</pre>
 *
 * <p>The first term is equivalently {@code indices.shape[:-1]}. Data {@code [2, 3, 4]} with
 * indices {@code [5, 2]}, {@code B=0}, and {@code K=2} requires updates {@code [5, 4]}. The same
 * data with indices {@code [2, 5, 1]}, {@code B=1}, and {@code K=1} requires updates
 * {@code [2, 5, 4]}. Data {@code [2, 3]} with indices {@code [2]}, {@code B=0}, and {@code K=2}
 * requires canonical rank-zero scalar updates {@code []}. Each conceptual result has exactly the
 * corresponding data Shape.</p>
 *
 * <p>The non-null {@link #reduction()} value selects replacement, addition, multiplication,
 * maximum, or minimum. Replacement through {@link ScatterReduction#NONE} requires unique target
 * tuples. A target is the result coordinate or suffix slice addressed by one tuple; duplicate
 * target tuples are multiple tuples that address that same target. Such duplicates are invalid
 * for replacement, but detecting them requires index values and is not a constructor check.
 * {@code null} never means {@code NONE}.</p>
 *
 * <p>This value stores no operand, rank, Shape, tuple depth, data type, descriptor, result, or
 * provenance. It therefore cannot prove that {@code B} fits particular input ranks, that shared
 * batch Dimensions match, that {@code K} is valid, or that the updates formula holds. Tuple depth
 * remains occurrence-specific final-indices-Dimension data rather than duplicated attribute
 * state. Zero and {@link Integer#MAX_VALUE} are structurally valid batch counts because no ranks
 * are present. Task 0018J owns those future public input-aware rank, Shape, type, and result checks
 * because they depend on concrete operands and those facts are not stored here.</p>
 *
 * <p>The exact semantic composition is an {@code Operation} whose kind is
 * {@link ScatterNdKind#SCATTER_ND} and whose attributes are this value. Generic operation
 * composition retains both values but does not enforce that pairing or the ordered three-input
 * context. The immutable record retains both components unchanged. Record-generated equality and
 * hashing use both components, and generated text is diagnostic rather than a serialization,
 * parsing, compiler-dispatch, backend, or execution contract.</p>
 *
 * <p>These attributes define no value access, bounds or duplicate checking, mutation, numeric
 * order, gradients, graph or compiler behavior, backend support, or execution.</p>
 *
 * @param batchDimensions the already normalized, non-negative number of shared leading batch
 *     Dimensions
 * @param reduction the non-null replacement or reduction meaning applied at addressed targets
 */
public record ScatterNdAttrs(int batchDimensions, ScatterReduction reduction)
        implements OperationAttrs {
    /**
     * Creates immutable normalized batch and reduction parameters for Scatter-ND.
     *
     * <p>The batch count is validated before the reduction reference. Both valid values are then
     * retained unchanged. Construction does not inspect operands, validate ranks, Shapes, shared
     * batch Dimensions, tuple depth, data types, bounds, duplicate targets, or construct a
     * result.</p>
     *
     * @param batchDimensions the already normalized number of shared leading batch Dimensions;
     *     must be non-negative
     * @param reduction the explicit replacement or reduction meaning; must not be {@code null}
     * @throws IllegalArgumentException if {@code batchDimensions} is negative, with message
     *     {@code batchDimensions must be non-negative: <batchDimensions>}
     * @throws NullPointerException if {@code reduction} is {@code null}, with message
     *     {@code reduction}
     */
    public ScatterNdAttrs {
        if (batchDimensions < 0) {
            throw new IllegalArgumentException(
                    "batchDimensions must be non-negative: " + batchDimensions);
        }
        Objects.requireNonNull(reduction, "reduction");
    }

    /**
     * Returns the already normalized number of shared leading data and indices batch Dimensions.
     *
     * @return the exact non-negative batch-dimension count supplied at construction; it has not
     *     been validated against operand ranks, their batch prefix, or tuple depth
     */
    @Override
    public int batchDimensions() {
        return batchDimensions;
    }

    /**
     * Returns the selected replacement or reduction meaning.
     *
     * @return the exact non-null {@link ScatterReduction} supplied at construction
     */
    @Override
    public ScatterReduction reduction() {
        return reduction;
    }
}
