package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized number of shared leading batch Dimensions for one
 * {@link GatherNdKind#GATHER_ND} operation.
 *
 * <p>Gather-ND has ordered logical inputs {@code [data, indices]}. The non-negative
 * {@link #batchDimensions()} value {@code B} means that input axes {@code [0, B)} form a shared
 * batch prefix. The final indices Dimension has extent {@code K}, which is the coordinate-tuple
 * depth; each tuple indexes data axes {@code [B, B + K)}. Data axes after those indexed axes are
 * the untouched suffix included in every selected result.</p>
 *
 * <p>For data rank {@code R} and indices rank {@code Q}, the conceptual result Shape is
 * {@code indices.shape[0:Q-1] + data.shape[B+K:R]}. Data {@code [2, 3, 4]} with indices
 * {@code [5, 2]}, {@code B=0}, and {@code K=2} gives {@code [5, 4]}. Data {@code [2, 3, 4]} with
 * indices {@code [2, 5, 1]}, {@code B=1}, and {@code K=1} gives {@code [2, 5, 4]}. Data
 * {@code [2, 3]} with indices {@code [2]}, {@code B=0}, and {@code K=2} gives the canonical
 * rank-zero scalar Shape {@code []}, not {@code [1]}.</p>
 *
 * <p>This value stores neither input nor its rank, Shape, or final indices Dimension. It therefore
 * cannot verify that {@code B < Q}, that the batch Dimensions match, or that
 * {@code 1 <= K <= R - B}. Tuple depth remains occurrence-specific input Shape data rather than
 * duplicated attribute state. A later input-aware expression boundary owns those checks, index
 * data-type validation, and result construction. Zero and {@link Integer#MAX_VALUE} are
 * structurally valid here because no input ranks are present.</p>
 *
 * <p>The exact semantic composition is an {@code Operation} whose kind is
 * {@link GatherNdKind#GATHER_ND} and whose attributes are this value. Generic operation
 * composition retains both values but does not enforce their compatibility. The immutable record
 * retains its primitive component unchanged. Record-generated equality and hashing use that
 * component, and generated text is diagnostic rather than a serialization, request, parsing,
 * compiler-dispatch, or backend contract. Future zero-batch convenience uses
 * {@code new GatherNdAttrs(0)} rather than a default constructor, singleton, or separate kind.</p>
 *
 * <p>These attributes define no Tensor construction, descriptor, provenance, storage, value
 * access, index bounds, gradient, graph or compiler behavior, backend support, or execution.
 * Gather-ND remains distinct from scalar select, one-axis gather, and scatter-ND.</p>
 *
 * @param batchDimensions the already normalized, non-negative number of shared leading batch
 *     Dimensions
 */
public record GatherNdAttrs(int batchDimensions) implements OperationAttrs {
    /**
     * Creates immutable normalized batch parameters for Gather-ND.
     *
     * <p>The primitive value is retained unchanged after the non-negative check. Construction does
     * not inspect data or indices, validate ranks, shared batch Dimensions, tuple depth, index data
     * type, bounds, or derive a result Shape.</p>
     *
     * @param batchDimensions the already normalized number of shared leading batch Dimensions;
     *     must be non-negative
     * @throws IllegalArgumentException if {@code batchDimensions} is negative, with message
     *     {@code batchDimensions must be non-negative: <batchDimensions>}
     */
    public GatherNdAttrs {
        if (batchDimensions < 0) {
            throw new IllegalArgumentException(
                    "batchDimensions must be non-negative: " + batchDimensions);
        }
    }

    /**
     * Returns the already normalized number of shared leading data and indices batch Dimensions.
     *
     * <p>The result is structurally non-negative but has not been validated against input ranks,
     * batch-prefix compatibility, or tuple depth by this attributes value.</p>
     *
     * @return the exact non-negative batch-dimension count supplied at construction
     */
    @Override
    public int batchDimensions() {
        return batchDimensions;
    }
}
