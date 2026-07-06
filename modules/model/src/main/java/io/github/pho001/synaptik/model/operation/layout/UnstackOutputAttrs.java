package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized source axis and logical coordinate for one indexed unstack result.
 *
 * <p>For conceptual input Shape {@code [2, 3, 4]} and source axis one, output indices zero, one,
 * and two identify three separate logical results, each with conceptual Shape {@code [2, 4]}.
 * The selected coordinate is fixed and the source axis is conceptually removed. These attributes
 * pair only with {@link TensorCompositionKind#UNSTACK}.</p>
 *
 * <p>{@link #outputIndex()} is required because a public unstack request returns several tensors
 * while each current Tensor carries one independent provenance value. The index distinguishes
 * those semantic outputs without adding producer grouping or graph output-slot state. It is not
 * a result-collection object index, graph value or node identity, output count, producer-group
 * identity, storage offset, or runtime slot.</p>
 *
 * <p>Both values are already normalized and structurally non-negative. This record has no input
 * rank or selected-axis extent, so it cannot prove that the axis exists or that the output index
 * is within the output count. Zero, positive values, and {@link Integer#MAX_VALUE} are retained
 * unchanged. Validation checks axis before output index.</p>
 *
 * <p>Record-generated equality and hashing use both components in declaration order, and
 * generated text is diagnostic only. This value contains no Tensor, input list, Shape, result
 * descriptor, layout, provenance, graph grouping, gradient, compiler, backend, ONNX, or execution
 * behavior.</p>
 *
 * @param axis the already normalized, non-negative source-axis position that is conceptually
 *     removed
 * @param outputIndex the non-negative logical coordinate fixed on the source axis for this
 *     individual result
 */
public record UnstackOutputAttrs(int axis, int outputIndex) implements OperationAttrs {
    /**
     * Creates immutable parameters identifying one UNSTACK output.
     *
     * <p>Validation checks the axis before the output index and retains both values unchanged.
     * Construction does not normalize raw syntax, inspect rank or axis extent, determine output
     * count, derive a Shape, group outputs, or create provenance or graph state.</p>
     *
     * @param axis the already normalized source-axis position; must be non-negative
     * @param outputIndex the logical coordinate fixed on that axis; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     * @throws IllegalArgumentException if {@code outputIndex} is negative after axis validation,
     *     with message {@code outputIndex must be non-negative: <outputIndex>}
     */
    public UnstackOutputAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        if (outputIndex < 0) {
            throw new IllegalArgumentException(
                    "outputIndex must be non-negative: " + outputIndex);
        }
    }

    /**
     * Returns the already normalized source-axis position removed by the unstack meaning.
     *
     * <p>The value is structurally non-negative but is not checked against an input rank by this
     * attributes record.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the logical source-axis coordinate identifying this individual unstack result.
     *
     * <p>The value is structurally non-negative but is not checked against the selected-axis
     * extent. It distinguishes public result semantics and is not a graph output slot or producer
     * grouping identity.</p>
     *
     * @return the exact non-negative output index supplied at construction
     */
    @Override
    public int outputIndex() {
        return outputIndex;
    }
}
