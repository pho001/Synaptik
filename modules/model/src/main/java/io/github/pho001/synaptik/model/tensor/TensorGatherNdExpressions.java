package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free Gather-ND expressions for {@link Tensor}.
 *
 * <p>Each operation consumes ordered {@code [data, indices]} inputs. Indices must be exact
 * {@link DataType#INT32} or {@link DataType#INT64}, have rank at least one, and end in a static
 * positive tuple-depth Dimension. A non-negative batch count must fit both ranks, and corresponding
 * leading batch Dimensions must be structurally equal. The result Shape contains the indices
 * prefix without tuple depth followed by the unindexed data suffix; two empty parts produce the
 * canonical scalar Shape.</p>
 *
 * <p>Every fresh result preserves data type and gradient eligibility, leaves layout unresolved,
 * and records exact {@link GatherNdKind#GATHER_ND} semantics and provenance without a label or
 * storage. This field-free helper never reads index values, checks bounds, defines gradient or
 * scatter behavior, captures a graph, chooses materialization or backend support, or executes
 * work.</p>
 */
final class TensorGatherNdExpressions {
    /** Prevents instantiation because Gather-ND expression construction owns no state. */
    private TensorGatherNdExpressions() {
    }

    /**
     * Delegates a zero-batch request to the explicit shared validation and construction path.
     *
     * @param data data tensor passed unchanged to the explicit path
     * @param indices indices tensor passed unchanged to the explicit path
     * @return the non-null fresh Gather-ND tensor created with zero batch Dimensions
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type, rank, batch fit, or tuple depth is invalid
     * @throws ArithmeticException if checked result-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor gatherNd(Tensor data, Tensor indices) {
        return gatherNd(data, indices, 0);
    }

    /**
     * Validates one concrete data/indices pair and creates its Gather-ND expression metadata.
     *
     * <p>Validation checks nulls, index type, indices rank, normalized batch count, batch-rank fit,
     * batch-prefix equality, and static positive tuple depth in that order before result identity
     * allocation. For ranks {@code R} and {@code Q}, batch count {@code B}, and tuple depth
     * {@code K}, the result is {@code indices[0:Q-1] + data[B+K:R]}.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one;
     *     values and bounds are never inspected
     * @param batchDimensions non-negative shared leading batch count smaller than both input ranks
     * @return a non-null fresh Gather-ND tensor with derived Shape and unresolved layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type, rank, batch prefix, or tuple depth is invalid
     * @throws ArithmeticException if checked result-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor gatherNd(Tensor data, Tensor indices, int batchDimensions) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        validateIndexType(indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        Shape indicesShape = indicesDescriptor.shape();
        if (indicesShape.rank() == 0) {
            throw new IllegalArgumentException("gatherNd indices rank must be at least 1");
        }
        GatherNdAttrs attrs = new GatherNdAttrs(batchDimensions);
        validateBatchDimensions(dataShape, indicesShape, batchDimensions);
        validateBatchPrefix(dataShape, indicesShape, batchDimensions);
        int tupleDepth = tupleDepth(dataShape, indicesShape, batchDimensions);
        Shape resultShape = resultShape(dataShape, indicesShape, batchDimensions, tupleDepth);
        return create(data, indices, dataDescriptor, resultShape, attrs);
    }

    /**
     * Requires the exact supported signed-integral index representation.
     *
     * @param indicesDescriptor non-null exact descriptor supplying the index data type
     * @throws IllegalArgumentException if the type is not INT32 or INT64
     */
    private static void validateIndexType(TensorDescriptor indicesDescriptor) {
        DataType dataType = indicesDescriptor.dataType();
        if (dataType != DataType.INT32 && dataType != DataType.INT64) {
            throw new IllegalArgumentException(
                    "gatherNd indices data type must be INT32 or INT64: " + dataType);
        }
    }

    /**
     * Requires the normalized batch count to be smaller than indices rank, then data rank.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null indices Shape of rank at least one
     * @param batchDimensions already normalized non-negative shared leading batch count
     * @throws IllegalArgumentException if the count reaches either rank, with indices checked first
     */
    private static void validateBatchDimensions(
            Shape dataShape, Shape indicesShape, int batchDimensions) {
        if (batchDimensions >= indicesShape.rank()) {
            throw new IllegalArgumentException(
                    "gatherNd batchDimensions must be less than indices rank: batchDimensions="
                            + batchDimensions + ", indicesRank=" + indicesShape.rank());
        }
        if (batchDimensions >= dataShape.rank()) {
            throw new IllegalArgumentException(
                    "gatherNd batchDimensions must be less than data rank: batchDimensions="
                            + batchDimensions + ", dataRank=" + dataShape.rank());
        }
    }

    /**
     * Requires exact structural equality for each shared leading batch Dimension.
     *
     * @param dataShape non-null data Shape containing at least {@code batchDimensions} axes
     * @param indicesShape non-null indices Shape containing at least {@code batchDimensions} axes
     * @param batchDimensions validated number of leading Dimensions to compare
     * @throws IllegalArgumentException on the first increasing-axis structural mismatch
     */
    private static void validateBatchPrefix(
            Shape dataShape, Shape indicesShape, int batchDimensions) {
        for (int axis = 0; axis < batchDimensions; axis++) {
            Dimension expected = dataShape.dimensions().get(axis);
            Dimension actual = indicesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "gatherNd batch dimension at axis " + axis
                                + " must match data: expected=" + expected + ", actual=" + actual);
            }
        }
    }

    /**
     * Reads and validates the occurrence-specific tuple depth from the final indices Dimension.
     *
     * @param dataShape non-null data Shape whose rank bounds the indexed suffix
     * @param indicesShape non-null rank-one-or-greater indices Shape
     * @param batchDimensions validated shared leading batch count
     * @return positive tuple depth no greater than data rank minus batch count
     * @throws IllegalArgumentException if tuple depth is dynamic, zero, or too large
     */
    private static int tupleDepth(
            Shape dataShape, Shape indicesShape, int batchDimensions) {
        Dimension tupleDepthDimension =
                indicesShape.dimensions().get(indicesShape.rank() - 1);
        if (tupleDepthDimension.isDynamic()) {
            throw new IllegalArgumentException("gatherNd tuple depth must be statically known");
        }
        long depth = ((StaticDimension) tupleDepthDimension).size();
        int maximum = dataShape.rank() - batchDimensions;
        if (depth < 1 || depth > maximum) {
            throw new IllegalArgumentException(
                    "gatherNd tuple depth must be in [1, data rank - batchDimensions]: depth="
                            + depth + ", maximum=" + maximum);
        }
        return (int) depth;
    }

    /**
     * Concatenates the indices prefix and untouched data suffix using exact Dimension references.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null indices Shape whose final Dimension is omitted
     * @param batchDimensions validated shared leading batch count
     * @param tupleDepth validated positive number of indexed data axes after the batch prefix
     * @return a non-null derived Shape; empty prefix and suffix produce canonical scalar Shape
     * @throws ArithmeticException if checked result-rank addition overflows
     */
    private static Shape resultShape(
            Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth) {
        int indicesPrefixRank = indicesShape.rank() - 1;
        int dataSuffixStart = batchDimensions + tupleDepth;
        int dataSuffixRank = dataShape.rank() - dataSuffixStart;
        int resultRank = Math.addExact(indicesPrefixRank, dataSuffixRank);
        Dimension[] resultDimensions = new Dimension[resultRank];
        int resultAxis = 0;
        for (int indicesAxis = 0; indicesAxis < indicesPrefixRank; indicesAxis++) {
            resultDimensions[resultAxis++] = indicesShape.dimensions().get(indicesAxis);
        }
        for (int dataAxis = dataSuffixStart; dataAxis < dataShape.rank(); dataAxis++) {
            resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Creates exact descriptor, semantics, ordered provenance, and one fresh derived Tensor.
     *
     * @param data non-null exact provenance input zero
     * @param indices non-null exact provenance input one
     * @param dataDescriptor non-null exact descriptor supplying retained type and eligibility
     * @param resultShape non-null already derived result Shape
     * @param attrs non-null normalized batch attributes retained by the operation
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalArgumentException if descriptor invariants reject the supplied metadata
     * @throws IllegalStateException if tensor identifier space is exhausted at final delegation
     */
    private static Tensor create(
            Tensor data,
            Tensor indices,
            TensorDescriptor dataDescriptor,
            Shape resultShape,
            GatherNdAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataDescriptor.dataType(), resultShape, Optional.empty(), dataDescriptor.requiresGrad());
        Operation operation = new Operation(GatherNdKind.GATHER_ND, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(data, indices));
    }
}
