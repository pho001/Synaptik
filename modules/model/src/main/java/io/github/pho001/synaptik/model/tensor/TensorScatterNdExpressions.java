package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free functional Scatter-ND expressions for {@link Tensor}.
 *
 * <p>Each operation consumes ordered {@code [data, indices, updates]} inputs. Indices contain
 * coordinate tuples and must use exact {@link DataType#INT32} or {@link DataType#INT64}. A shared
 * batch prefix is the structurally equal leading Dimensions of data and indices. The final
 * indices Dimension is the statically known positive tuple depth, and updates must have the
 * indices prefix without tuple depth followed by the unindexed data suffix. Each tuple position
 * contributes that complete suffix slice scalar by scalar. A non-replacement reduction includes
 * each target's base exactly once and every addressed scalar exactly once, including distinct
 * contributions from duplicate tuples; an unaddressed coordinate preserves the exact base
 * representation.</p>
 *
 * <p>Every fresh result retains exact data Shape/type, combines data/update gradient eligibility,
 * leaves layout unresolved, and records exact {@link ScatterNdKind#SCATTER_ND} semantics and
 * ordered provenance without a label or storage. This field-free helper never reads values,
 * checks bounds or duplicate targets, applies writes or reductions, mutates an input, defines
 * derivatives or subgradients, captures a graph, selects a numerical algorithm or backend, or
 * executes work. {@link ScatterReduction} fixes the portable represented-value target
 * independently of encounter, layout, stride, atomic, tree, or backend order.</p>
 */
final class TensorScatterNdExpressions {
    /** Prevents instantiation because Scatter-ND expression construction owns no state. */
    private TensorScatterNdExpressions() {
    }

    /**
     * Delegates replacement with no shared batch prefix to the explicit validation path.
     *
     * @param data data tensor passed unchanged to the explicit path
     * @param indices indices tensor passed unchanged to the explicit path
     * @param updates updates tensor passed unchanged to the explicit path
     * @return the non-null fresh Scatter-ND tensor created with NONE and zero batch Dimensions
     * @throws NullPointerException if an input is null, checked in declaration order
     * @throws IllegalArgumentException if type, rank, batch, tuple, or Shape validation fails
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates) {
        return scatterNd(data, indices, updates, ScatterReduction.NONE, 0);
    }

    /**
     * Delegates an explicit reduction with no shared batch prefix to the complete path.
     *
     * @param data data tensor passed unchanged to the explicit path
     * @param indices indices tensor passed unchanged to the explicit path
     * @param updates updates tensor passed unchanged to the explicit path
     * @param reduction reduction passed unchanged to the explicit path
     * @return the non-null fresh Scatter-ND tensor created with zero batch Dimensions
     * @throws NullPointerException if an input or reduction is null, checked in declaration order
     * @throws IllegalArgumentException if type, reduction, rank, tuple, or Shape validation fails
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor scatterNd(
            Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction) {
        return scatterNd(data, indices, updates, reduction, 0);
    }

    /**
     * Validates one concrete data/indices/updates tuple and constructs exact Scatter-ND metadata.
     *
     * <p>Validation checks nulls, descriptors, index type, matching update type, reduction type,
     * indices rank, attributes, batch fit, batch prefix, tuple depth, and expected updates Shape
     * in that order before result identity allocation. For ranks {@code R}/{@code Q}, batch count
     * {@code B}, and tuple depth {@code K}, updates must equal
     * {@code indices[0:Q-1] + data[B+K:R]}. Each tuple contributes its complete suffix slice
     * scalar by scalar. For a non-replacement reduction, every target includes its base exactly
     * once and each addressed scalar exactly once; duplicate tuples remain distinct
     * contributions, while unaddressed coordinates retain the exact base representation.
     * {@link ScatterReduction} defines the floating and integral represented-value target.
     * Construction reads no values.</p>
     *
     * @param data non-null base tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 tuple tensor retained as provenance input one
     * @param updates non-null exact-data-type tensor retained as provenance input two
     * @param reduction non-null replacement or arithmetic reduction retained exactly
     * @param batchDimensions non-negative shared leading batch count smaller than both ranks
     * @return a non-null fresh data-shaped Scatter-ND tensor with unresolved layout
     * @throws NullPointerException if an input or reduction is null, checked in declaration order
     * @throws IllegalArgumentException if type, reduction, rank, batch, tuple, or Shape validation
     *     fails
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor scatterNd(
            Tensor data,
            Tensor indices,
            Tensor updates,
            ScatterReduction reduction,
            int batchDimensions) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        Objects.requireNonNull(updates, "updates");
        Objects.requireNonNull(reduction, "reduction");
        TensorDescriptor dataDescriptor = data.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        TensorDescriptor updatesDescriptor = updates.descriptor();
        validateIndexType(indicesDescriptor);
        validateMatchingDataType(dataDescriptor, updatesDescriptor);
        validateReductionDataType(dataDescriptor, reduction);
        Shape dataShape = dataDescriptor.shape();
        Shape indicesShape = indicesDescriptor.shape();
        Shape updatesShape = updatesDescriptor.shape();
        if (indicesShape.rank() == 0) {
            throw new IllegalArgumentException("scatterNd indices rank must be at least 1");
        }
        ScatterNdAttrs attrs = new ScatterNdAttrs(batchDimensions, reduction);
        validateBatchDimensions(dataShape, indicesShape, batchDimensions);
        validateBatchPrefix(dataShape, indicesShape, batchDimensions);
        int tupleDepth = tupleDepth(dataShape, indicesShape, batchDimensions);
        Shape expectedUpdatesShape =
                expectedUpdatesShape(dataShape, indicesShape, batchDimensions, tupleDepth);
        if (!updatesShape.equals(expectedUpdatesShape)) {
            throw new IllegalArgumentException(
                    "scatterNd updates shape must equal indices prefix plus data suffix: expected="
                            + expectedUpdatesShape + ", actual=" + updatesShape);
        }
        return create(data, indices, updates, dataDescriptor, updatesDescriptor, attrs);
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
                    "scatterNd indices data type must be INT32 or INT64: " + dataType);
        }
    }

    /**
     * Requires updates to use the exact data element type without promotion or conversion.
     *
     * @param dataDescriptor non-null descriptor supplying the expected data type
     * @param updatesDescriptor non-null descriptor supplying the actual updates type
     * @throws IllegalArgumentException if the data types differ
     */
    private static void validateMatchingDataType(
            TensorDescriptor dataDescriptor, TensorDescriptor updatesDescriptor) {
        DataType expected = dataDescriptor.dataType();
        DataType actual = updatesDescriptor.dataType();
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "scatterNd updates data type must match data: expected="
                            + expected + ", actual=" + actual);
        }
    }

    /**
     * Requires BOOL data to use replacement rather than an arithmetic reduction.
     *
     * @param dataDescriptor non-null descriptor supplying the matched data/update type
     * @param reduction non-null explicit reduction already validated by the complete path
     * @throws IllegalArgumentException if BOOL is paired with a non-NONE reduction
     */
    private static void validateReductionDataType(
            TensorDescriptor dataDescriptor, ScatterReduction reduction) {
        if (dataDescriptor.dataType() == DataType.BOOL && reduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException(
                    "scatterNd BOOL data supports only NONE reduction: " + reduction);
        }
    }

    /**
     * Requires the normalized batch count to be smaller than indices rank, then data rank.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null indices Shape of rank at least one
     * @param batchDimensions already normalized non-negative shared leading batch count
     * @throws IllegalArgumentException if the count reaches either rank, indices checked first
     */
    private static void validateBatchDimensions(
            Shape dataShape, Shape indicesShape, int batchDimensions) {
        if (batchDimensions >= indicesShape.rank()) {
            throw new IllegalArgumentException(
                    "scatterNd batchDimensions must be less than indices rank: batchDimensions="
                            + batchDimensions + ", indicesRank=" + indicesShape.rank());
        }
        if (batchDimensions >= dataShape.rank()) {
            throw new IllegalArgumentException(
                    "scatterNd batchDimensions must be less than data rank: batchDimensions="
                            + batchDimensions + ", dataRank=" + dataShape.rank());
        }
    }

    /**
     * Requires exact structural equality for each shared leading batch Dimension.
     *
     * @param dataShape non-null data Shape containing the batch prefix
     * @param indicesShape non-null indices Shape containing the batch prefix
     * @param batchDimensions validated number of leading Dimensions to compare
     * @throws IllegalArgumentException on the first increasing-axis mismatch
     */
    private static void validateBatchPrefix(
            Shape dataShape, Shape indicesShape, int batchDimensions) {
        for (int axis = 0; axis < batchDimensions; axis++) {
            Dimension expected = dataShape.dimensions().get(axis);
            Dimension actual = indicesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "scatterNd batch dimension at axis " + axis
                                + " must match data: expected=" + expected + ", actual=" + actual);
            }
        }
    }

    /**
     * Reads and validates tuple depth from the final indices Dimension exactly once.
     *
     * @param dataShape non-null data Shape whose remaining rank bounds tuple depth
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
            throw new IllegalArgumentException("scatterNd tuple depth must be statically known");
        }
        long depth = ((StaticDimension) tupleDepthDimension).size();
        int maximum = dataShape.rank() - batchDimensions;
        if (depth < 1 || depth > maximum) {
            throw new IllegalArgumentException(
                    "scatterNd tuple depth must be in [1, data rank - batchDimensions]: depth="
                            + depth + ", maximum=" + maximum);
        }
        return (int) depth;
    }

    /**
     * Concatenates the indices prefix and untouched data suffix with exact Dimension references.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null indices Shape whose final tuple-depth Dimension is omitted
     * @param batchDimensions validated shared leading batch count
     * @param tupleDepth validated positive number of indexed data axes after the batch prefix
     * @return a non-null expected updates Shape; two empty parts yield canonical scalar Shape
     * @throws ArithmeticException if checked expected-rank addition overflows
     */
    private static Shape expectedUpdatesShape(
            Shape dataShape, Shape indicesShape, int batchDimensions, int tupleDepth) {
        int indicesPrefixRank = indicesShape.rank() - 1;
        int dataSuffixStart = batchDimensions + tupleDepth;
        int dataSuffixRank = dataShape.rank() - dataSuffixStart;
        int updatesRank = Math.addExact(indicesPrefixRank, dataSuffixRank);
        Dimension[] updatesDimensions = new Dimension[updatesRank];
        int updatesAxis = 0;
        for (int indicesAxis = 0; indicesAxis < indicesPrefixRank; indicesAxis++) {
            updatesDimensions[updatesAxis++] = indicesShape.dimensions().get(indicesAxis);
        }
        for (int dataAxis = dataSuffixStart; dataAxis < dataShape.rank(); dataAxis++) {
            updatesDimensions[updatesAxis++] = dataShape.dimensions().get(dataAxis);
        }
        return Shape.ofDimensions(updatesDimensions);
    }

    /**
     * Creates exact descriptor, semantics, ordered provenance, and one fresh derived Tensor.
     *
     * @param data non-null exact provenance input zero and source of result Shape/type
     * @param indices non-null exact provenance input one
     * @param updates non-null exact provenance input two
     * @param dataDescriptor non-null descriptor supplying exact retained metadata
     * @param updatesDescriptor non-null descriptor supplying eligibility to combine
     * @param attrs non-null normalized batch/reduction attributes retained by the operation
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalArgumentException if descriptor invariants reject supplied metadata
     * @throws IllegalStateException if tensor identifier space is exhausted at final delegation
     */
    private static Tensor create(
            Tensor data,
            Tensor indices,
            Tensor updates,
            TensorDescriptor dataDescriptor,
            TensorDescriptor updatesDescriptor,
            ScatterNdAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataDescriptor.dataType(),
                dataDescriptor.shape(),
                Optional.empty(),
                dataDescriptor.requiresGrad() || updatesDescriptor.requiresGrad());
        Operation operation = new Operation(ScatterNdKind.SCATTER_ND, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(data, indices, updates));
    }
}
