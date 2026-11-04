package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free axis-gather expressions and embedding composition
 * for {@link Tensor}.
 *
 * <p>Every operation consumes ordered {@code [data, indices]} inputs and requires exact
 * {@link DataType#INT32} or {@link DataType#INT64} indices. {@link AxisGatherKind#GATHER}
 * replaces one data axis with the complete indices Shape. {@link AxisGatherKind#GATHER_ELEMENTS}
 * requires same-rank non-axis alignment and retains the exact indices Shape.</p>
 *
 * <p>Every fresh result preserves the data tensor's type and gradient eligibility, leaves layout
 * unresolved, and records exact ordered provenance without a label or storage. This field-free
 * helper never reads values, checks index bounds, defines gradient or scatter behavior, captures
 * a graph, chooses materialization or backend support, or executes work.</p>
 */
final class TensorAxisGatherExpressions {
    /** Prevents instantiation because axis-gather expression construction owns no state. */
    private TensorAxisGatherExpressions() {
    }

    /**
     * Validates and creates one canonical GATHER expression.
     *
     * <p>The complete indices Shape replaces the selected data Dimension. Thus data
     * {@code [2, 3, 4]}, axis {@code 1}, and indices {@code [5, 6]} produce
     * {@code [2, 5, 6, 4]}; scalar indices produce {@code [2, 4]}. Every inserted indices
     * Dimension and unaffected data Dimension is retained exactly.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER tensor with inserted indices Shape and unresolved layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the indices data type is not INT32 or INT64
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gather(Tensor data, Tensor indices, int axis) {
        return gatherChecked(data, indices, axis, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinate tensor retained as provenance input one
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the indices data type is not INT32 or INT64
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gatherChecked(Tensor data, Tensor indices, int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return gatherDescriptors(
                data == null ? null : data.descriptor(),
                indices == null ? null : indices.descriptor(),
                axis,
                descriptorCheck,
                data,
                indices);
    }

    /**
     * Validates an embedding-weight table and creates one canonical axis-zero GATHER expression.
     *
     * <p>{@code weights} must have rank two and exact BFLOAT16, FLOAT16, OFP8, FLOAT32, or FLOAT64 type.
     * Its axis zero is the vocabulary axis and axis one is the embedding Dimension. Indices may
     * have any rank, including scalar, but must have exact INT32 or INT64 type. The result Shape is
     * the complete indices Shape followed by the exact weight axis-one Dimension.</p>
     *
     * <p>After embedding-specific validation, this method delegates directly to
     * {@link #gather(Tensor, Tensor, int)} with axis zero. The sole resulting producer therefore
     * retains the ordinary {@link AxisGatherKind#GATHER} operation, {@code IndexAxisAttrs(0)},
     * exact ordered inputs {@code [weights, indices]}, and one output at provenance index zero.
     * It preserves the weight type and gradient eligibility and creates a fresh ID only during
     * final derived-Tensor construction. No index value is read: negative and out-of-range values
     * remain invalid for future ordinary Gather execution, where bounds must be enforced safely
     * without wrapping, clamping, padding, or selecting a default row. No padding-index,
     * sparse-gradient, maximum-norm, or frequency-scaling option changes that ordinary Gather
     * occurrence.</p>
     *
     * @param weights non-null rank-two BFLOAT16, FLOAT16, OFP8, FLOAT32, or FLOAT64 table retained as
     *     provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinates of any rank retained as provenance input
     *     one and never mutated
     * @return a non-null fresh storage-free GATHER tensor whose Shape is the indices Shape plus
     *     the exact weight embedding Dimension, with weight metadata and unresolved layout
     * @throws NullPointerException if {@code weights} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if weights are not rank two, weights are not floating, or
     *     indices are not INT32 or INT64, checked in that order before ID allocation
     * @throws ArithmeticException if checked Gather result-Shape metadata construction overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor embedding(Tensor weights, Tensor indices) {
        return embeddingChecked(weights, indices, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param weights non-null rank-two BFLOAT16, FLOAT16, OFP8, FLOAT32, or FLOAT64 table retained as
     *     provenance input zero and never mutated
     * @param indices non-null INT32 or INT64 coordinates of any rank retained as provenance input
     *     one and never mutated
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code weights} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if weights are not rank two, weights are not floating, or
     *     indices are not INT32 or INT64, checked in that order before ID allocation
     * @throws ArithmeticException if checked Gather result-Shape metadata construction overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor embeddingChecked(Tensor weights, Tensor indices,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return embeddingDescriptors(
                weights == null ? null : weights.descriptor(),
                indices == null ? null : indices.descriptor(),
                descriptorCheck,
                weights,
                indices);
    }

    /**
     * Validates and creates one GATHER_ELEMENTS expression with aligned non-axis coordinates.
     *
     * <p>Data and indices ranks must match, and every non-selected Dimension must be structurally
     * equal. Selected extents may differ. Data {@code [2, 3, 4]}, indices {@code [2, 7, 4]}, and
     * axis {@code 1} retain the exact indices Shape {@code [2, 7, 4]} as the result.</p>
     *
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null same-rank INT32 or INT64 tensor retained as provenance input one;
     *     its non-axis Dimensions must equal data and its exact Shape becomes the result
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @return a non-null fresh GATHER_ELEMENTS tensor retaining exact indices Shape and unresolved
     *     layout
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if index type, rank, or non-axis alignment is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gatherElements(Tensor data, Tensor indices, int axis) {
        return gatherElementsChecked(data, indices, axis, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param data non-null value tensor retained as provenance input zero and never mutated
     * @param indices non-null same-rank INT32 or INT64 tensor retained as provenance input one;
     *     its non-axis Dimensions must equal data and its exact Shape becomes the result
     * @param axis raw positive or negative data axis normalized once against the data Shape
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code data} or {@code indices} is null, checked in order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if index type, rank, or non-axis alignment is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data Shape
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor gatherElementsChecked(Tensor data, Tensor indices, int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return gatherElementsDescriptors(
                data == null ? null : data.descriptor(),
                indices == null ? null : indices.descriptor(),
                axis,
                descriptorCheck,
                data,
                indices);
    }

    /**
     * Requires the exact supported signed-integral index representation.
     *
     * @param operation non-null stable public operation name used in exact failure text
     * @param indicesDescriptor non-null exact descriptor supplying the index data type
     * @throws IllegalArgumentException if the type is not INT32 or INT64
     */
    private static void validateIndexType(
            String operation, TensorDescriptor indicesDescriptor) {
        DataType dataType = indicesDescriptor.dataType();
        if (dataType != DataType.INT32 && dataType != DataType.INT64) {
            throw new IllegalArgumentException(
                    operation + " indices data type must be INT32 or INT64: " + dataType);
        }
    }

    /**
     * Replaces one data axis with every indices Dimension in original order.
     *
     * @param dataShape non-null source data Shape of rank at least one
     * @param indicesShape non-null index Shape whose complete Dimensions are inserted
     * @param normalizedAxis normalized existing data axis to replace
     * @return a non-null Shape ordered as data-before, all indices, then data-after
     */
    private static Shape gatherShape(
            Shape dataShape, Shape indicesShape, int normalizedAxis) {
        Dimension[] resultDimensions =
                new Dimension[dataShape.rank() - 1 + indicesShape.rank()];
        int resultAxis = 0;
        for (int dataAxis = 0; dataAxis < normalizedAxis; dataAxis++) {
            resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
        }
        for (Dimension indicesDimension : indicesShape.dimensions()) {
            resultDimensions[resultAxis++] = indicesDimension;
        }
        for (int dataAxis = normalizedAxis + 1; dataAxis < dataShape.rank(); dataAxis++) {
            resultDimensions[resultAxis++] = dataShape.dimensions().get(dataAxis);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Validates equal ranks and structural equality of every non-selected Dimension.
     *
     * @param dataShape non-null source data Shape
     * @param indicesShape non-null candidate indices Shape
     * @param normalizedAxis normalized data axis whose extents are deliberately not compared
     * @throws IllegalArgumentException if ranks differ or the first increasing non-axis Dimension
     *     differs structurally
     */
    private static void validateGatherElements(
            Shape dataShape, Shape indicesShape, int normalizedAxis) {
        if (indicesShape.rank() != dataShape.rank()) {
            throw new IllegalArgumentException(
                    "gatherElements indices rank must match data rank: expected="
                            + dataShape.rank() + ", actual=" + indicesShape.rank());
        }
        for (int axis = 0; axis < dataShape.rank(); axis++) {
            if (axis == normalizedAxis) {
                continue;
            }
            Dimension expected = dataShape.dimensions().get(axis);
            Dimension actual = indicesShape.dimensions().get(axis);
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "gatherElements indices dimension at axis " + axis
                                + " must match data: expected=" + expected + ", actual=" + actual);
            }
        }
    }


    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param data exact real provenance operand, or null for descriptor completion
     * @param indices exact real provenance operand, or null for descriptor completion
     * @param dataDescriptor non-null exact descriptor supplying retained type and eligibility
     * @param resultShape non-null operation-specific result Shape
     * @param kind non-null exact axis-gather semantic identity
     * @param attrs non-null normalized axis attributes retained by the operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if descriptor invariants reject the supplied metadata
     * @throws IllegalStateException if tensor identifier space is exhausted at final delegation
     */
    private static Tensor createChecked(
            Tensor data,
            Tensor indices,
            TensorDescriptor dataDescriptor,
            Shape resultShape,
            AxisGatherKind kind,
            IndexAxisAttrs attrs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataDescriptor.dataType(),
                resultShape,
                Optional.empty(),
                dataDescriptor.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(data, indices));
    }

    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The real construction path calls these same descriptor guards and result helpers.
     * @param operation non-null canonical operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, retaining aliases and positions
     * @param outputCount exact canonical sibling count, one
     * @param sink non-null unretained consumer called once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject input
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != 2 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        int axis = ((IndexAxisAttrs) operation.attrs()).axis();
        switch ((AxisGatherKind) operation.kind()) {
            case GATHER -> gatherDescriptors(inputs.get(0), inputs.get(1), axis, sink, null, null);
            case GATHER_ELEMENTS -> gatherElementsDescriptors(inputs.get(0), inputs.get(1), axis, sink, null, null);
        }
    }


    /**
     * Runs the ordinary descriptor guards shared by real and identity-free construction.
     * @param data exact operand descriptor, subject to ordinary null/semantic guards
     * @param indices exact operand descriptor, subject to ordinary null/semantic guards
     * @param axis ordinary checked construction parameter
     * @param descriptorCheck ordinary checked construction parameter
     * @param originalData exact provenance operand, or null for descriptor completion
     * @param originalIndices exact provenance operand, or null for descriptor completion
     * @return the real result, or null after synchronous, unretained descriptor completion
     */
    private static Tensor gatherDescriptors(
            TensorDescriptor data,
            TensorDescriptor indices,
            int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalData,
            Tensor originalIndices) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data;
        TensorDescriptor indicesDescriptor = indices;
        validateIndexType("gather", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape resultShape = gatherShape(
                dataShape, indicesDescriptor.shape(), normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return createChecked(originalData, originalIndices, dataDescriptor, resultShape, AxisGatherKind.GATHER, attrs, descriptorCheck);
    }

    /**
     * Runs the ordinary descriptor guards shared by real and identity-free construction.
     * @param weights exact operand descriptor, subject to ordinary null/semantic guards
     * @param indices exact operand descriptor, subject to ordinary null/semantic guards
     * @param descriptorCheck ordinary checked construction parameter
     * @param originalWeights exact provenance operand, or null for descriptor completion
     * @param originalIndices exact provenance operand, or null for descriptor completion
     * @return the real result, or null after synchronous, unretained descriptor completion
     */
    private static Tensor embeddingDescriptors(
            TensorDescriptor weights,
            TensorDescriptor indices,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalWeights,
            Tensor originalIndices) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor weightsDescriptor = weights;
        int weightsRank = weightsDescriptor.shape().rank();
        if (weightsRank != 2) {
            throw new IllegalArgumentException(
                    "embedding weights rank must be 2: actual=" + weightsRank);
        }
        DataType weightsDataType = weightsDescriptor.dataType();
        if (!weightsDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "embedding weights data type must be BFLOAT16, FLOAT16, FLOAT32, or FLOAT64: "
                            + weightsDataType);
        }
        validateIndexType("embedding", indices);
        return gatherDescriptors(weights, indices, 0, descriptorCheck, originalWeights, originalIndices);
    }

    /**
     * Runs the ordinary descriptor guards shared by real and identity-free construction.
     * @param data exact operand descriptor, subject to ordinary null/semantic guards
     * @param indices exact operand descriptor, subject to ordinary null/semantic guards
     * @param axis ordinary checked construction parameter
     * @param descriptorCheck ordinary checked construction parameter
     * @param originalData exact provenance operand, or null for descriptor completion
     * @param originalIndices exact provenance operand, or null for descriptor completion
     * @return the real result, or null after synchronous, unretained descriptor completion
     */
    private static Tensor gatherElementsDescriptors(
            TensorDescriptor data,
            TensorDescriptor indices,
            int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalData,
            Tensor originalIndices) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(indices, "indices");
        TensorDescriptor dataDescriptor = data;
        TensorDescriptor indicesDescriptor = indices;
        validateIndexType("gatherElements", indicesDescriptor);
        Shape dataShape = dataDescriptor.shape();
        int normalizedAxis = dataShape.normalizeAxis(axis);
        Shape indicesShape = indicesDescriptor.shape();
        validateGatherElements(dataShape, indicesShape, normalizedAxis);
        IndexAxisAttrs attrs = new IndexAxisAttrs(normalizedAxis);
        return createChecked(originalData, originalIndices, dataDescriptor, indicesShape, AxisGatherKind.GATHER_ELEMENTS, attrs, descriptorCheck);
    }
}
