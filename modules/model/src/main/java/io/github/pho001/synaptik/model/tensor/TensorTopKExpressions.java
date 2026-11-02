package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Constructs storage-free top-K values and indices from one shared producer.
 *
 * <p>This helper derives descriptors and pre-capture provenance only. It does not read input
 * storage, evaluate values, construct gradients, capture a graph, choose an algorithm or backend,
 * or execute the occurrence.</p>
 */
final class TensorTopKExpressions {
    private TensorTopKExpressions() {
    }

    /**
     * Validates local metadata and creates one two-output top-K occurrence.
     *
     * <p>The values result preserves the input data type and gradient request. The indices result
     * uses non-differentiable INT64. Both share one derived Shape whose selected dimension is
     * static {@code k}; every other dimension reference is preserved. A known static selected
     * extent is checked locally, while a dynamic or expression-bound extent is deferred to later
     * compiler or binding validation.</p>
     *
     * @param input non-null exact producer input, never mutated
     * @param k non-negative selection count within a known selected static extent; a bound against
     *     a dynamic or expression extent is deferred
     * @param axis positive or negative input axis
     * @param largest whether selection requests largest rather than smallest values
     * @param sorted whether output retains selection order rather than logical-index order
     * @return fresh, non-null values and INT64 indices wrappers sharing one exact producer, Shape,
     *     and ordered provenance slots zero and one
     * @throws NullPointerException if {@code input} is null
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape
     * @throws IllegalArgumentException if {@code k} is negative or exceeds a selected static extent
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static TopKResult apply(Tensor input, long k, int axis, boolean largest, boolean sorted) {
        return applyChecked(input, k, axis, largest, sorted, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null exact producer input, never mutated
     * @param k non-negative selection count within a known selected static extent; a bound against
     *     a dynamic or expression extent is deferred
     * @param axis positive or negative input axis
     * @param largest whether selection requests largest rather than smallest values
     * @param sorted whether output retains selection order rather than logical-index order
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} is null
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if {@code k} is negative or exceeds a selected static extent
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static TopKResult applyChecked(Tensor input, long k, int axis, boolean largest, boolean sorted,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                input == null ? null : input.descriptor(),
                k,
                axis,
                largest,
                sorted,
                descriptorCheck,
                input);
    }

    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The same descriptor guard core serves real construction; the sink is never retained.
     * @param operation non-null ordinary operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, including repeated occurrences
     * @param outputCount exact canonical sibling count
     * @param sink non-null consumer called exactly once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject the request
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != 1 || outputCount != 2) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        TopKAttrs attrs = (TopKAttrs) operation.attrs();
        applyDescriptors(inputs.get(0), attrs.k(), attrs.axis(), attrs.largest(), attrs.sorted(), sink, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param k ordinary checked construction parameter
     * @param axis ordinary checked construction parameter
     * @param largest ordinary checked construction parameter
     * @param sorted ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static TopKResult applyDescriptors(
            TensorDescriptor input,
            long k,
            int axis,
            boolean largest,
            boolean sorted,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input;
        Shape inputShape = inputDescriptor.shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        TopKAttrs attrs = new TopKAttrs(normalizedAxis, k, largest, sorted);

        OptionalLong staticExtent = inputShape.dimensions().get(normalizedAxis).staticSize();
        if (staticExtent.isPresent() && k > staticExtent.getAsLong()) {
            throw new IllegalArgumentException(
                    "k must not exceed selected static extent: k=" + k
                            + ", axis=" + normalizedAxis
                            + ", extent=" + staticExtent.getAsLong());
        }

        Dimension[] outputDimensions = inputShape.dimensions().toArray(Dimension[]::new);
        outputDimensions[normalizedAxis] = new StaticDimension(k);
        Shape outputShape = Shape.ofDimensions(outputDimensions);
        TensorDescriptor valuesDescriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                outputShape,
                Optional.empty(),
                inputDescriptor.requiresGrad());
        TensorDescriptor indicesDescriptor = new TensorDescriptor(
                DataType.INT64,
                outputShape,
                Optional.empty(),
                false);
        Operation operation = new Operation(TopKKind.TOP_K, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(valuesDescriptor, indicesDescriptor));
            return null;
        }
        List<Tensor> outputs = TensorFactory.createDerivedOutputs(
                operation,
                List.of(originalInput),
                List.of(valuesDescriptor, indicesDescriptor));
        return new TopKResult(outputs.get(0), outputs.get(1));
    }
}
