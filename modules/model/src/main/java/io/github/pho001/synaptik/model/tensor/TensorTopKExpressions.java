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
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input.descriptor();
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
        List<Tensor> outputs = TensorFactory.createDerivedOutputs(
                operation,
                List.of(input),
                List.of(valuesDescriptor, indicesDescriptor));
        return new TopKResult(outputs.get(0), outputs.get(1));
    }
}
