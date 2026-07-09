package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free tensor-composition expressions.
 *
 * <p>Concat and stack defensively snapshot an ordered non-empty varargs input sequence, require
 * exact data types, validate their operation-specific Shape relationships, and combine gradient
 * eligibility by logical OR. Concat folds selected extents through canonical symbolic addition;
 * static values, symbolic coefficients, and symbolic offsets use checked {@code long} arithmetic.
 * Unstack creates an immutable ordered list of scalar-select expressions only when the selected
 * source extent has a statically known {@code int}-sized count.</p>
 *
 * <p>This field-free helper derives immutable model metadata only. It does not promote or
 * broadcast inputs, inspect values or storage, bind or evaluate symbolic extents, group unstack
 * outputs, create graph state, define gradients, choose materialization or lowering, map ONNX, or
 * execute work.</p>
 */
final class TensorCompositionExpressions {
    /** Prevents instantiation because composition expression construction owns no state. */
    private TensorCompositionExpressions() {
    }

    /**
     * Validates and constructs one ordered CONCAT expression.
     *
     * <p>The input container, non-empty requirement, and copied elements are validated in that
     * order before descriptors are inspected. Axis normalization then precedes encounter-order
     * validation of exact data type, rank, and every non-concat dimension. Selected extents are
     * then folded in input order from static zero through canonical symbolic addition. Static-zero
     * companions and a one-input concat preserve the opposing selected Dimension reference when
     * the canonical addition rule permits it.</p>
     *
     * @param axis positive or negative existing axis
     * @param inputs non-null caller-owned non-empty array with no null elements
     * @return a non-null fresh unlabeled and storage-free CONCAT tensor with unresolved layout
     * @throws NullPointerException if {@code inputs} or an indexed element is null
     * @throws IllegalArgumentException if input count, type, rank, or non-axis Shape rules fail
     * @throws IndexOutOfBoundsException if {@code axis} is outside the first input's rank
     * @throws ArithmeticException if a static selected extent, symbolic coefficient, or symbolic
     *     offset overflows {@code long}
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor concat(int axis, Tensor[] inputs) {
        List<Tensor> snapshot = snapshotInputs("concat", inputs);
        TensorDescriptor firstDescriptor = snapshot.getFirst().descriptor();
        Shape firstShape = firstDescriptor.shape();
        DataType dataType = firstDescriptor.dataType();
        int normalizedAxis = normalizeExistingAxis("concat", axis, firstShape.rank());
        boolean requiresGrad = false;
        for (int inputIndex = 0; inputIndex < snapshot.size(); inputIndex++) {
            TensorDescriptor descriptor = snapshot.get(inputIndex).descriptor();
            if (descriptor.dataType() != dataType) {
                throw new IllegalArgumentException(
                        "concat inputs must have matching data types: inputs[" + inputIndex
                                + "] is " + descriptor.dataType() + ", expected " + dataType);
            }
            Shape shape = descriptor.shape();
            if (shape.rank() != firstShape.rank()) {
                throw new IllegalArgumentException(
                        "concat inputs must have matching ranks: inputs[" + inputIndex + "] has "
                                + shape.rank() + ", expected " + firstShape.rank());
            }
            for (int shapeAxis = 0; shapeAxis < shape.rank(); shapeAxis++) {
                if (shapeAxis != normalizedAxis
                        && !shape.dimensions().get(shapeAxis)
                                .equals(firstShape.dimensions().get(shapeAxis))) {
                    throw new IllegalArgumentException(
                            "concat inputs differ at non-concat axis " + shapeAxis + ": inputs["
                                    + inputIndex + "]");
                }
            }
            requiresGrad |= descriptor.requiresGrad();
        }
        Shape resultShape = concatShape(snapshot, firstShape, normalizedAxis);
        Operation operation = new Operation(
                TensorCompositionKind.CONCAT, new CompositionAxisAttrs(normalizedAxis));
        return create(dataType, resultShape, requiresGrad, operation, snapshot);
    }

    /**
     * Validates and constructs one ordered STACK expression.
     *
     * <p>The input container, non-empty requirement, and copied elements are validated in that
     * order before descriptors are inspected. Insertion-axis normalization then precedes
     * encounter-order validation of exact data types and structural Shape equality. The result
     * inserts the input count as one static dimension and otherwise reuses the first Shape's exact
     * Dimension references.</p>
     *
     * @param axis positive or negative result-axis insertion position
     * @param inputs non-null caller-owned non-empty array with no null elements
     * @return a non-null fresh unlabeled and storage-free STACK tensor with unresolved layout
     * @throws NullPointerException if {@code inputs} or an indexed element is null
     * @throws IllegalArgumentException if input count, exact type, or Shape equality fails
     * @throws IndexOutOfBoundsException if {@code axis} is outside the insertion range
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor stack(int axis, Tensor[] inputs) {
        List<Tensor> snapshot = snapshotInputs("stack", inputs);
        TensorDescriptor firstDescriptor = snapshot.getFirst().descriptor();
        Shape firstShape = firstDescriptor.shape();
        DataType dataType = firstDescriptor.dataType();
        int normalizedAxis = normalizeInsertionAxis(axis, firstShape.rank());
        boolean requiresGrad = false;
        for (int inputIndex = 0; inputIndex < snapshot.size(); inputIndex++) {
            TensorDescriptor descriptor = snapshot.get(inputIndex).descriptor();
            if (descriptor.dataType() != dataType) {
                throw new IllegalArgumentException(
                        "stack inputs must have matching data types: inputs[" + inputIndex
                                + "] is " + descriptor.dataType() + ", expected " + dataType);
            }
            if (!descriptor.shape().equals(firstShape)) {
                throw new IllegalArgumentException(
                        "stack inputs must have identical shapes: inputs[" + inputIndex
                                + "] differs from inputs[0]");
            }
            requiresGrad |= descriptor.requiresGrad();
        }
        Shape resultShape = stackShape(firstShape, normalizedAxis, snapshot.size());
        Operation operation = new Operation(
                TensorCompositionKind.STACK, new CompositionAxisAttrs(normalizedAxis));
        return create(dataType, resultShape, requiresGrad, operation, snapshot);
    }

    /**
     * Constructs one independent scalar-select tensor per selected-axis coordinate.
     *
     * <p>After the input null check, the selected existing axis is normalized, then required to be
     * static and no larger than
     * {@link Integer#MAX_VALUE}. Its Dimension is removed once to form the shared immutable result
     * Shape. Each output delegates to scalar-select construction and therefore has an independent
     * one-output producer and provenance output index zero. A zero extent returns {@link List#of()}
     * before any operation or Tensor allocation.</p>
     *
     * @param input non-null exact sole provenance input for every result
     * @param axis positive or negative existing source axis
     * @return a non-null immutable ordered list of fresh scalar-select tensors
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IndexOutOfBoundsException if {@code axis} is outside the input rank
     * @throws IllegalArgumentException if the selected extent is dynamic or greater than
     *     {@link Integer#MAX_VALUE}
     * @throws IllegalStateException if identifier space is exhausted during output construction;
     *     already consumed identifiers are not rolled back
     */
    static List<Tensor> unstack(Tensor input, int axis) {
        Objects.requireNonNull(input, "input");
        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = normalizeExistingAxis("unstack", axis, inputShape.rank());
        Dimension selectedDimension = inputShape.dimensions().get(normalizedAxis);
        if (!(selectedDimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    "unstack axis " + normalizedAxis + " must have a statically known dimension");
        }
        long size = staticDimension.size();
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "unstack axis " + normalizedAxis + " size " + size
                            + " exceeds maximum result count " + Integer.MAX_VALUE);
        }
        if (size == 0) {
            return List.of();
        }
        int outputCount = (int) size;
        List<Tensor> outputs = new ArrayList<>(outputCount);
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            outputs.add(TensorSelectExpressions.apply(input, normalizedAxis, outputIndex));
        }
        return List.copyOf(outputs);
    }

    /**
     * Validates and snapshots one caller-owned ordered Tensor array.
     *
     * @param operation operation name used in the empty-input failure text; internal callers pass
     *     a non-null literal
     * @param inputs non-null caller-owned array to clone exactly once
     * @return a non-null immutable ordered list retaining exact input references; the caller's
     *     array object and its clone are not retained
     * @throws NullPointerException if {@code inputs} or an indexed copied element is null
     * @throws IllegalArgumentException if {@code inputs} is empty
     */
    private static List<Tensor> snapshotInputs(String operation, Tensor[] inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.length == 0) {
            throw new IllegalArgumentException(operation + " requires at least one input");
        }
        Tensor[] copy = inputs.clone();
        for (int index = 0; index < copy.length; index++) {
            Objects.requireNonNull(copy[index], "inputs[" + index + "]");
        }
        return List.copyOf(Arrays.asList(copy));
    }

    /**
     * Normalizes one existing-axis request with one negative-axis adjustment.
     *
     * @param operation non-null operation name used in failure text
     * @param axis raw positive or negative axis
     * @param rank non-negative existing Shape rank
     * @return the normalized axis in {@code [0, rank)}
     * @throws IndexOutOfBoundsException if the raw axis is invalid
     */
    private static int normalizeExistingAxis(String operation, int axis, int rank) {
        long normalized = axis;
        if (normalized < 0) {
            normalized += rank;
        }
        if (normalized < 0 || normalized >= rank) {
            throw new IndexOutOfBoundsException(
                    operation + " axis " + axis + " is outside shape rank " + rank);
        }
        return (int) normalized;
    }

    /**
     * Normalizes one STACK insertion-axis request with one negative-axis adjustment.
     *
     * @param axis raw positive or negative insertion axis
     * @param rank non-negative input Shape rank
     * @return normalized result insertion axis in {@code [0, rank]}
     * @throws IndexOutOfBoundsException if the raw axis is outside the insertion range
     */
    private static int normalizeInsertionAxis(int axis, int rank) {
        long insertionCount = (long) rank + 1;
        long normalized = axis;
        if (normalized < 0) {
            normalized += insertionCount;
        }
        if (normalized < 0 || normalized > rank) {
            throw new IndexOutOfBoundsException(
                    "stack axis " + axis + " is outside insertion range for shape rank " + rank);
        }
        return (int) normalized;
    }

    /**
     * Derives one canonical same-rank CONCAT result Shape.
     *
     * <p>Non-selected axes preserve exact first-input Dimension references. The selected axis
     * starts at static zero and encounter-order folds every input extent through
     * {@link DimensionExpressions#add(Dimension, Dimension)}. Static inputs stay static, named
     * extents {@code N} and {@code M} become canonical {@code N + M}, repeated terms combine, and
     * existing linear expressions flatten. Division and constrained-unknown dimensions remain
     * atomic terms. Static-zero companions and a one-input concat preserve the opposing exact
     * reference when canonical addition permits it. Construction retains the formula without
     * binding or evaluating a concrete size.</p>
     *
     * @param inputs non-null immutable ordered validated inputs
     * @param firstShape non-null exact first-input Shape
     * @param normalizedAxis existing selected axis
     * @return a non-null same-rank immutable result Shape
     * @throws ArithmeticException if a static selected extent, symbolic coefficient, or symbolic
     *     offset overflows {@code long}
     */
    private static Shape concatShape(
            List<Tensor> inputs, Shape firstShape, int normalizedAxis) {
        Dimension[] resultDimensions = new Dimension[firstShape.rank()];
        for (int axis = 0; axis < firstShape.rank(); axis++) {
            resultDimensions[axis] = firstShape.dimensions().get(axis);
        }
        Dimension selectedExtent = new StaticDimension(0);
        for (Tensor input : inputs) {
            Dimension selected = input.descriptor().shape().dimensions().get(normalizedAxis);
            selectedExtent = DimensionExpressions.add(selectedExtent, selected);
        }
        resultDimensions[normalizedAxis] = selectedExtent;
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Inserts one static input-count Dimension into an identical input Shape.
     *
     * @param inputShape non-null exact first-input Shape
     * @param normalizedAxis normalized insertion position in {@code [0, inputRank]}
     * @param inputCount positive number of validated inputs
     * @return a non-null rank-plus-one Shape preserving all original Dimension references
     */
    private static Shape stackShape(Shape inputShape, int normalizedAxis, int inputCount) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank() + 1];
        for (int resultAxis = 0; resultAxis < resultDimensions.length; resultAxis++) {
            if (resultAxis < normalizedAxis) {
                resultDimensions[resultAxis] = inputShape.dimensions().get(resultAxis);
            } else if (resultAxis == normalizedAxis) {
                resultDimensions[resultAxis] = new StaticDimension(inputCount);
            } else {
                resultDimensions[resultAxis] = inputShape.dimensions().get(resultAxis - 1);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Creates one unresolved descriptor and delegates exact producer state for one fresh Tensor.
     *
     * @param dataType non-null exact result element type
     * @param resultShape non-null exact locally derived result Shape
     * @param requiresGrad gradient-eligibility value for the result descriptor
     * @param operation non-null exact composition operation
     * @param inputs non-null immutable ordered exact provenance inputs
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    private static Tensor create(
            DataType dataType,
            Shape resultShape,
            boolean requiresGrad,
            Operation operation,
            List<Tensor> inputs) {
        TensorDescriptor descriptor =
                new TensorDescriptor(dataType, resultShape, Optional.empty(), requiresGrad);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, inputs);
    }
}
