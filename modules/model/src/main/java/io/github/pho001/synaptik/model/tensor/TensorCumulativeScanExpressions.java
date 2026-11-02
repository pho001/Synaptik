package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, one-axis cumulative-scan expressions.
 *
 * <p>A cumulative scan preserves logical shape while accumulating addition or multiplication
 * along one axis. Inclusive mode includes the current position; exclusive mode omits it and gives
 * the first traversed position the selected kind's identity. Reverse mode changes traversal
 * direction without reversing output positions.</p>
 *
 * <p>This package-private boundary accepts exactly FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, and
 * INT64 inputs, normalizes one axis, preserves the exact input Shape, data type, and
 * gradient-eligibility metadata, and records exact one-input provenance. It does not inspect or accumulate values,
 * preserve resolved layout,
 * allocate storage, define gradient rules, capture a graph, or provide compiler, runtime,
 * backend, or execution behavior.</p>
 */
final class TensorCumulativeScanExpressions {
    /** Prevents instantiation because cumulative-scan expression construction owns no state. */
    private TensorCumulativeScanExpressions() {
    }

    /**
     * Validates local input metadata and creates one fresh cumulative-scan expression.
     *
     * <p>Validation checks {@code input}, then {@code kind}, then accepts exactly floating or
     * integral data types, reads the exact input Shape, normalizes {@code axis} exactly once, and
     * constructs one {@link CumulativeScanAttrs}. Failures before factory delegation consume no
     * Tensor identity.</p>
     *
     * @param input non-null FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, or INT64 Tensor retained as
     *     the sole provenance input
     * @param kind non-null cumulative arithmetic kind retained by the operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param exclusive {@code true} to omit each current position from its traversed prefix, or
     *     {@code false} to include it
     * @param reverse {@code true} to traverse from the axis end without changing output order, or
     *     {@code false} to traverse forward
     * @return the non-null fresh storage-free expression with exact Shape, type, gradient
     *     eligibility, normalized attributes, and one-input provenance
     * @throws NullPointerException if {@code input} or {@code kind} is null, with the matching
     *     parameter name as message; input validation occurs first
     * @throws IllegalArgumentException if the input is not floating or integral, with message
     *     {@code input must have a numeric data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(
            Tensor input,
            CumulativeScanKind kind,
            int axis,
            boolean exclusive,
            boolean reverse) {
        return applyChecked(input, kind, axis, exclusive, reverse, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, or INT64 Tensor retained as
     *     the sole provenance input
     * @param kind non-null cumulative arithmetic kind retained by the operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param exclusive {@code true} to omit each current position from its traversed prefix, or
     *     {@code false} to include it
     * @param reverse {@code true} to traverse from the axis end without changing output order, or
     *     {@code false} to traverse forward
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code kind} is null, with the matching
     *     parameter name as message; input validation occurs first
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input is not floating or integral, with message
     *     {@code input must have a numeric data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor applyChecked(
            Tensor input,
            CumulativeScanKind kind,
            int axis,
            boolean exclusive,
            boolean reverse,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        return applyDescriptors(
                input.descriptor(), kind, axis, exclusive, reverse, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axis ordinary construction argument, validated in existing order
     * @param exclusive ordinary construction argument, validated in existing order
     * @param reverse ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            CumulativeScanKind kind,
            int axis,
            boolean exclusive,
            boolean reverse,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateNumericInput(input);
        Shape inputShape = input.shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        CumulativeScanAttrs attrs =
                new CumulativeScanAttrs(normalizedAxis, exclusive, reverse);
        return createDescriptors(input, inputShape, kind, attrs, descriptorCheck, inputTensor);
    }

    /**
     * Accepts exactly floating or integral input metadata without conversion or promotion.
     *
     * @param input non-null descriptor supplying the data type
     * @throws IllegalArgumentException if the input type is neither floating nor integral, with
     *     message {@code input must have a numeric data type, but was <dataType>}
     */
    private static void validateNumericInput(TensorDescriptor input) {
        DataType dataType = input.dataType();
        if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type, but was " + dataType);
        }
    }


    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param shape ordinary construction argument, validated in existing order
     * @param kind ordinary construction argument, validated in existing order
     * @param attrs ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            Shape shape,
            CumulativeScanKind kind,
            CumulativeScanAttrs attrs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.dataType(),
                shape,
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(inputTensor));
    }

    /**
     * Completes one admitted ordinary occurrence synchronously using the real construction guards.
     * No Tensor or identity is created and the callback is not retained. The owning facade checks
     * occurrence cardinality and compares the returned canonical operation with the supplied one.
     *
     * @param operation non-null ordinary operation admitted to this family by the owning selector
     * @param inputs non-null immutable operand metadata in canonical input order
     * @param outputCount complete canonical output count, already validated by the owning facade
     * @param sink non-null synchronous consumer of the canonical operation and all output metadata
     * @throws IllegalArgumentException if ordinary semantic guards or the consumer reject metadata
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(sink, "sink");
        CumulativeScanAttrs attrs = (CumulativeScanAttrs) operation.attrs();
        applyDescriptors(inputs.get(0), (CumulativeScanKind) operation.kind(), attrs.axis(),
                attrs.exclusive(), attrs.reverse(), sink, null);
    }
}
