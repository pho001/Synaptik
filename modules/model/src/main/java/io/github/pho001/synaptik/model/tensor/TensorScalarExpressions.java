package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free scalar elementwise tensor expressions.
 *
 * <p>This package-private boundary owns the deterministic construction path for scalar
 * arithmetic and inclusive clamp requests. One-value ADD, SUB, MUL, MIN, and MAX requests accept
 * floating or signed-integral input, while DIV, POW, and the first-class two-bound CLAMP remain
 * floating-only. Every accepted scalar exactly matches the input data type. Construction retains
 * the input's exact data type and shape reference, preserves gradient eligibility, and records
 * exact typed scalar parameters unchanged in operation attributes. It does not inspect values or
 * storage, convert parameters, evaluate mathematics,
 * validate numerical domains, canonicalize or decompose expressions, create gradient rules, or
 * capture a graph.</p>
 */
final class TensorScalarExpressions {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorScalarExpressions() {
    }

    /**
     * Creates one fresh derived tensor for a one-parameter scalar elementwise request.
     *
     * <p>Validation occurs in this exact order: null-check {@code input}, {@code kind}, and
     * {@code value}; reject {@link ScalarElementwiseKind#CLAMP} because it requires range
     * attributes; validate a floating or signed-integral input; reject integral {@code DIV} and
     * {@code POW}; then require exact value/input data-type equality. The exact supplied reference
     * is retained in one
     * {@link ScalarValueAttrs}, which is paired with the exact supplied kind before common result
     * construction. A failed validation allocates no tensor identity.</p>
     *
     * @param input non-null floating or integral tensor retained by exact reference in result
     *     provenance
     * @param kind non-null scalar kind other than {@code CLAMP}, retained in the result operation
     * @param value non-null exact scalar arithmetic parameter; its data type must
     *     equal the input data type and it is retained by exact reference
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code value} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is {@code CLAMP}, the input is boolean, an
     *     integral request is {@code DIV} or {@code POW}, or the scalar and input types differ
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyScalar(Tensor input, ScalarElementwiseKind kind, ScalarValue value) {
        return applyScalarChecked(input, kind, value, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating or integral tensor retained by exact reference in result
     *     provenance
     * @param kind non-null scalar kind other than {@code CLAMP}, retained in the result operation
     * @param value non-null exact scalar arithmetic parameter; its data type must
     *     equal the input data type and it is retained by exact reference
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code value} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if {@code kind} is {@code CLAMP}, the input is boolean, an
     *     integral request is {@code DIV} or {@code POW}, or the scalar and input types differ
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyScalarChecked(Tensor input, ScalarElementwiseKind kind, ScalarValue value,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyScalarDescriptors(
                input == null ? null : input.descriptor(),
                kind,
                value,
                descriptorCheck,
                input);
    }

    /**
     * Creates one fresh derived tensor for an inclusive two-bound clamp request.
     *
     * <p>The input and bounds are null-checked in parameter order before the input type is
     * validated as floating. The bounds are then validated and retained in one
     * {@link ClampRangeAttrs}, so a non-floating input failure precedes range compatibility and
     * ordering checks. Exact common bound/input type equality is checked next. One operation with
     * {@link ScalarElementwiseKind#CLAMP} is then passed to common result construction.</p>
     *
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param minValue non-null exact inclusive lower bound retained by reference
     * @param maxValue non-null exact inclusive upper bound of the same type, retained by reference
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input}, {@code minValue}, or {@code maxValue} is
     *     null, checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the input data type is not floating, the bounds have
     *     different types or BOOL type, the range is inverted, or the bounds and input types
     *     differ
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyClamp(Tensor input, ScalarValue minValue, ScalarValue maxValue) {
        return applyClampChecked(input, minValue, maxValue, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param minValue non-null exact inclusive lower bound retained by reference
     * @param maxValue non-null exact inclusive upper bound of the same type, retained by reference
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code minValue}, or {@code maxValue} is
     *     null, checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input data type is not floating, the bounds have
     *     different types or BOOL type, the range is inverted, or the bounds and input types
     *     differ
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyClampChecked(Tensor input, ScalarValue minValue, ScalarValue maxValue,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyClampDescriptors(
                input == null ? null : input.descriptor(),
                minValue,
                maxValue,
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
        if (inputs.size() != 1 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        if (operation.attrs() instanceof ClampRangeAttrs attrs) {
            applyClampDescriptors(inputs.get(0), attrs.minValue(), attrs.maxValue(), sink, null);
        } else {
            applyScalarDescriptors(inputs.get(0), (ScalarElementwiseKind) operation.kind(),
                    ((ScalarValueAttrs) operation.attrs()).value(), sink, null);
        }
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param kind ordinary checked construction parameter
     * @param value ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyScalarDescriptors(
            TensorDescriptor input,
            ScalarElementwiseKind kind,
            ScalarValue value,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(value, "value");
        if (kind == ScalarElementwiseKind.CLAMP) {
            throw new IllegalArgumentException("CLAMP requires ClampRangeAttrs");
        }

        DataType dataType = input.dataType();
        if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must be a numeric data type, but was " + dataType);
        }
        if (dataType.isIntegral()
                && (kind == ScalarElementwiseKind.DIV || kind == ScalarElementwiseKind.POW)) {
            throw new IllegalArgumentException(kind + " does not support integral data types");
        }
        if (value.dataType() != dataType) {
            throw new IllegalArgumentException(
                    "scalar data type " + value.dataType()
                            + " must match input data type " + dataType);
        }

        ScalarValueAttrs attrs = new ScalarValueAttrs(value);
        Operation operation = new Operation(kind, attrs);
        return createDescriptors(
                input,
                dataType,
                operation,
                descriptorCheck,
                originalInput);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param minValue ordinary checked construction parameter
     * @param maxValue ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyClampDescriptors(
            TensorDescriptor input,
            ScalarValue minValue,
            ScalarValue maxValue,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(minValue, "minValue");
        Objects.requireNonNull(maxValue, "maxValue");

        DataType dataType = input.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        ClampRangeAttrs attrs = new ClampRangeAttrs(minValue, maxValue);
        if (attrs.minValue().dataType() != dataType) {
            throw new IllegalArgumentException(
                    "clamp data type " + attrs.minValue().dataType()
                            + " must match input data type " + dataType);
        }
        Operation operation = new Operation(ScalarElementwiseKind.CLAMP, attrs);
        return createDescriptors(
                input,
                dataType,
                operation,
                descriptorCheck,
                originalInput);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param dataType ordinary checked construction parameter
     * @param operation ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            DataType dataType,
            Operation operation,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                input.shape(),
                Optional.empty(),
                input.requiresGrad());
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(originalInput));
    }
}
