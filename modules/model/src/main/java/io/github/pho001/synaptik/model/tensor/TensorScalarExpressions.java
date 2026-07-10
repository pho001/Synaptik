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
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(value, "value");
        if (kind == ScalarElementwiseKind.CLAMP) {
            throw new IllegalArgumentException("CLAMP requires ClampRangeAttrs");
        }

        DataType dataType = input.descriptor().dataType();
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
        return create(input, dataType, operation);
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
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(minValue, "minValue");
        Objects.requireNonNull(maxValue, "maxValue");

        DataType dataType = input.descriptor().dataType();
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
        return create(input, dataType, operation);
    }

    /**
     * Creates the common descriptor, one-input provenance, and derived tensor result.
     *
     * <p>The descriptor retains {@code dataType}, the exact input shape reference, and unchanged
     * gradient eligibility while leaving layout unresolved. Provenance retains the exact supplied
     * operation and input reference. The central factory is invoked exactly once with no label;
     * it supplies the fresh identity and storage-free result.</p>
     *
     * @param input non-null source tensor whose shape, gradient eligibility, and exact reference
     *     are retained without mutation
     * @param dataType non-null validated floating or integral input data type retained exactly
     * @param operation non-null fully constructed scalar operation retained exactly in provenance
     * @return the non-null fresh, unlabeled, storage-free derived tensor returned by the factory
     * @throws NullPointerException if an internal caller violates a documented non-null argument
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    private static Tensor create(Tensor input, DataType dataType, Operation operation) {
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                input.descriptor().shape(),
                Optional.empty(),
                input.descriptor().requiresGrad());
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
