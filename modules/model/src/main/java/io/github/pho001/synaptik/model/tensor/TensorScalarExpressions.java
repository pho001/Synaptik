package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
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
 * multiplication, scalar exponentiation, and inclusive clamp requests. It accepts only floating
 * inputs, retains the input's exact data type and shape reference, preserves gradient eligibility,
 * and records the supplied binary64 parameters unchanged in typed operation attributes. It does
 * not inspect values or storage, convert parameters to an input format, evaluate mathematics,
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
     * <p>Validation occurs in this exact order: null-check {@code input} and {@code kind}, reject
     * {@link ScalarElementwiseKind#CLAMP} because it requires range attributes, then validate the
     * input's floating data type. The exact primitive bits are retained in one
     * {@link ScalarValueAttrs}, which is paired with the exact supplied kind before common result
     * construction. A failed validation allocates no tensor identity.</p>
     *
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param kind non-null scalar kind other than {@code CLAMP}, retained in the result operation
     * @param value binary64 multiplier, exponent, or single clamp bound retained without
     *     conversion or normalization, including signed zero, infinity, and NaN payload bits
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is {@code CLAMP}, or if the input data type
     *     is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyScalar(Tensor input, ScalarElementwiseKind kind, double value) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        if (kind == ScalarElementwiseKind.CLAMP) {
            throw new IllegalArgumentException("CLAMP requires ClampRangeAttrs");
        }

        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        ScalarValueAttrs attrs = new ScalarValueAttrs(value);
        Operation operation = new Operation(kind, attrs);
        return create(input, dataType, operation);
    }

    /**
     * Creates one fresh derived tensor for an inclusive two-bound clamp request.
     *
     * <p>The input is null-checked before its exact data type is read and validated as floating.
     * Only then are the supplied bounds retained in one {@link ClampRangeAttrs}; consequently a
     * non-floating input failure precedes range validation. The attributes constructor rejects
     * only a strictly inverted primitive range and otherwise preserves both binary64 values,
     * including equal bounds, signed zeros, infinities, and NaNs. One operation with
     * {@link ScalarElementwiseKind#CLAMP} is then passed to common result construction.</p>
     *
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param minValue inclusive lower bound retained without conversion or normalization
     * @param maxValue inclusive upper bound retained without conversion or normalization
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input} is null, with {@code input} as the message
     * @throws IllegalArgumentException if the input data type is not floating, or if
     *     {@code minValue > maxValue}
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyClamp(Tensor input, double minValue, double maxValue) {
        Objects.requireNonNull(input, "input");

        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        ClampRangeAttrs attrs = new ClampRangeAttrs(minValue, maxValue);
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
     * @param dataType non-null validated floating input data type retained exactly
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
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
