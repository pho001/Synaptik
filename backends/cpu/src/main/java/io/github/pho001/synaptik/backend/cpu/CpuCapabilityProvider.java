package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the executable semantic coverage currently delivered by the CPU backend.
 *
 * <p>The provider has a stable CPU ownership identity and advertises the bounded, fully static
 * pointwise matrix implemented by the portable route: selected same-type arithmetic including
 * extrema and floating Tensor power, exact scalar arithmetic and floating range clamp,
 * canonical-BOOL logic, negation and classification, comparisons, floating {@code WHERE},
 * same-type {@code CAST}, and
 * exact {@code FLOAT64} {@code GELU}. Every descriptor has a resolved layout, and results obey the
 * Model family's shape rule. Complete-partition lowering remains stricter: it validates a connected
 * one-to-eight-occurrence chain, normalizes exact layout geometry, and applies alias, fan-out,
 * publication, and partition-boundary checks before resource declaration.</p>
 */
public final class CpuCapabilityProvider implements BackendCapabilityProvider {
    /** Stable Planning ownership identity for the CPU backend. */
    public static final BackendId CPU_BACKEND_ID = new BackendId("cpu");

    /** Creates a stateless, narrowly fail-closed CPU capability provider. */
    public CpuCapabilityProvider() {
    }

    /**
     * Returns the stable CPU ownership identity.
     *
     * @return {@link #CPU_BACKEND_ID} by exact reference; never {@code null}
     */
    @Override
    public BackendId backendId() {
        return CPU_BACKEND_ID;
    }

    /**
     * Reports whether an occurrence belongs to the exact implemented semantic set.
     * Binary and comparison results must equal the current right-aligned broadcast result;
     * unary, classification, scalar-arithmetic, range-clamp, logical-NOT, and same-type-cast
     * results preserve shape; binary logical rows use the same right-aligned broadcast rule;
     * {@code WHERE} applies branch-first then condition broadcasting. Every descriptor must be
     * fully static with a resolved layout. Cross-type casts and all rows outside the implemented
     * matrix return {@code false} without defining conversion or fallback behavior.
     *
     * @param query non-null immutable operation occurrence to validate structurally
     * @return {@code true} only for the exact implemented occurrence-local matrix; otherwise
     *     {@code false}; complete-partition eligibility may still be stricter
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.outputs().size() != 1 || !query.inputs().stream().allMatch(CpuCapabilityProvider::staticResolved)
                || !query.outputs().stream().allMatch(CpuCapabilityProvider::staticResolved)) {
            return false;
        }
        var kind = query.operation().kind();
        var attrs = query.operation().attrs();
        TensorDescriptor output = query.outputs().getFirst();
        try {
            if (kind instanceof BinaryArithmeticKind arithmetic) {
                return attrs == NoOperationAttrs.INSTANCE
                        && (arithmetic == BinaryArithmeticKind.ADD || arithmetic == BinaryArithmeticKind.SUB
                            || arithmetic == BinaryArithmeticKind.MUL
                            || arithmetic == BinaryArithmeticKind.MIN
                            || arithmetic == BinaryArithmeticKind.MAX
                            || arithmetic == BinaryArithmeticKind.POW
                                && floating(output.dataType())
                            || arithmetic == BinaryArithmeticKind.DIV
                                && floating(output.dataType()))
                        && sameNumeric(query.inputs(), output)
                        && broadcast(query.inputs().get(0), query.inputs().get(1), output);
            }
            if (kind instanceof ScalarElementwiseKind scalar) {
                if (scalar == ScalarElementwiseKind.CLAMP) {
                    return attrs instanceof ClampRangeAttrs range
                            && query.inputs().size() == 1 && floating(output.dataType())
                            && sameTypeAndShape(query.inputs().getFirst(), output)
                            && range.minValue().dataType() == output.dataType()
                            && range.maxValue().dataType() == output.dataType();
                }
                return attrs instanceof ScalarValueAttrs value
                        && (scalar == ScalarElementwiseKind.ADD || scalar == ScalarElementwiseKind.SUB
                            || scalar == ScalarElementwiseKind.MUL
                            || scalar == ScalarElementwiseKind.MIN
                            || scalar == ScalarElementwiseKind.MAX
                            || (scalar == ScalarElementwiseKind.DIV
                                || scalar == ScalarElementwiseKind.POW)
                                && floating(output.dataType()))
                        && query.inputs().size() == 1 && supportedNumeric(query.inputs().getFirst().dataType())
                        && sameTypeAndShape(query.inputs().getFirst(), output)
                        && value.value().dataType() == output.dataType();
            }
            if (kind instanceof BooleanLogicalKind logical) {
                if (attrs != NoOperationAttrs.INSTANCE || output.dataType() != DataType.BOOL) return false;
                if (logical == BooleanLogicalKind.NOT) {
                    return query.inputs().size() == 1
                            && query.inputs().getFirst().dataType() == DataType.BOOL
                            && output.shape().equals(query.inputs().getFirst().shape());
                }
                return query.inputs().size() == 2
                        && query.inputs().stream().allMatch(input -> input.dataType() == DataType.BOOL)
                        && broadcast(query.inputs().get(0), query.inputs().get(1), output);
            }
            if (kind instanceof UnaryElementwiseKind unary) {
                return attrs == NoOperationAttrs.INSTANCE && query.inputs().size() == 1
                        && ((unary == UnaryElementwiseKind.NEG
                                && floating(query.inputs().getFirst().dataType()))
                            || (unary == UnaryElementwiseKind.GELU
                                && query.inputs().getFirst().dataType() == DataType.FLOAT64))
                        && sameTypeAndShape(query.inputs().getFirst(), output);
            }
            if (kind instanceof FloatingClassificationKind) {
                return attrs == NoOperationAttrs.INSTANCE && query.inputs().size() == 1
                        && floating(query.inputs().getFirst().dataType())
                        && output.dataType() == DataType.BOOL
                        && output.shape().equals(query.inputs().getFirst().shape());
            }
            if (kind instanceof BinaryComparisonKind) {
                return attrs == NoOperationAttrs.INSTANCE && query.inputs().size() == 2
                        && sameNumericInputs(query.inputs()) && output.dataType() == DataType.BOOL
                        && ShapeBroadcast.broadcast(query.inputs().get(0).shape(),
                                query.inputs().get(1).shape()).equals(output.shape());
            }
            if (kind == WhereSelectionKind.WHERE) {
                if (attrs != NoOperationAttrs.INSTANCE || query.inputs().size() != 3
                        || query.inputs().get(0).dataType() != DataType.BOOL) return false;
                TensorDescriptor whenTrue = query.inputs().get(1);
                TensorDescriptor whenFalse = query.inputs().get(2);
                if (!floating(whenTrue.dataType()) || whenTrue.dataType() != whenFalse.dataType()
                        || output.dataType() != whenTrue.dataType()) return false;
                var branches = ShapeBroadcast.broadcast(whenTrue.shape(), whenFalse.shape());
                return ShapeBroadcast.broadcast(query.inputs().get(0).shape(), branches)
                        .equals(output.shape());
            }
            if (kind == CastKind.CAST) {
                return attrs instanceof CastAttrs cast && query.inputs().size() == 1
                        && supportedCast(query.inputs().getFirst().dataType())
                        && cast.targetDataType() == query.inputs().getFirst().dataType()
                        && sameTypeAndShape(query.inputs().getFirst(), output);
            }
        } catch (IllegalArgumentException incompatible) { return false; }
        return false;
    }

    private static boolean staticResolved(TensorDescriptor descriptor) {
        return descriptor.shape().isFullyStatic() && descriptor.layout().isPresent();
    }

    private static boolean floating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32;
    }

    private static boolean supportedNumeric(DataType type) {
        return floating(type) || type == DataType.INT32 || type == DataType.INT64;
    }

    private static boolean supportedCast(DataType type) {
        return supportedNumeric(type) || type == DataType.BOOL;
    }

    private static boolean sameNumeric(java.util.List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        return inputs.size() == 2 && sameNumericInputs(inputs)
                && output.dataType() == inputs.getFirst().dataType();
    }

    private static boolean sameNumericInputs(java.util.List<TensorDescriptor> inputs) {
        return inputs.size() == 2 && supportedNumeric(inputs.getFirst().dataType())
                && inputs.get(1).dataType() == inputs.getFirst().dataType();
    }

    private static boolean sameTypeAndShape(TensorDescriptor input, TensorDescriptor output) {
        return input.dataType() == output.dataType() && input.shape().equals(output.shape());
    }

    private static boolean broadcast(TensorDescriptor left, TensorDescriptor right,
            TensorDescriptor output) {
        return ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape());
    }
}
