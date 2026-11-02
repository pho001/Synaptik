package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Arrays;
import java.util.List;

/**
 * Cold admission/lowering and independent encoded validation for non-hardware exact operations.
 *
 * <p>Only original FLOAT64 and integral occurrences are selected. Floating extrema widen within
 * their Model category; integral arithmetic and extrema never use a floating proxy. Existing
 * FLOAT32 and low routes retain their independent admission. Outputs are canonical; reads may be
 * positive affine or explicit zero-stride views. Comparisons publish exact nondifferentiable BOOL
 * results even when floating inputs require gradients. No FP64 numerical arithmetic is introduced.</p>
 */
final class MetalExactSupport {
    private MetalExactSupport() {}

    /**
     * Selects this owner's kinds and original carriers without claiming their validity.
     * @param operation non-null Model occurrence
     * @param inputs ordered original input descriptors
     * @param outputs ordered original result descriptors
     * @return whether independent exact validation owns this occurrence
     */
    static boolean handles(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        return kind(operation) != null && !inputs.isEmpty()
                && inputs.stream().anyMatch(input -> exactCarrier(input.dataType()));
    }

    /**
     * Checks a Model occurrence's attributes, promotion, gradients and physical mapping.
     * @param operation non-null Model occurrence
     * @param inputs ordered original input descriptors
     * @param outputs ordered original result descriptors
     * @return true only for the complete exact kernel domain
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        try {
            var kind = kind(operation);
            if (!handles(operation, inputs, outputs) || outputs.size() != 1) return false;
            TensorDescriptor output = outputs.getFirst();
            if (!output.shape().isFullyStatic() || !physical(output.shape().toLongArray(),
                    output.layout().orElse(null), output.dataType(), true)) return false;
            for (var input : inputs) {
                if (!input.shape().isFullyStatic() || !physical(input.shape().toLongArray(),
                        input.layout().orElse(null), input.dataType(), false)) return false;
            }
            TensorDescriptor left = inputs.getFirst();
            boolean grad = inputs.stream().anyMatch(TensorDescriptor::requiresGrad);
            if (output.requiresGrad() != (!comparison(kind) && grad)) return false;
            if (unary(kind)) return inputs.size() == 1 && operation.attrs() == NoOperationAttrs.INSTANCE
                    && left.dataType() == DataType.FLOAT64 && output.dataType() == DataType.FLOAT64
                    && left.shape().equals(output.shape());
            if (binary(kind)) {
                if (inputs.size() != 2 || operation.attrs() != NoOperationAttrs.INSTANCE) return false;
                TensorDescriptor right = inputs.get(1);
                DataType promoted = DataTypePromotion.promoteNumeric(left.dataType(), right.dataType());
                return allowedBinary(kind, left.dataType(), right.dataType(), promoted)
                        && output.dataType() == (comparison(kind) ? DataType.BOOL : promoted)
                        && Arrays.equals(broadcast(left.shape().toLongArray(), right.shape().toLongArray()),
                                output.shape().toLongArray());
            }
            if (inputs.size() != 1 || output.dataType() != left.dataType()
                    || !left.shape().equals(output.shape()) || !allowedScalar(kind, left.dataType())) return false;
            if (kind == MetalMpsGraphProgram.NodeKind.CLAMP) {
                return operation.attrs() instanceof ClampRangeAttrs attrs
                        && attrs.minValue().dataType() == left.dataType()
                        && attrs.maxValue().dataType() == left.dataType();
            }
            return operation.attrs() instanceof ScalarValueAttrs attrs
                    && attrs.value().dataType() == left.dataType();
        } catch (IllegalArgumentException | ArithmeticException invalid) {
            return false;
        }
    }

    /**
     * Encodes a supported occurrence using existing operation and scalar grammar.
     * @param operation Model occurrence
     * @param inputs ordered program value indices
     * @param outputs single program result index
     * @param inputDescriptors original input descriptors
     * @param outputDescriptors original output descriptors
     * @return exact fixed-kernel node
     * @throws IllegalArgumentException if the occurrence is outside this owner's domain
     */
    static MetalMpsGraphProgram.Node lower(Operation operation, int[] inputs, int[] outputs,
            List<TensorDescriptor> inputDescriptors, List<TensorDescriptor> outputDescriptors) {
        if (!supports(operation, inputDescriptors, outputDescriptors)
                || inputs.length != inputDescriptors.size() || outputs.length != 1)
            throw new IllegalArgumentException("Unsupported Metal exact occurrence");
        var kind = kind(operation);
        if (operation.attrs() instanceof ClampRangeAttrs attrs)
            return MetalMpsGraphProgram.Node.clamp(inputs[0], outputs[0], attrs.minValue().dataType(),
                    raw(attrs.minValue()), raw(attrs.maxValue()));
        if (operation.attrs() instanceof ScalarValueAttrs attrs)
            return MetalMpsGraphProgram.Node.scalarValue(kind, inputs[0], outputs[0], attrs.value().dataType(), raw(attrs.value()));
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs,
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    /**
     * Selects encoded original-carrier occurrences before independent validation.
     * @param node encoded node
     * @param values full ordered value table
     * @return whether an exact kind consumes an original FLOAT64/integral carrier
     */
    static boolean handles(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (!unary(node.kind()) && !binary(node.kind()) && !scalar(node.kind())) return false;
        for (int input : node.inputs())
            if (input >= 0 && input < values.size() && exactCarrier(values.get(input).dataType())) return true;
        return false;
    }

    /**
     * Re-derives encoded grammar, physical spans, promotion, gradients and result shape directly.
     * This boundary never calls Model/provider admission or a projected proxy predicate.
     * @param node raw encoded occurrence
     * @param values full ordered encoded value descriptors
     * @throws IllegalArgumentException for any malformed or unsupported occurrence
     */
    static void validate(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            int[] inputs = node.inputs(), outputs = node.outputs();
            if (!handles(node, values) || outputs.length != 1
                    || inputs.length != (binary(node.kind()) ? 2 : 1)) throw invalid();
            for (int input : inputs) if (input < 0 || input >= values.size()) throw invalid();
            if (outputs[0] < 0 || outputs[0] >= values.size()) throw invalid();
            var output = values.get(outputs[0]);
            if (!physical(output.dimensions(), output.layout().orElse(null), output.dataType(), true)) throw invalid();
            boolean grad = false;
            for (int input : inputs) {
                var value = values.get(input);
                if (!physical(value.dimensions(), value.layout().orElse(null), value.dataType(), false)) throw invalid();
                if (value.requiresGrad() && !value.dataType().isDifferentiable()) throw invalid();
                grad |= value.requiresGrad();
            }
            if (output.requiresGrad() != (!comparison(node.kind()) && grad)
                    || output.requiresGrad() && !output.dataType().isDifferentiable()) throw invalid();
            var left = values.get(inputs[0]);
            long[] attrs = node.attributeWords();
            if (unary(node.kind())) {
                if (node.attributeKind() != MetalMpsGraphProgram.AttributeKind.NONE || attrs.length != 0
                        || left.dataType() != DataType.FLOAT64 || output.dataType() != DataType.FLOAT64
                        || !Arrays.equals(left.dimensions(), output.dimensions())) throw invalid();
            } else if (binary(node.kind())) {
                var right = values.get(inputs[1]);
                DataType promoted = DataTypePromotion.promoteNumeric(left.dataType(), right.dataType());
                if (node.attributeKind() != MetalMpsGraphProgram.AttributeKind.NONE || attrs.length != 0
                        || !allowedBinary(node.kind(), left.dataType(), right.dataType(), promoted)
                        || output.dataType() != (comparison(node.kind()) ? DataType.BOOL : promoted)
                        || !Arrays.equals(broadcast(left.dimensions(), right.dimensions()), output.dimensions())) throw invalid();
            } else {
                boolean clamp = node.kind() == MetalMpsGraphProgram.NodeKind.CLAMP;
                if (!allowedScalar(node.kind(), left.dataType()) || output.dataType() != left.dataType()
                        || !Arrays.equals(left.dimensions(), output.dimensions())
                        || node.attributeKind() != (clamp ? MetalMpsGraphProgram.AttributeKind.CLAMP_RANGE
                                : MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE)
                        || attrs.length != (clamp ? 4 : 2)) throw invalid();
                validateScalar(attrs[0], attrs[1], left.dataType());
                if (clamp) {
                    validateScalar(attrs[2], attrs[3], left.dataType());
                    if (left.dataType() == DataType.INT32 && (int) attrs[1] > (int) attrs[3]
                            || left.dataType() == DataType.INT64 && attrs[1] > attrs[3]
                            || left.dataType() == DataType.FLOAT64
                                    && Double.longBitsToDouble(attrs[1]) > Double.longBitsToDouble(attrs[3])) throw invalid();
                }
            }
        } catch (ArithmeticException | IndexOutOfBoundsException malformed) {
            throw new IllegalArgumentException("Malformed Metal exact occurrence", malformed);
        }
    }

    private static boolean physical(long[] dims, LayoutDescriptor layout, DataType type, boolean output) {
        if (dims.length > MetalMpsGraphProgram.MAX_RANK || layout == null || layout.rank() != dims.length) return false;
        long count = 1;
        for (long dim : dims) { if (dim <= 0) return false; count = Math.multiplyExact(count, dim); }
        Math.multiplyExact(count, type.byteWidth());
        Math.multiplyExact(layout.referencedElementSpan(), type.byteWidth());
        if (output) return layout.equals(LayoutDescriptor.contiguous(Shape.of(dims)));
        for (long stride : layout.strides()) if (stride < 0) return false;
        if (layout.hasZeroStride() && !layout.isView()) return false;
        // Broadcast axes do not excuse overlap among the remaining physical axes.
        return new MetalMpsGraphProgram.ValueDescriptor(type, dims, java.util.Optional.of(layout), false, false)
                .supportsStorageLayout(0);
    }

    private static long[] broadcast(long[] left, long[] right) {
        int rank = Math.max(left.length, right.length);
        long[] result = new long[rank];
        for (int i = 0; i < rank; i++) {
            long a = i < rank - left.length ? 1 : left[i - rank + left.length];
            long b = i < rank - right.length ? 1 : right[i - rank + right.length];
            if (a != b && a != 1 && b != 1) throw invalid();
            result[i] = Math.max(a, b);
        }
        return result;
    }

    private static boolean exactCarrier(DataType type) { return type == DataType.FLOAT64 || type.isIntegral(); }
    private static boolean allowedBinary(MetalMpsGraphProgram.NodeKind kind, DataType left, DataType right, DataType promoted) {
        if (comparison(kind) || kind == MetalMpsGraphProgram.NodeKind.TENSOR_MIN || kind == MetalMpsGraphProgram.NodeKind.TENSOR_MAX)
            return promoted.isIntegral() || promoted == DataType.FLOAT64;
        return left.isIntegral() && right.isIntegral() && promoted.isIntegral();
    }
    private static boolean allowedScalar(MetalMpsGraphProgram.NodeKind kind, DataType type) {
        return scalar(kind) && (type.isIntegral() || type == DataType.FLOAT64
                && (kind == MetalMpsGraphProgram.NodeKind.SCALAR_MIN
                        || kind == MetalMpsGraphProgram.NodeKind.SCALAR_MAX || kind == MetalMpsGraphProgram.NodeKind.CLAMP));
    }
    private static boolean unary(MetalMpsGraphProgram.NodeKind kind) {
        return switch (kind) { case NEG, ABS, FLOOR, CEIL, SIGN, RELU -> true; default -> false; };
    }
    private static boolean binary(MetalMpsGraphProgram.NodeKind kind) {
        return comparison(kind) || switch (kind) { case ADD, SUB, MUL, TENSOR_MIN, TENSOR_MAX -> true; default -> false; };
    }
    private static boolean comparison(MetalMpsGraphProgram.NodeKind kind) {
        return switch (kind) { case GT, GE, LT, LE, EQ, NE -> true; default -> false; };
    }
    private static boolean scalar(MetalMpsGraphProgram.NodeKind kind) {
        return switch (kind) { case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_MIN, SCALAR_MAX, CLAMP -> true; default -> false; };
    }
    private static MetalMpsGraphProgram.NodeKind kind(Operation operation) {
        if (operation.kind() instanceof UnaryElementwiseKind kind) return switch (kind) {
            case NEG -> MetalMpsGraphProgram.NodeKind.NEG; case ABS -> MetalMpsGraphProgram.NodeKind.ABS;
            case FLOOR -> MetalMpsGraphProgram.NodeKind.FLOOR; case CEIL -> MetalMpsGraphProgram.NodeKind.CEIL;
            case SIGN -> MetalMpsGraphProgram.NodeKind.SIGN; case RELU -> MetalMpsGraphProgram.NodeKind.RELU;
            default -> null;
        };
        if (operation.kind() instanceof BinaryComparisonKind kind) return switch (kind) {
            case GREATER_THAN -> MetalMpsGraphProgram.NodeKind.GT;
            case GREATER_OR_EQUAL -> MetalMpsGraphProgram.NodeKind.GE;
            case LESS_THAN -> MetalMpsGraphProgram.NodeKind.LT;
            case LESS_OR_EQUAL -> MetalMpsGraphProgram.NodeKind.LE;
            case EQUAL -> MetalMpsGraphProgram.NodeKind.EQ;
            case NOT_EQUAL -> MetalMpsGraphProgram.NodeKind.NE;
        };
        if (operation.kind() instanceof BinaryArithmeticKind kind) return switch (kind) {
            case ADD -> MetalMpsGraphProgram.NodeKind.ADD; case SUB -> MetalMpsGraphProgram.NodeKind.SUB;
            case MUL -> MetalMpsGraphProgram.NodeKind.MUL; case MIN -> MetalMpsGraphProgram.NodeKind.TENSOR_MIN;
            case MAX -> MetalMpsGraphProgram.NodeKind.TENSOR_MAX; default -> null;
        };
        if (operation.kind() instanceof ScalarElementwiseKind kind) return switch (kind) {
            case ADD -> MetalMpsGraphProgram.NodeKind.SCALAR_ADD; case SUB -> MetalMpsGraphProgram.NodeKind.SCALAR_SUB;
            case MUL -> MetalMpsGraphProgram.NodeKind.SCALAR_MUL; case MIN -> MetalMpsGraphProgram.NodeKind.SCALAR_MIN;
            case MAX -> MetalMpsGraphProgram.NodeKind.SCALAR_MAX; case CLAMP -> MetalMpsGraphProgram.NodeKind.CLAMP;
            default -> null;
        };
        return null;
    }
    private static void validateScalar(long wire, long word, DataType type) {
        if (wire != MetalMpsGraphProgram.dataTypeWire(type)
                || type == DataType.INT32 && (word & ~0xffff_ffffL) != 0) throw invalid();
    }
    private static long raw(ScalarValue value) {
        return switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case INT32 -> Integer.toUnsignedLong(value.int32Value());
            case INT64 -> value.int64Value();
            default -> throw invalid();
        };
    }
    private static IllegalArgumentException invalid() { return new IllegalArgumentException("Malformed Metal exact occurrence"); }
}
