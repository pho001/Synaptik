package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
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
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Optional;

/**
 * Derives descriptors and validates operands for elementwise, scalar, comparison, logical,
 * classification, selection, and cast operation families.
 */
final class ElementwiseInference {
    private ElementwiseInference() {}

    /**
     * Verifies one supported elementwise-family occurrence.
     *
     * @param operation non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @return immutable derived descriptors and candidate constraints; never {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or an operand violates its
     *     family contract
     */
    static CapturedGraphInference.InferenceResult infer(Operation operation, List<TensorDescriptor> in) {
        var kind = operation.kind();
        if (kind instanceof BinaryArithmeticKind binary) return binary(binary, in);
        if (kind instanceof ScalarElementwiseKind scalar) return scalar(scalar, operation.attrs(), in);
        if (kind instanceof UnaryElementwiseKind) return unaryFloating(in);
        if (kind instanceof BinaryComparisonKind) return comparison(in);
        if (kind instanceof BooleanLogicalKind logical) return logical(logical, in);
        if (kind instanceof FloatingClassificationKind) return classification(in);
        if (kind instanceof WhereSelectionKind) return where(in);
        if (kind instanceof CastKind) return cast((CastAttrs) operation.attrs(), in);
        throw new IllegalArgumentException("unsupported elementwise kind");
    }

    private static CapturedGraphInference.InferenceResult binary(BinaryArithmeticKind kind, List<TensorDescriptor> in) {
        TensorDescriptor l = in.get(0), r = in.get(1);
        DataType type = DataTypePromotion.promoteNumeric(l.dataType(), r.dataType());
        if (type.isIntegral() && (kind == BinaryArithmeticKind.DIV || kind == BinaryArithmeticKind.POW))
            throw new IllegalArgumentException(kind + " does not support integral data types");
        Shape shape = ShapeBroadcast.broadcast(l.shape(), r.shape());
        return CapturedGraphInference.InferenceResult.of(descriptor(type, shape, l.requiresGrad() || r.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult scalar(
            ScalarElementwiseKind kind, Object attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); DataType type = input.dataType();
        if (!type.isFloating() && !type.isIntegral()) throw new IllegalArgumentException("input must be numeric");
        if (kind == ScalarElementwiseKind.CLAMP) {
            if (!(attrs instanceof ClampRangeAttrs range)) throw new IllegalArgumentException("CLAMP requires range attributes");
            if (!type.isFloating()) throw new IllegalArgumentException("CLAMP requires floating input");
            if (range.minValue().dataType() != type || range.maxValue().dataType() != type)
                throw new IllegalArgumentException("clamp scalar data type must match input");
        } else {
            if (!(attrs instanceof ScalarValueAttrs value)) throw new IllegalArgumentException("scalar kind requires scalar attributes");
            if (type.isIntegral() && (kind == ScalarElementwiseKind.DIV || kind == ScalarElementwiseKind.POW))
                throw new IllegalArgumentException(kind + " does not support integral data types");
            if (value.value().dataType() != type) throw new IllegalArgumentException("scalar data type must match input");
        }
        return CapturedGraphInference.InferenceResult.of(descriptor(type, input.shape(), input.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult unaryFloating(List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); requireFloating(input, "input");
        return CapturedGraphInference.InferenceResult.of(descriptor(input.dataType(), input.shape(), input.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult comparison(List<TensorDescriptor> in) {
        TensorDescriptor l = in.get(0), r = in.get(1);
        DataTypePromotion.promoteNumeric(l.dataType(), r.dataType());
        return CapturedGraphInference.InferenceResult.of(descriptor(DataType.BOOL,
                ShapeBroadcast.broadcast(l.shape(), r.shape()), false));
    }

    private static CapturedGraphInference.InferenceResult logical(BooleanLogicalKind kind, List<TensorDescriptor> in) {
        TensorDescriptor l = in.get(0); requireBool(l, "input[0]");
        if (kind == BooleanLogicalKind.NOT)
            return CapturedGraphInference.InferenceResult.of(descriptor(DataType.BOOL, l.shape(), false));
        TensorDescriptor r = in.get(1); requireBool(r, "input[1]");
        return CapturedGraphInference.InferenceResult.of(descriptor(DataType.BOOL,
                ShapeBroadcast.broadcast(l.shape(), r.shape()), false));
    }

    private static CapturedGraphInference.InferenceResult classification(List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); requireFloating(input, "input");
        return CapturedGraphInference.InferenceResult.of(descriptor(DataType.BOOL, input.shape(), false));
    }

    private static CapturedGraphInference.InferenceResult where(List<TensorDescriptor> in) {
        TensorDescriptor condition = in.get(0), t = in.get(1), f = in.get(2); requireBool(condition, "condition");
        DataType type = DataTypePromotion.promoteFloating(t.dataType(), f.dataType());
        Shape branches = ShapeBroadcast.broadcast(t.shape(), f.shape());
        return CapturedGraphInference.InferenceResult.of(descriptor(type,
                ShapeBroadcast.broadcast(condition.shape(), branches), t.requiresGrad() || f.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult cast(CastAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); DataType target = attrs.targetDataType();
        boolean grad = input.requiresGrad() && input.dataType().isFloating() && target.isFloating();
        return CapturedGraphInference.InferenceResult.of(descriptor(target, input.shape(), grad));
    }

    /**
     * Creates the common unresolved-layout descriptor used by semantic, non-view results.
     *
     * @param type non-null result data type
     * @param shape non-null result Shape
     * @param grad whether the result remains gradient-eligible
     * @return a new unresolved-layout descriptor; never {@code null}
     */
    static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.empty(), grad);
    }
    /**
     * Requires a floating descriptor.
     *
     * @param descriptor non-null descriptor to inspect
     * @param role non-null diagnostic operand role
     * @throws IllegalArgumentException if the descriptor is not floating
     */
    static void requireFloating(TensorDescriptor descriptor, String role) {
        if (!descriptor.dataType().isFloating()) throw new IllegalArgumentException(role + " must be floating");
    }
    /**
     * Requires a floating or signed-integral descriptor.
     *
     * @param descriptor non-null descriptor to inspect
     * @param role non-null diagnostic operand role
     * @throws IllegalArgumentException if the descriptor is not numeric
     */
    static void requireNumeric(TensorDescriptor descriptor, String role) {
        if (!descriptor.dataType().isFloating() && !descriptor.dataType().isIntegral())
            throw new IllegalArgumentException(role + " must be numeric");
    }
    /**
     * Requires an exact BOOL descriptor.
     *
     * @param descriptor non-null descriptor to inspect
     * @param role non-null diagnostic operand role
     * @throws IllegalArgumentException if the descriptor is not BOOL
     */
    static void requireBool(TensorDescriptor descriptor, String role) {
        if (descriptor.dataType() != DataType.BOOL) throw new IllegalArgumentException(role + " must be BOOL");
    }
    /**
     * Requires an exact INT32 or INT64 index descriptor.
     *
     * @param descriptor non-null descriptor to inspect
     * @param role non-null diagnostic operand role
     * @throws IllegalArgumentException if the descriptor is not an index type
     */
    static void requireIndex(TensorDescriptor descriptor, String role) {
        if (descriptor.dataType() != DataType.INT32 && descriptor.dataType() != DataType.INT64)
            throw new IllegalArgumentException(role + " must be INT32 or INT64");
    }
}
