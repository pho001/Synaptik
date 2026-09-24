package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Objects;

/**
 * Reports the exact operation-occurrence capability of the current Metal backend.
 *
 * <p>This provider is immutable and performs no native loading, device discovery, allocation,
 * registration, or caching. Under {@link NumericalProfile#STRICT_IEEE}, support is
 * exactly unary {@code NEG}, five FLOAT32 affine transforms, and the explicit {@code CONTIGUOUS}
 * canonicalization barrier. Under {@link NumericalProfile#ACCELERATOR}, support is exactly tensor
 * {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV}; every strict baseline operation remains
 * closed. Binary inputs and outputs are canonical dense non-views with exact right-aligned
 * broadcasting. Strict {@code NEG} descriptors remain canonical. A strict affine or contiguous
 * input may be canonical or an exact resolved zero-offset logical view; complete-partition
 * analysis authenticates every admitted view as a prior local affine result. Affine outputs
 * retain the exact inferred view descriptor, while {@code CONTIGUOUS} outputs are canonical.
 * Every admitted occurrence is fully static, has positive rank-1..16 checked geometry, and
 * preserves one common gradient-eligibility flag.</p>
 */
public final class MetalCapabilityProvider implements BackendCapabilityProvider {
    /**
     * Stable Planning ownership identity for the Metal backend.
     *
     * <p>The immutable value is shared by every provider instance and says nothing about device
     * availability or executable readiness.</p>
     */
    public static final BackendId METAL_BACKEND_ID = new BackendId("metal");

    /**
     * Creates a stateless, immutable, and thread-safe Metal capability provider.
     *
     * <p>Construction performs no native loading, device discovery, registration, allocation,
     * or caching.</p>
     */
    public MetalCapabilityProvider() {}

    /**
     * Returns the stable Metal ownership identity without probing native state.
     *
     * @return the exact shared {@link #METAL_BACKEND_ID} instance; never {@code null}
     */
    @Override
    public BackendId backendId() {
        return METAL_BACKEND_ID;
    }

    /**
     * Reports support only for the exact profile-qualified prepared Metal domain.
     *
     * @param query the non-null immutable operation occurrence to classify without probing a
     *     device or native library
     * @return {@code true} exactly for a retained strict baseline occurrence or one of the four
     *     accelerator binary occurrences
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        return supportsOccurrence(
                query.numericalProfile(), query.operation(), query.inputs(), query.outputs());
    }

    /**
     * Validates one projected occurrence against the same profile-qualified domain used by
     * Planning.
     *
     * @param numericalProfile non-null cold graph-wide numerical-profile identity
     * @param operation non-null typed operation
     * @param inputs non-null ordered input descriptors
     * @param outputs non-null ordered output descriptors
     * @return whether the occurrence is supported
     */
    static boolean supportsOccurrence(
            NumericalProfile numericalProfile,
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        Objects.requireNonNull(numericalProfile, "numericalProfile");
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        if (outputs.size() != 1) {
            return false;
        }
        TensorDescriptor output = outputs.getFirst();
        try {
            if (numericalProfile == NumericalProfile.ACCELERATOR) {
                return supportsBinary(operation, inputs, output);
            }
            if (operation.kind() == UnaryElementwiseKind.NEG) {
                if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) {
                    return false;
                }
                TensorDescriptor input = inputs.getFirst();
                return canonical(input)
                        && canonical(output)
                        && input.shape().equals(output.shape())
                        && input.requiresGrad() == output.requiresGrad();
            }
            if (operation.kind() == ContiguousKind.CONTIGUOUS) {
                return supportsContiguous(operation, inputs, output);
            }
            return supportsAffine(operation, inputs, output);
        } catch (IllegalArgumentException | ArithmeticException incompatible) {
            return false;
        }
    }

    private static boolean supportsBinary(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.kind() instanceof BinaryArithmeticKind binary)
                || operation.attrs() != NoOperationAttrs.INSTANCE
                || (binary != BinaryArithmeticKind.ADD
                        && binary != BinaryArithmeticKind.SUB
                        && binary != BinaryArithmeticKind.MUL
                        && binary != BinaryArithmeticKind.DIV)
                || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        return canonical(left)
                && canonical(right)
                && canonical(output)
                && left.requiresGrad() == right.requiresGrad()
                && left.requiresGrad() == output.requiresGrad()
                && ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape());
    }

    private static boolean supportsAffine(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!affineInput(input)
                || !geometry(output)
                || input.requiresGrad() != output.requiresGrad()) {
            return false;
        }
        LayoutDescriptor inputLayout = input.layout().orElseThrow();
        LayoutDescriptor expected;
        if (operation.kind() instanceof ShapeTransformKind kind) {
            if (!(operation.attrs() instanceof TargetShapeAttrs attrs)
                    || !attrs.targetShape().equals(output.shape())) {
                return false;
            }
            if (kind == ShapeTransformKind.RESHAPE) {
                if (!inputLayout.isContiguous()) {
                    return false;
                }
                if (input.shape().knownElementCount().orElseThrow()
                        != output.shape().knownElementCount().orElseThrow()) {
                    return false;
                }
                expected = LayoutDescriptor.of(
                        output.shape(),
                        LayoutDescriptor.contiguous(output.shape()).strides(),
                        0L,
                        true);
            } else if (kind == ShapeTransformKind.EXPAND) {
                long[] inputShape = input.shape().toLongArray();
                long[] outputShape = output.shape().toLongArray();
                if (inputShape.length > outputShape.length) {
                    return false;
                }
                long[] strides = new long[outputShape.length];
                int padding = outputShape.length - inputShape.length;
                for (int axis = 0; axis < inputShape.length; axis++) {
                    long source = inputShape[axis];
                    long target = outputShape[axis + padding];
                    if (source != target && source != 1L) {
                        return false;
                    }
                    strides[axis + padding] = source == 1L && target != 1L
                            ? 0L : inputLayout.stride(axis);
                }
                expected = LayoutDescriptor.of(output.shape(), strides, 0L, true);
            } else {
                return false;
            }
        } else if (operation.kind() instanceof AxisTransformKind kind) {
            long[] inputShape = input.shape().toLongArray();
            if (kind == AxisTransformKind.PERMUTE) {
                if (!(operation.attrs() instanceof PermutationAttrs attrs)
                        || attrs.axes().size() != inputShape.length
                        || output.shape().rank() != inputShape.length) {
                    return false;
                }
                boolean[] seen = new boolean[inputShape.length];
                long[] expectedShape = new long[inputShape.length];
                long[] strides = new long[inputShape.length];
                for (int axis = 0; axis < inputShape.length; axis++) {
                    int source = attrs.axes().get(axis);
                    if (source < 0 || source >= inputShape.length || seen[source]) {
                        return false;
                    }
                    seen[source] = true;
                    expectedShape[axis] = inputShape[source];
                    strides[axis] = inputLayout.stride(source);
                }
                if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) {
                    return false;
                }
                expected = LayoutDescriptor.of(output.shape(), strides, 0L, true);
            } else {
                if (!(operation.attrs() instanceof AxisTransformAttrs attrs)) {
                    return false;
                }
                int axis = attrs.axis();
                if (kind == AxisTransformKind.EXPAND_DIMS) {
                    if (axis < 0 || axis > inputShape.length
                            || output.shape().rank() != inputShape.length + 1) {
                        return false;
                    }
                    long[] expectedShape = new long[inputShape.length + 1];
                    long[] strides = new long[inputShape.length + 1];
                    for (int outputAxis = 0; outputAxis < expectedShape.length; outputAxis++) {
                        if (outputAxis == axis) {
                            expectedShape[outputAxis] = 1L;
                            strides[outputAxis] = outputAxis == inputShape.length
                                    ? 1L
                                    : Math.multiplyExact(
                                            inputLayout.stride(outputAxis),
                                            inputShape[outputAxis]);
                        } else {
                            int source = outputAxis < axis ? outputAxis : outputAxis - 1;
                            expectedShape[outputAxis] = inputShape[source];
                            strides[outputAxis] = inputLayout.stride(source);
                        }
                    }
                    if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) {
                        return false;
                    }
                    expected = LayoutDescriptor.of(output.shape(), strides, 0L, true);
                } else if (kind == AxisTransformKind.SQUEEZE) {
                    if (axis < 0 || axis >= inputShape.length || inputShape[axis] != 1L
                            || output.shape().rank() != inputShape.length - 1) {
                        return false;
                    }
                    long[] expectedShape = new long[inputShape.length - 1];
                    long[] strides = new long[inputShape.length - 1];
                    for (int source = 0, target = 0; source < inputShape.length; source++) {
                        if (source != axis) {
                            expectedShape[target] = inputShape[source];
                            strides[target++] = inputLayout.stride(source);
                        }
                    }
                    if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) {
                        return false;
                    }
                    expected = LayoutDescriptor.of(output.shape(), strides, 0L, true);
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
        return output.layout().orElseThrow().equals(expected);
    }

    private static boolean supportsContiguous(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return affineInput(input)
                && canonical(output)
                && input.shape().equals(output.shape())
                && input.requiresGrad() == output.requiresGrad();
    }

    private static boolean affineInput(TensorDescriptor descriptor) {
        if (!geometry(descriptor)) {
            return false;
        }
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        return layout.storageOffset() == 0L
                && (layout.equals(LayoutDescriptor.contiguous(descriptor.shape()))
                        || layout.isView());
    }

    private static boolean canonical(TensorDescriptor descriptor) {
        return geometry(descriptor)
                && descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()));
    }

    private static boolean geometry(TensorDescriptor descriptor) {
        if (descriptor.dataType() != DataType.FLOAT32
                || !descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() < 1 || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) {
                return false;
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        Math.multiplyExact(elements, Float.BYTES);
        return true;
    }

}
