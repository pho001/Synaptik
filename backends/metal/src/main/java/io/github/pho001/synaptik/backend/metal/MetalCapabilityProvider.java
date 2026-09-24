package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
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
 * registration, or caching. Under either numerical profile, support includes unary {@code NEG}
 * and {@code ABS}, five FLOAT32 affine transforms, the explicit {@code CONTIGUOUS}
 * canonicalization barrier, canonical positive-rank INT32 {@code GATHER} indices selecting
 * FLOAT32 data, canonical positive-rank INT32 {@code ONE_HOT} indices producing terminal BOOL
 * values, and canonical FLOAT32/INT32/FLOAT32 {@code SCATTER_ELEMENTS} replacement with unique
 * valid targets. {@link NumericalProfile#ACCELERATOR} additionally supports tensor {@code ADD}/
 * {@code SUB}/{@code MUL}/{@code DIV}, canonical FLOAT32 {@code SUM}/{@code MEAN}/
 * {@code SUM_TO_SHAPE}, and positive static rank-two FLOAT32 {@code MATMUL}. MATMUL accepts each
 * operand only as canonical or as the exact rank-two transpose layout that complete-partition
 * analysis must authenticate to a local {@code PERMUTE [1,0]} producer from a canonical source.
 * Its output is canonical and carries the logical OR of the operand gradient flags. Strict MATMUL
 * remains unsupported. Accelerator reductions admit only full, normalized single-axis, ordered
 * normalized multi-axis (including empty identity), and binding-resolved sum-to-Shape forms. Their
 * input is canonical positive-rank {@code 1..16}; canonical outputs may be rank zero only as
 * locally produced reduction results. Strict reductions remain unsupported. Binary inputs and
 * outputs are canonical dense non-views with exact right-aligned broadcasting. {@code ABS} and
 * {@code NEG} descriptors remain canonical. An affine or contiguous input may be canonical or an
 * exact resolved zero-offset logical view; complete-partition analysis authenticates every
 * admitted view as a prior local affine result. Every admitted occurrence uses checked positive
 * extents. GATHER requires its exact replacement-axis output formula and matched data/output
 * gradient flag; ONE_HOT appends its positive depth and is entirely non-differentiable.
 * SCATTER_ELEMENTS requires reduction NONE, equal indices/update Shapes, matching non-axis data
 * extents, exact data-shaped output, non-differentiable indices, and data/update gradient OR.</p>
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
     * @return {@code true} exactly for one occurrence in the complete strict or accelerator
     *     matrix
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
            if (operation.kind() == AxisGatherKind.GATHER) {
                return supportsGather(operation, inputs, output);
            }
            if (operation.kind() == AxisScatterKind.SCATTER_ELEMENTS) {
                return supportsScatterElements(operation, inputs, output);
            }
            if (operation.kind() == OneHotKind.ONE_HOT) {
                return supportsOneHot(operation, inputs, output);
            }
            if (operation.kind() == UnaryElementwiseKind.NEG
                    || operation.kind() == UnaryElementwiseKind.ABS) {
                return supportsCanonicalUnary(operation, inputs, output);
            }
            if (operation.kind() == ContiguousKind.CONTIGUOUS) {
                return supportsContiguous(operation, inputs, output);
            }
            if (operation.kind() instanceof ShapeTransformKind
                    || operation.kind() instanceof AxisTransformKind) {
                return supportsAffine(operation, inputs, output);
            }
            if (numericalProfile == NumericalProfile.ACCELERATOR) {
                if (operation.kind() instanceof AggregateReductionKind reduction) {
                    return supportsReduction(operation, inputs, output, reduction);
                }
                if (operation.kind() == MatmulKind.MATMUL) {
                    return supportsMatmul(operation, inputs, output);
                }
                return supportsBinary(operation, inputs, output);
            }
            return false;
        } catch (IllegalArgumentException | ArithmeticException incompatible) {
            return false;
        }
    }

    private static boolean supportsGather(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof IndexAxisAttrs attrs) || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        if (!canonical(data)
                || !canonicalTyped(indices, DataType.INT32)
                || !canonical(output)
                || indices.requiresGrad()
                || data.requiresGrad() != output.requiresGrad()) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        int axis = attrs.axis();
        int outputRank = Math.addExact(Math.subtractExact(dataShape.length, 1), indexShape.length);
        if (axis >= dataShape.length || outputRank < 1 || outputRank > 16
                || output.shape().rank() != outputRank) {
            return false;
        }
        long[] expected = new long[outputRank];
        System.arraycopy(dataShape, 0, expected, 0, axis);
        System.arraycopy(indexShape, 0, expected, axis, indexShape.length);
        System.arraycopy(dataShape, axis + 1, expected, axis + indexShape.length,
                dataShape.length - axis - 1);
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static boolean supportsScatterElements(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof ScatterElementsAttrs attrs)
                || attrs.reduction() != ScatterReduction.NONE
                || inputs.size() != 3) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        TensorDescriptor updates = inputs.get(2);
        if (!canonical(data)
                || !canonicalTyped(indices, DataType.INT32)
                || !canonical(updates)
                || !canonical(output)
                || indices.requiresGrad()
                || output.requiresGrad() != (data.requiresGrad() || updates.requiresGrad())
                || !indices.shape().equals(updates.shape())
                || !data.shape().equals(output.shape())) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        int axis = attrs.axis();
        if (dataShape.length != indexShape.length || axis >= dataShape.length) {
            return false;
        }
        for (int dimension = 0; dimension < dataShape.length; dimension++) {
            if (dimension != axis && dataShape[dimension] != indexShape[dimension]) {
                return false;
            }
        }
        return true;
    }

    private static boolean supportsOneHot(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof OneHotAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor indices = inputs.getFirst();
        if (!canonicalTyped(indices, DataType.INT32)
                || !canonicalTyped(output, DataType.BOOL)
                || indices.requiresGrad()
                || output.requiresGrad()
                || indices.shape().rank() >= 16) {
            return false;
        }
        long[] indexShape = indices.shape().toLongArray();
        long[] expected = java.util.Arrays.copyOf(indexShape, indexShape.length + 1);
        expected[indexShape.length] = attrs.depth();
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static boolean supportsCanonicalUnary(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonical(output)
                && input.shape().equals(output.shape())
                && input.requiresGrad() == output.requiresGrad();
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

    private static boolean supportsMatmul(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        if (!matmulInput(left)
                || !matmulInput(right)
                || !canonical(output)
                || left.shape().rank() != 2
                || right.shape().rank() != 2
                || output.shape().rank() != 2
                || output.requiresGrad() != (left.requiresGrad() || right.requiresGrad())) {
            return false;
        }
        long[] leftShape = left.shape().toLongArray();
        long[] rightShape = right.shape().toLongArray();
        long[] outputShape = output.shape().toLongArray();
        return leftShape[1] == rightShape[0]
                && outputShape[0] == leftShape[0]
                && outputShape[1] == rightShape[1];
    }

    private static boolean matmulInput(TensorDescriptor descriptor) {
        if (!geometry(descriptor) || descriptor.shape().rank() != 2) {
            return false;
        }
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        if (layout.equals(LayoutDescriptor.contiguous(descriptor.shape()))) {
            return true;
        }
        long[] shape = descriptor.shape().toLongArray();
        return layout.equals(LayoutDescriptor.of(
                descriptor.shape(), new long[] {1L, shape[0]}, 0L, true));
    }


    private static boolean supportsReduction(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            AggregateReductionKind kind) {
        if ((kind != AggregateReductionKind.SUM && kind != AggregateReductionKind.MEAN)
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonical(input) || !canonicalReductionOutput(output)
                || input.requiresGrad() != output.requiresGrad()) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        long[] expected;
        if (operation.attrs() == NoOperationAttrs.INSTANCE) {
            expected = new long[0];
        } else if (operation.attrs() instanceof AxisReductionAttrs attrs) {
            if (attrs.axis() >= inputShape.length) return false;
            expected = reducedShape(inputShape, List.of(attrs.axis()), attrs.keepDimensions());
        } else if (operation.attrs() instanceof MultiAxisReductionAttrs attrs) {
            for (int axis : attrs.axes()) if (axis >= inputShape.length) return false;
            expected = reducedShape(inputShape, attrs.axes(), attrs.keepDimensions());
        } else if (operation.attrs() instanceof SumToShapeAttrs attrs) {
            if (kind != AggregateReductionKind.SUM
                    || !attrs.targetShape().equals(output.shape())) {
                return false;
            }
            expected = attrs.targetShape().toLongArray();
            if (expected.length > inputShape.length) return false;
            int padding = inputShape.length - expected.length;
            for (int axis = 0; axis < expected.length; axis++) {
                long target = expected[axis];
                long source = inputShape[axis + padding];
                if (target != 1L && target != source) return false;
            }
        } else {
            return false;
        }
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static long[] reducedShape(
            long[] inputShape, List<Integer> orderedAxes, boolean keepDimensions) {
        boolean[] reduced = new boolean[inputShape.length];
        for (int axis : orderedAxes) {
            if (axis < 0 || axis >= inputShape.length || reduced[axis]) {
                throw new IllegalArgumentException("reduction axes are malformed");
            }
            reduced[axis] = true;
        }
        long[] result = new long[keepDimensions
                ? inputShape.length : inputShape.length - orderedAxes.size()];
        for (int source = 0, target = 0; source < inputShape.length; source++) {
            if (reduced[source]) {
                if (keepDimensions) result[target++] = 1L;
            } else {
                result[target++] = inputShape[source];
            }
        }
        return result;
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
        return canonicalTyped(descriptor, DataType.FLOAT32);
    }

    private static boolean canonicalTyped(TensorDescriptor descriptor, DataType dataType) {
        return geometry(descriptor, dataType, false)
                && descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()));
    }

    private static boolean canonicalReductionOutput(TensorDescriptor descriptor) {
        return geometry(descriptor, DataType.FLOAT32, true)
                && descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()));
    }

    private static boolean geometry(TensorDescriptor descriptor) {
        return geometry(descriptor, DataType.FLOAT32, false);
    }

    private static boolean geometry(
            TensorDescriptor descriptor, DataType dataType, boolean allowScalar) {
        int rank = descriptor.shape().rank();
        if (descriptor.dataType() != dataType
                || !descriptor.shape().isFullyStatic()
                || rank < (allowScalar ? 0 : 1) || rank > 16
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
        Math.multiplyExact(elements, dataType.byteWidth());
        return true;
    }

}
