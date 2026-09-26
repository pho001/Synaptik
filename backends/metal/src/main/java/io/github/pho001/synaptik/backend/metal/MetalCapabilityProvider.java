package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Arrays;
import java.util.Objects;

/**
 * Reports the exact operation-occurrence capability of the current Metal backend.
 *
 * <p>This provider is immutable and performs no native loading, device discovery, allocation,
 * registration, or caching. Under either numerical profile, support includes unary {@code NEG},
 * {@code ABS}, {@code FLOOR}, {@code CEIL}, {@code SIGN}, and {@code RELU}, five FLOAT32 affine
 * transforms, the explicit {@code CONTIGUOUS} canonicalization barrier, bounded canonical FLOAT32
 * {@code UNFOLD_AXIS} materialization, canonical positive-rank INT32 {@code GATHER} indices
 * selecting FLOAT32 data, canonical positive-rank INT32 {@code ONE_HOT} indices producing
 * canonical BOOL values, canonical FLOAT32/INT32/FLOAT32 {@code SCATTER_ELEMENTS} replacement
 * with unique valid targets, and the exact profile-common wires for FLOAT32 classification, BOOL
 * logic, and FLOAT32 {@code WHERE}. The seven BOOL-domain operations require canonical positive
 * rank {@code 1..16}; classification accepts either input gradient flag and produces no-grad BOOL,
 * logic is entirely no-grad, and WHERE propagates the branch gradient OR after branch-first then
 * condition broadcasting.
 * {@code ACCELERATOR} additionally admits tensor {@code ADD}/{@code SUB}/{@code MUL}/{@code DIV},
 * canonical FLOAT32 {@code SUM}/{@code MEAN}/{@code SUM_TO_SHAPE}, positive static rank-two
 * FLOAT32 {@code MATMUL}, and canonical positive-rank FLOAT32 no-gradient scalar
 * {@code ADD}/{@code SUB}/{@code MUL}/{@code DIV} and {@code RECIPROCAL}. Scalar arithmetic
 * retains the exact FLOAT32 raw attribute and operand order; reciprocal is exact {@code 1 / input}.
 * MATMUL accepts each
 * operand only as canonical or as the exact rank-two transpose layout that complete-partition
 * analysis must authenticate to a local {@code PERMUTE [1,0]} producer from a canonical source.
 * Its output is canonical and carries the logical OR of the operand gradient flags. Strict MATMUL
 * remains unsupported. Accelerator reductions admit full, normalized single-axis, ordered
 * normalized multi-axis (including empty-axis identity), and binding-resolved sum-to-Shape forms.
 * The exact Task-0060 integral PROD and BOOL ALL/ANY domain below is profile-common. Other strict
 * reductions remain unsupported. Reduction inputs are canonical with positive dimensions;
 * canonical outputs may be rank zero only as locally produced reduction results.
 * Binary inputs and outputs are canonical dense non-views with exact right-aligned broadcasting. The six exact
 * unary descriptor pairs are canonical. An affine or contiguous input may be canonical or an
 * exact resolved zero-offset logical view; complete-partition analysis authenticates every
 * admitted view as a prior local affine result. Every admitted occurrence uses checked positive
 * extents. GATHER requires its exact replacement-axis output formula and matched data/output
 * gradient flag; ONE_HOT appends its positive depth and is entirely non-differentiable.
 * SCATTER_ELEMENTS requires reduction NONE, equal indices/update Shapes, matching non-axis data
 * extents, exact data-shaped output, non-differentiable indices, and data/update gradient OR.
 * UNFOLD_AXIS requires a canonical rank {@code 1..15} input, size {@code 1..16}, exact floor-count
 * Shape with the window size appended, and a fresh canonical materialization.</p>
 *
 * <p>The profile-common Task-0059 domain also admits exactly nineteen static canonical no-gradient
 * CAST carrier pairs: six identities, BOOL to or from each other carrier, INT32/INT64 in both
 * directions, and BFLOAT16 to FLOAT32. GATHER_ELEMENTS and GATHER_ND preserve any of the six data
 * carriers and require canonical INT32 or INT64 indices. PAD, CONCAT, STACK, and TILE preserve any
 * carrier; UNFOLD2D and UNFOLD3D accept only Model-legal FLOAT64, FLOAT32, or BFLOAT16 inputs.
 * SELECT and positive-step SLICE preserve all six carriers through authenticated physical
 * storage-layout materializations. They require exact static positive-rank Shapes, resolved
 * positive-stride non-overlapping layouts, exact operation-derived offset/stride/span geometry,
 * normalized attributes, and no gradients. Unresolved, zero-stride, negative-stride, overlapping,
 * empty, gradient-bearing, and every unlisted conversion occurrence remains unsupported.</p>
 *
 * <p>The profile-common Task-0060 domain additionally admits replacement-only SCATTER_ND and
 * signed non-zero-step SLICE_UPDATE for all six carriers, including target-relative crop
 * placement. Every scatter tuple is bounds-checked and globally destination-unique before any
 * write. FOLD_AXIS, FOLD2D, and FOLD3D admit only FLOAT64, FLOAT32, or BFLOAT16 with statically
 * proven non-overlapping windows, so every in-bounds contributor is a raw copy and every uncovered
 * cell is the carrier's exact zero.
 * Integral PROD admits INT32/INT64 modular multiplication; ALL and ANY admit canonical BOOL.
 * Those reductions accept full, single-axis, and ordered multi-axis forms, including empty-axis
 * identity on positive-dimensional and rank-zero tensors. All Task-0060 rows are static,
 * canonical, no-gradient, and shape-exact. Zero-length slice updates, scatter reductions,
 * colliding scatter destinations, overlapping folds, floating PROD, non-BOOL ALL/ANY, and
 * LOG_SUM_EXP through L2_NORM remain production-false.</p>
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
            if (operation.kind() == CastKind.CAST) {
                return supportsCast(operation, inputs, output);
            }
            if (operation.kind() == AxisGatherKind.GATHER_ELEMENTS) {
                return supportsGatherElements(operation, inputs, output);
            }
            if (operation.kind() == GatherNdKind.GATHER_ND) {
                return supportsGatherNd(operation, inputs, output);
            }
            if (operation.kind() == SelectKind.SELECT) {
                return supportsSelect(operation, inputs, output);
            }
            if (operation.kind() == SliceKind.SLICE) {
                return supportsSlice(operation, inputs, output);
            }
            if (operation.kind() == ScatterNdKind.SCATTER_ND) {
                return supportsScatterNdReplacement(operation, inputs, output);
            }
            if (operation.kind() == SliceKind.SLICE_UPDATE) {
                return supportsSliceUpdate(operation, inputs, output);
            }
            if (operation.kind() == PadKind.PAD) {
                return supportsPad(operation, inputs, output);
            }
            if (operation.kind() instanceof TensorCompositionKind composition) {
                return supportsComposition(operation, inputs, output, composition);
            }
            if (operation.kind() == TileKind.TILE) {
                return supportsTile(operation, inputs, output);
            }
            if (operation.kind() == WindowTransformKind.UNFOLD2D
                    || operation.kind() == WindowTransformKind.UNFOLD3D) {
                return supportsImageUnfold(operation, inputs, output);
            }
            if (operation.kind() == WindowTransformKind.FOLD_AXIS
                    || operation.kind() == WindowTransformKind.FOLD2D
                    || operation.kind() == WindowTransformKind.FOLD3D) {
                return supportsNonOverlappingFold(operation, inputs, output);
            }
            if (operation.kind() == AxisGatherKind.GATHER) {
                return supportsGather(operation, inputs, output);
            }
            if (operation.kind() == AxisScatterKind.SCATTER_ELEMENTS) {
                return supportsScatterElements(operation, inputs, output);
            }
            if (operation.kind() == OneHotKind.ONE_HOT) {
                return supportsOneHot(operation, inputs, output);
            }
            if (operation.kind() == WindowTransformKind.UNFOLD_AXIS) {
                return supportsUnfoldAxis(operation, inputs, output);
            }
            if (operation.kind() == UnaryElementwiseKind.NEG
                    || operation.kind() == UnaryElementwiseKind.ABS
                    || operation.kind() == UnaryElementwiseKind.FLOOR
                    || operation.kind() == UnaryElementwiseKind.CEIL
                    || operation.kind() == UnaryElementwiseKind.SIGN
                    || operation.kind() == UnaryElementwiseKind.RELU) {
                return supportsCanonicalUnary(operation, inputs, output);
            }
            if (operation.kind() == ContiguousKind.CONTIGUOUS) {
                return supportsContiguous(operation, inputs, output);
            }
            if (operation.kind() instanceof ShapeTransformKind
                    || operation.kind() instanceof AxisTransformKind) {
                return supportsAffine(operation, inputs, output);
            }
            if (operation.kind() instanceof FloatingClassificationKind classification) {
                return supportsClassification(operation, inputs, output, classification);
            }
            if (operation.kind() instanceof BooleanLogicalKind logical) {
                return supportsLogical(operation, inputs, output, logical);
            }
            if (operation.kind() == WhereSelectionKind.WHERE) {
                return supportsWhere(operation, inputs, output);
            }
            if (operation.kind() instanceof AggregateReductionKind reduction
                    && (reduction == AggregateReductionKind.PROD
                            || reduction == AggregateReductionKind.ALL
                            || reduction == AggregateReductionKind.ANY)) {
                return supportsExactReduction(operation, inputs, output, reduction);
            }
            if (numericalProfile == NumericalProfile.ACCELERATOR) {
                if (operation.kind() instanceof BinaryComparisonKind comparison) {
                    return supportsComparison(operation, inputs, output, comparison);
                }
                if (operation.kind() instanceof ScalarElementwiseKind scalar) {
                    return supportsScalar(operation, inputs, output, scalar);
                }
                if (operation.kind() == UnaryElementwiseKind.RECIPROCAL) {
                    return supportsNoGradReciprocal(operation, inputs, output);
                }
                if (operation.kind() instanceof CumulativeScanKind scan) {
                    return supportsScan(operation, inputs, output, scan);
                }
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

    private static boolean supportsSelect(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof SelectAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!supportedStorageLayout(input, 2)
                || !supportedStorageLayout(output, 1)
                || input.dataType() != output.dataType()
                || input.requiresGrad()
                || output.requiresGrad()) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        int axis = attrs.axis();
        if (axis < 0 || axis >= inputShape.length
                || attrs.index() < 0L || attrs.index() >= inputShape[axis]) {
            return false;
        }
        long[] expectedShape = new long[inputShape.length - 1];
        System.arraycopy(inputShape, 0, expectedShape, 0, axis);
        System.arraycopy(inputShape, axis + 1, expectedShape, axis,
                inputShape.length - axis - 1);
        if (!Arrays.equals(expectedShape, output.shape().toLongArray())) {
            return false;
        }
        LayoutDescriptor inputLayout = input.layout().orElseThrow();
        long[] inputStrides = inputLayout.strides();
        long[] expectedStrides = new long[inputStrides.length - 1];
        System.arraycopy(inputStrides, 0, expectedStrides, 0, axis);
        System.arraycopy(inputStrides, axis + 1, expectedStrides, axis,
                inputStrides.length - axis - 1);
        long offset = Math.addExact(
                inputLayout.storageOffset(),
                Math.multiplyExact(attrs.index(), inputStrides[axis]));
        LayoutDescriptor expected =
                LayoutDescriptor.of(output.shape(), expectedStrides, offset, true);
        return output.layout().orElseThrow().equals(expected);
    }

    private static boolean supportsSlice(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!supportedStorageLayout(input, 1)
                || !supportedStorageLayout(output, 1)
                || input.dataType() != output.dataType()
                || input.requiresGrad()
                || output.requiresGrad()
                || input.shape().rank() != output.shape().rank()) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        long[] expectedShape = inputShape.clone();
        long[] starts;
        int[] axes;
        long[] steps;
        if (operation.attrs() instanceof SliceAttrs attrs) {
            int count = attrs.axes().size();
            starts = new long[count];
            axes = new int[count];
            steps = new long[count];
            for (int index = 0; index < count; index++) {
                int axis = attrs.axes().get(index);
                long start = attrs.starts().get(index);
                long length = attrs.lengths().get(index);
                long step = attrs.steps().get(index);
                if (axis < 0 || axis >= inputShape.length || length <= 0L || step <= 0L) {
                    return false;
                }
                long last = Math.addExact(start, Math.multiplyExact(length - 1L, step));
                if (start >= inputShape[axis] || last < 0L || last >= inputShape[axis]) {
                    return false;
                }
                starts[index] = start;
                axes[index] = axis;
                steps[index] = step;
                expectedShape[axis] = length;
            }
        } else if (operation.attrs() instanceof CropToShapeAttrs attrs) {
            if (!attrs.targetShape().isFullyStatic()
                    || !attrs.prefixShape().isFullyStatic()
                    || attrs.targetShape().rank() != inputShape.length
                    || attrs.prefixShape().rank() != inputShape.length) {
                return false;
            }
            expectedShape = attrs.targetShape().toLongArray();
            starts = attrs.prefixShape().toLongArray();
            axes = new int[inputShape.length];
            steps = new long[inputShape.length];
            Arrays.fill(steps, 1L);
            for (int axis = 0; axis < inputShape.length; axis++) {
                axes[axis] = axis;
                if (expectedShape[axis] <= 0L
                        || starts[axis] < 0L
                        || Math.addExact(starts[axis], expectedShape[axis])
                                > inputShape[axis]) {
                    return false;
                }
            }
        } else {
            return false;
        }
        if (!Arrays.equals(expectedShape, output.shape().toLongArray())) {
            return false;
        }
        LayoutDescriptor inputLayout = input.layout().orElseThrow();
        long[] expectedStrides = inputLayout.strides();
        long offset = inputLayout.storageOffset();
        for (int index = 0; index < axes.length; index++) {
            int axis = axes[index];
            long inputStride = expectedStrides[axis];
            offset = Math.addExact(offset, Math.multiplyExact(starts[index], inputStride));
            expectedStrides[axis] = Math.multiplyExact(inputStride, steps[index]);
        }
        LayoutDescriptor expected =
                LayoutDescriptor.of(output.shape(), expectedStrides, offset, true);
        return output.layout().orElseThrow().equals(expected);
    }

    static boolean supportedStorageLayout(TensorDescriptor descriptor, int minimumRank) {
        int rank = descriptor.shape().rank();
        if (!descriptor.shape().isFullyStatic()
                || rank < minimumRank
                || rank > MetalMpsGraphProgram.MAX_RANK
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long[] dimensions = descriptor.shape().toLongArray();
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        int[] order = new int[rank];
        int count = 0;
        long elements = 1L;
        for (int axis = 0; axis < rank; axis++) {
            long dimension = dimensions[axis];
            if (dimension <= 0L || strides[axis] <= 0L) return false;
            elements = Math.multiplyExact(elements, dimension);
            if (dimension > 1L) {
                int insertion = count;
                while (insertion > 0
                        && strides[order[insertion - 1]] > strides[axis]) {
                    order[insertion] = order[insertion - 1];
                    insertion--;
                }
                order[insertion] = axis;
                count++;
            }
        }
        long covered = 1L;
        for (int index = 0; index < count; index++) {
            int axis = order[index];
            long stride = strides[axis];
            if (stride < covered) return false;
            covered = Math.addExact(
                    covered, Math.multiplyExact(dimensions[axis] - 1L, stride));
        }
        long expectedSpan = Math.addExact(layout.storageOffset(), covered);
        if (expectedSpan != layout.referencedElementSpan()) return false;
        Math.multiplyExact(expectedSpan, descriptor.dataType().byteWidth());
        Math.multiplyExact(elements, descriptor.dataType().byteWidth());
        return true;
    }

    private static boolean supportsCast(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof CastAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        DataType source = input.dataType();
        DataType target = attrs.targetDataType();
        boolean provedPair = source == target
                || source == DataType.BOOL
                || target == DataType.BOOL
                || source == DataType.INT32 && target == DataType.INT64
                || source == DataType.INT64 && target == DataType.INT32
                || source == DataType.BFLOAT16 && target == DataType.FLOAT32;
        return provedPair
                && canonicalAny(input, true)
                && canonicalAny(output, true)
                && !input.requiresGrad()
                && !output.requiresGrad()
                && output.dataType() == target
                && input.shape().equals(output.shape());
    }

    private static boolean supportsGatherElements(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof IndexAxisAttrs attrs) || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        if (!canonicalAny(data, false)
                || !canonicalIndex(indices)
                || !canonicalAny(output, false)
                || data.requiresGrad()
                || indices.requiresGrad()
                || output.requiresGrad()
                || output.dataType() != data.dataType()
                || !output.shape().equals(indices.shape())) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        if (dataShape.length != indexShape.length || attrs.axis() >= dataShape.length) {
            return false;
        }
        for (int axis = 0; axis < dataShape.length; axis++) {
            if (axis != attrs.axis() && dataShape[axis] != indexShape[axis]) return false;
        }
        return true;
    }

    private static boolean supportsGatherNd(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof GatherNdAttrs attrs) || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        if (!canonicalAny(data, false)
                || !canonicalIndex(indices)
                || !canonicalAny(output, true)
                || data.requiresGrad()
                || indices.requiresGrad()
                || output.requiresGrad()
                || output.dataType() != data.dataType()) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        int batch = attrs.batchDimensions();
        if (batch >= indexShape.length || batch > dataShape.length) return false;
        for (int axis = 0; axis < batch; axis++) {
            if (dataShape[axis] != indexShape[axis]) return false;
        }
        long tupleWord = indexShape[indexShape.length - 1];
        if (tupleWord < 1L || tupleWord > Integer.MAX_VALUE) return false;
        int tuple = (int) tupleWord;
        if (tuple > dataShape.length - batch) return false;
        int outputRank = Math.addExact(indexShape.length - 1, dataShape.length - batch - tuple);
        if (outputRank > 16 || output.shape().rank() != outputRank) return false;
        long[] expected = new long[outputRank];
        System.arraycopy(indexShape, 0, expected, 0, indexShape.length - 1);
        System.arraycopy(dataShape, batch + tuple, expected, indexShape.length - 1,
                dataShape.length - batch - tuple);
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }


    private static boolean supportsPad(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof PadAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalAny(input, true)
                || !canonicalAny(output, true)
                || input.dataType() != output.dataType()
                || attrs.constantValue().dataType() != input.dataType()
                || input.requiresGrad()
                || output.requiresGrad()) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        if (attrs.before().size() != inputShape.length
                || attrs.after().size() != inputShape.length
                || output.shape().rank() != inputShape.length) {
            return false;
        }
        long[] expected = new long[inputShape.length];
        for (int axis = 0; axis < inputShape.length; axis++) {
            expected[axis] = Math.addExact(
                    Math.addExact(inputShape[axis], attrs.before().get(axis)),
                    attrs.after().get(axis));
        }
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }


    private static boolean supportsComposition(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            TensorCompositionKind kind) {
        if (!(operation.attrs() instanceof CompositionAxisAttrs attrs)
                || inputs.isEmpty() || inputs.size() > MetalMpsGraphProgram.MAX_SELECTOR_EXPANSION
                || !canonicalAny(output, kind == TensorCompositionKind.STACK)
                || output.requiresGrad()) {
            return false;
        }
        DataType type = output.dataType();
        for (TensorDescriptor input : inputs) {
            if (!canonicalAny(input, kind == TensorCompositionKind.STACK)
                    || input.dataType() != type || input.requiresGrad()) return false;
        }
        long[] first = inputs.getFirst().shape().toLongArray();
        if (kind == TensorCompositionKind.CONCAT) {
            if (first.length == 0 || attrs.axis() >= first.length
                    || output.shape().rank() != first.length) return false;
            long[] expected = first.clone();
            expected[attrs.axis()] = 0L;
            for (TensorDescriptor input : inputs) {
                long[] shape = input.shape().toLongArray();
                if (shape.length != first.length) return false;
                for (int axis = 0; axis < shape.length; axis++) {
                    if (axis != attrs.axis() && shape[axis] != first[axis]) return false;
                }
                expected[attrs.axis()] =
                        Math.addExact(expected[attrs.axis()], shape[attrs.axis()]);
            }
            return java.util.Arrays.equals(expected, output.shape().toLongArray());
        }
        if (first.length >= 16 || attrs.axis() > first.length
                || output.shape().rank() != first.length + 1) return false;
        for (TensorDescriptor input : inputs) {
            if (!input.shape().equals(inputs.getFirst().shape())) return false;
        }
        long[] expected = new long[first.length + 1];
        System.arraycopy(first, 0, expected, 0, attrs.axis());
        expected[attrs.axis()] = inputs.size();
        System.arraycopy(first, attrs.axis(), expected, attrs.axis() + 1,
                first.length - attrs.axis());
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static boolean supportsTile(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof TileAttrs attrs) || inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalAny(input, true)
                || !canonicalAny(output, true)
                || input.dataType() != output.dataType()
                || input.requiresGrad()
                || output.requiresGrad()) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        if (attrs.repeats().size() != inputShape.length
                || output.shape().rank() != inputShape.length) return false;
        long[] expected = new long[inputShape.length];
        for (int axis = 0; axis < inputShape.length; axis++) {
            expected[axis] = Math.multiplyExact(inputShape[axis], attrs.repeats().get(axis));
        }
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static boolean supportsImageUnfold(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalAny(input, false)
                || !canonicalAny(output, false)
                || input.dataType() != output.dataType()
                || input.requiresGrad()
                || output.requiresGrad()
                || output.shape().rank() != 3) {
            return false;
        }
        if (input.dataType() != DataType.FLOAT64
                && input.dataType() != DataType.FLOAT32
                && input.dataType() != DataType.BFLOAT16) {
            return false;
        }
        if (operation.kind() == WindowTransformKind.UNFOLD2D) {
            Window2dAttrs window;
            if (operation.attrs() instanceof Window2dAttrs direct) {
                window = direct;
            } else if (operation.attrs() instanceof Unfold2dAttrs explicit
                    && explicit.paddingValue().dataType() == input.dataType()) {
                window = explicit.window();
            } else {
                return false;
            }
            long[] shape = input.shape().toLongArray();
            if (shape.length != 4) return false;
            long height = windowExtent(shape[2], window.kernelHeight(), window.paddingHeight(),
                    window.strideHeight(), window.dilationHeight(), window.ceilMode());
            long width = windowExtent(shape[3], window.kernelWidth(), window.paddingWidth(),
                    window.strideWidth(), window.dilationWidth(), window.ceilMode());
            long[] expected = {
                    shape[0],
                    Math.multiplyExact(Math.multiplyExact(shape[1], window.kernelHeight()),
                            window.kernelWidth()),
                    Math.multiplyExact(height, width)
            };
            return java.util.Arrays.equals(expected, output.shape().toLongArray());
        }
        Window3dAttrs window;
        if (operation.attrs() instanceof Window3dAttrs direct) {
            window = direct;
        } else if (operation.attrs() instanceof Unfold3dAttrs explicit
                && explicit.paddingValue().dataType() == input.dataType()) {
            window = explicit.window();
        } else {
            return false;
        }
        long[] shape = input.shape().toLongArray();
        if (shape.length != 5) return false;
        long depth = windowExtent(shape[2], window.kernelDepth(), window.paddingDepth(),
                window.strideDepth(), window.dilationDepth(), window.ceilMode());
        long height = windowExtent(shape[3], window.kernelHeight(), window.paddingHeight(),
                window.strideHeight(), window.dilationHeight(), window.ceilMode());
        long width = windowExtent(shape[4], window.kernelWidth(), window.paddingWidth(),
                window.strideWidth(), window.dilationWidth(), window.ceilMode());
        long[] expected = {
                shape[0],
                Math.multiplyExact(
                        Math.multiplyExact(
                                Math.multiplyExact(shape[1], window.kernelDepth()),
                                window.kernelHeight()),
                        window.kernelWidth()),
                Math.multiplyExact(Math.multiplyExact(depth, height), width)
        };
        return java.util.Arrays.equals(expected, output.shape().toLongArray());
    }

    private static boolean supportsScatterNdReplacement(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof ScatterNdAttrs attrs)
                || attrs.reduction() != ScatterReduction.NONE
                || inputs.size() != 3) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        TensorDescriptor updates = inputs.get(2);
        if (!canonicalAny(data, false)
                || !canonicalIndex(indices)
                || !canonicalAny(updates, true)
                || !canonicalAny(output, false)
                || data.requiresGrad()
                || indices.requiresGrad()
                || updates.requiresGrad()
                || output.requiresGrad()
                || data.dataType() != updates.dataType()
                || data.dataType() != output.dataType()
                || !data.shape().equals(output.shape())) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        int batch = attrs.batchDimensions();
        if (indexShape.length == 0
                || batch >= indexShape.length
                || batch > dataShape.length) {
            return false;
        }
        for (int axis = 0; axis < batch; axis++) {
            if (dataShape[axis] != indexShape[axis]) return false;
        }
        long tupleWord = indexShape[indexShape.length - 1];
        if (tupleWord < 1L || tupleWord > dataShape.length - batch) return false;
        int tuple = Math.toIntExact(tupleWord);
        long[] expected =
                new long[indexShape.length - 1 + dataShape.length - batch - tuple];
        System.arraycopy(indexShape, 0, expected, 0, indexShape.length - 1);
        System.arraycopy(
                dataShape,
                batch + tuple,
                expected,
                indexShape.length - 1,
                dataShape.length - batch - tuple);
        return Arrays.equals(expected, updates.shape().toLongArray());
    }

    private static boolean supportsSliceUpdate(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (inputs.size() != 2) return false;
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor updates = inputs.get(1);
        if (!canonicalAny(data, false)
                || !canonicalAny(updates, false)
                || !canonicalAny(output, false)
                || data.requiresGrad()
                || updates.requiresGrad()
                || output.requiresGrad()
                || data.dataType() != updates.dataType()
                || data.dataType() != output.dataType()
                || !data.shape().equals(output.shape())) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] expected;
        if (operation.attrs() instanceof CropToShapeAttrs attrs) {
            expected = attrs.targetShape().toLongArray();
            long[] prefix = attrs.prefixShape().toLongArray();
            if (expected.length != dataShape.length || prefix.length != dataShape.length) {
                return false;
            }
            for (int axis = 0; axis < dataShape.length; axis++) {
                if (prefix[axis] < 0L
                        || Math.addExact(prefix[axis], expected[axis]) > dataShape[axis]) {
                    return false;
                }
            }
        } else if (operation.attrs() instanceof SliceAttrs attrs) {
            if (attrs.axes().size() > dataShape.length) return false;
            expected = dataShape.clone();
            boolean[] seen = new boolean[dataShape.length];
            for (int item = 0; item < attrs.axes().size(); item++) {
                int axis = attrs.axes().get(item);
                long start = attrs.starts().get(item);
                long length = attrs.lengths().get(item);
                long step = attrs.steps().get(item);
                if (axis < 0
                        || axis >= dataShape.length
                        || seen[axis]
                        || length < 0L
                        || step == 0L
                        || start < 0L
                        || start > dataShape[axis]) {
                    return false;
                }
                seen[axis] = true;
                if (length == 0L) {
                    expected[axis] = 0L;
                    continue;
                }
                if (start == dataShape[axis]) return false;
                long last = Math.addExact(start, Math.multiplyExact(length - 1L, step));
                if (last < 0L || last >= dataShape[axis]) return false;
                expected[axis] = length;
            }
        } else {
            return false;
        }
        return Arrays.equals(expected, updates.shape().toLongArray());
    }

    private static boolean supportsNonOverlappingFold(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        DataType type = input.dataType();
        if ((type != DataType.FLOAT64
                        && type != DataType.FLOAT32
                        && type != DataType.BFLOAT16)
                || !canonicalAny(input, false)
                || !canonicalAny(output, false)
                || input.requiresGrad()
                || output.requiresGrad()
                || type != output.dataType()) {
            return false;
        }
        long[] source = input.shape().toLongArray();
        long[] target = output.shape().toLongArray();
        if (operation.kind() == WindowTransformKind.FOLD_AXIS) {
            if (!(operation.attrs() instanceof FoldAxisAttrs attrs)
                    || source.length != target.length + 1
                    || attrs.axis() >= target.length
                    || target[attrs.axis()] != attrs.outputSize()) {
                return false;
            }
            long size = source[source.length - 1];
            if (attrs.step() < size || target[attrs.axis()] < size) return false;
            long positions = (target[attrs.axis()] - size) / attrs.step() + 1L;
            for (int axis = 0; axis < target.length; axis++) {
                if (source[axis] != (axis == attrs.axis() ? positions : target[axis])) {
                    return false;
                }
            }
            return true;
        }
        if (operation.kind() == WindowTransformKind.FOLD2D) {
            if (!(operation.attrs() instanceof Fold2dAttrs attrs)
                    || target.length != 4
                    || source.length != 3
                    || !attrs.outputShape().equals(output.shape())) {
                return false;
            }
            Window2dAttrs window = attrs.window();
            if (!nonOverlapping(
                            window.kernelHeight(), window.dilationHeight(), window.strideHeight())
                    || !nonOverlapping(
                            window.kernelWidth(), window.dilationWidth(), window.strideWidth())) {
                return false;
            }
            long kernelVolume =
                    Math.multiplyExact(window.kernelHeight(), window.kernelWidth());
            long positions = Math.multiplyExact(
                    windowExtent(
                            target[2],
                            window.kernelHeight(),
                            window.paddingHeight(),
                            window.strideHeight(),
                            window.dilationHeight(),
                            window.ceilMode()),
                    windowExtent(
                            target[3],
                            window.kernelWidth(),
                            window.paddingWidth(),
                            window.strideWidth(),
                            window.dilationWidth(),
                            window.ceilMode()));
            return source[0] == target[0]
                    && source[1] == Math.multiplyExact(target[1], kernelVolume)
                    && source[2] == positions;
        }
        if (!(operation.attrs() instanceof Fold3dAttrs attrs)
                || target.length != 5
                || source.length != 3
                || !attrs.outputShape().equals(output.shape())) {
            return false;
        }
        Window3dAttrs window = attrs.window();
        if (!nonOverlapping(window.kernelDepth(), window.dilationDepth(), window.strideDepth())
                || !nonOverlapping(
                        window.kernelHeight(), window.dilationHeight(), window.strideHeight())
                || !nonOverlapping(
                        window.kernelWidth(), window.dilationWidth(), window.strideWidth())) {
            return false;
        }
        long kernelVolume = Math.multiplyExact(
                Math.multiplyExact(window.kernelDepth(), window.kernelHeight()),
                window.kernelWidth());
        long positions = Math.multiplyExact(
                Math.multiplyExact(
                        windowExtent(
                                target[2],
                                window.kernelDepth(),
                                window.paddingDepth(),
                                window.strideDepth(),
                                window.dilationDepth(),
                                window.ceilMode()),
                        windowExtent(
                                target[3],
                                window.kernelHeight(),
                                window.paddingHeight(),
                                window.strideHeight(),
                                window.dilationHeight(),
                                window.ceilMode())),
                windowExtent(
                        target[4],
                        window.kernelWidth(),
                        window.paddingWidth(),
                        window.strideWidth(),
                        window.dilationWidth(),
                        window.ceilMode()));
        return source[0] == target[0]
                && source[1] == Math.multiplyExact(target[1], kernelVolume)
                && source[2] == positions;
    }

    private static boolean nonOverlapping(long kernel, long dilation, long stride) {
        return stride >= Math.addExact(Math.multiplyExact(dilation, kernel - 1L), 1L);
    }
    private static long windowExtent(
            long input, long kernel, long padding, long stride, long dilation, boolean ceil) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1L), 1L);
        long numerator =
                Math.subtractExact(Math.addExact(input, Math.multiplyExact(2L, padding)), effective);
        if (numerator < 0L) throw new IllegalArgumentException("window does not fit");
        return Math.addExact(numerator / stride + (ceil && numerator % stride != 0L ? 1L : 0L), 1L);
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

    private static boolean supportsUnfoldAxis(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof UnfoldAxisAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        int rank = input.shape().rank();
        if (!canonical(input)
                || !canonical(output)
                || input.requiresGrad() != output.requiresGrad()
                || rank < 1
                || rank > 15
                || output.shape().rank() != rank + 1
                || attrs.axis() >= rank
                || attrs.size() < 1L
                || attrs.size() > MetalMpsGraphProgram.MAX_SELECTOR_EXPANSION
                || attrs.step() <= 0L) {
            return false;
        }
        long[] inputShape = input.shape().toLongArray();
        long selected = inputShape[attrs.axis()];
        if (attrs.size() > selected) {
            return false;
        }
        long positions = Math.addExact(
                Math.subtractExact(selected, attrs.size()) / attrs.step(), 1L);
        long[] expected = java.util.Arrays.copyOf(inputShape, rank + 1);
        expected[attrs.axis()] = positions;
        expected[rank] = attrs.size();
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

    private static boolean supportsClassification(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            FloatingClassificationKind kind) {
        if (operation.kind() != kind
                || operation.attrs() != NoOperationAttrs.INSTANCE
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonicalTyped(output, DataType.BOOL)
                && input.shape().equals(output.shape())
                && !output.requiresGrad();
    }

    private static boolean supportsLogical(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            BooleanLogicalKind kind) {
        if (operation.kind() != kind || operation.attrs() != NoOperationAttrs.INSTANCE) {
            return false;
        }
        if (kind == BooleanLogicalKind.NOT) {
            if (inputs.size() != 1) return false;
            TensorDescriptor input = inputs.getFirst();
            return canonicalTyped(input, DataType.BOOL)
                    && canonicalTyped(output, DataType.BOOL)
                    && !input.requiresGrad()
                    && !output.requiresGrad()
                    && input.shape().equals(output.shape());
        }
        if (inputs.size() != 2) return false;
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        return canonicalTyped(left, DataType.BOOL)
                && canonicalTyped(right, DataType.BOOL)
                && canonicalTyped(output, DataType.BOOL)
                && !left.requiresGrad()
                && !right.requiresGrad()
                && !output.requiresGrad()
                && ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape());
    }

    private static boolean supportsWhere(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 3) {
            return false;
        }
        TensorDescriptor condition = inputs.get(0);
        TensorDescriptor whenTrue = inputs.get(1);
        TensorDescriptor whenFalse = inputs.get(2);
        if (!canonicalTyped(condition, DataType.BOOL)
                || !canonical(whenTrue)
                || !canonical(whenFalse)
                || !canonical(output)
                || condition.requiresGrad()
                || output.requiresGrad()
                        != (whenTrue.requiresGrad() || whenFalse.requiresGrad())) {
            return false;
        }
        var branchShape = ShapeBroadcast.broadcast(whenTrue.shape(), whenFalse.shape());
        return ShapeBroadcast.broadcast(condition.shape(), branchShape).equals(output.shape());
    }

    private static boolean supportsBinary(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.kind() instanceof BinaryArithmeticKind binary)
                || operation.attrs() != NoOperationAttrs.INSTANCE
                || (binary != BinaryArithmeticKind.ADD
                        && binary != BinaryArithmeticKind.SUB
                        && binary != BinaryArithmeticKind.MUL
                        && binary != BinaryArithmeticKind.DIV
                        && binary != BinaryArithmeticKind.MIN
                        && binary != BinaryArithmeticKind.MAX)
                || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        boolean gradientsValid = binary == BinaryArithmeticKind.MIN
                        || binary == BinaryArithmeticKind.MAX
                ? !left.requiresGrad() && !right.requiresGrad() && !output.requiresGrad()
                : left.requiresGrad() == right.requiresGrad()
                        && left.requiresGrad() == output.requiresGrad();
        return canonical(left)
                && canonical(right)
                && canonical(output)
                && gradientsValid
                && ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape());
    }

    private static boolean supportsComparison(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            BinaryComparisonKind kind) {
        if (operation.kind() != kind
                || operation.attrs() != NoOperationAttrs.INSTANCE
                || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        return canonical(left)
                && canonical(right)
                && canonicalTyped(output, DataType.BOOL)
                && !left.requiresGrad()
                && !right.requiresGrad()
                && !output.requiresGrad()
                && ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape());
    }

    private static boolean supportsScalar(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            ScalarElementwiseKind kind) {
        if (inputs.size() != 1
                || (kind != ScalarElementwiseKind.ADD
                        && kind != ScalarElementwiseKind.SUB
                        && kind != ScalarElementwiseKind.MUL
                        && kind != ScalarElementwiseKind.DIV
                        && kind != ScalarElementwiseKind.MIN
                        && kind != ScalarElementwiseKind.MAX
                        && kind != ScalarElementwiseKind.CLAMP)) {
            return false;
        }
        if (kind == ScalarElementwiseKind.CLAMP) {
            if (!(operation.attrs() instanceof ClampRangeAttrs attrs)
                    || attrs.minValue().dataType() != DataType.FLOAT32
                    || attrs.maxValue().dataType() != DataType.FLOAT32) {
                return false;
            }
        } else if (!(operation.attrs() instanceof ScalarValueAttrs attrs)
                || attrs.value().dataType() != DataType.FLOAT32) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonical(output)
                && !input.requiresGrad()
                && !output.requiresGrad()
                && input.shape().equals(output.shape());
    }

    private static boolean supportsNoGradReciprocal(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonical(output)
                && !input.requiresGrad()
                && !output.requiresGrad()
                && input.shape().equals(output.shape());
    }

    private static boolean supportsScan(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            CumulativeScanKind kind) {
        if (operation.kind() != kind
                || !(operation.attrs() instanceof CumulativeScanAttrs attrs)
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonical(output)
                && !input.requiresGrad()
                && !output.requiresGrad()
                && input.shape().equals(output.shape())
                && attrs.axis() < input.shape().rank();
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
    private static boolean supportsExactReduction(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            AggregateReductionKind kind) {
        if (inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalAny(input, true)
                || !canonicalAny(output, true)
                || input.requiresGrad()
                || output.requiresGrad()
                || input.dataType() != output.dataType()) {
            return false;
        }
        if (kind == AggregateReductionKind.PROD) {
            if (input.dataType() != DataType.INT32 && input.dataType() != DataType.INT64) {
                return false;
            }
        } else if (input.dataType() != DataType.BOOL) {
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
            for (int axis : attrs.axes()) {
                if (axis >= inputShape.length) return false;
            }
            expected = reducedShape(inputShape, attrs.axes(), attrs.keepDimensions());
        } else {
            return false;
        }
        return Arrays.equals(expected, output.shape().toLongArray());
    }



    private static boolean supportsReduction(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            AggregateReductionKind kind) {
        if ((kind != AggregateReductionKind.SUM
                        && kind != AggregateReductionKind.MEAN
                        && kind != AggregateReductionKind.MIN
                        && kind != AggregateReductionKind.MAX)
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        boolean gradientsValid = kind == AggregateReductionKind.MIN
                        || kind == AggregateReductionKind.MAX
                ? !input.requiresGrad() && !output.requiresGrad()
                : input.requiresGrad() == output.requiresGrad();
        if (!canonical(input) || !canonicalReductionOutput(output) || !gradientsValid) {
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

    private static boolean canonicalIndex(TensorDescriptor descriptor) {
        return (descriptor.dataType() == DataType.INT32
                        || descriptor.dataType() == DataType.INT64)
                && canonicalAny(descriptor, false);
    }

    private static boolean canonicalAny(TensorDescriptor descriptor, boolean allowScalar) {
        int rank = descriptor.shape().rank();
        if (!descriptor.shape().isFullyStatic()
                || rank < (allowScalar ? 0 : 1)
                || rank > MetalMpsGraphProgram.MAX_RANK
                || descriptor.layout().isEmpty()
                || !descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()))) {
            return false;
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) return false;
            elements = Math.multiplyExact(elements, dimension);
        }
        Math.multiplyExact(elements, descriptor.dataType().byteWidth());
        return true;
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
