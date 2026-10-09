package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
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
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
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
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Reports the exact operation-occurrence capability of the current Metal backend.
 *
 * <p>The current capability ledger calls this predicate over a stable representative basis and
 * snapshots true and false answers. The historical pre-cutover ledger remains frozen; the only
 * added EXP answers are the separately qualified bounded no-gradient FLOAT32, BFLOAT16, and
 * FLOAT16 occurrences. SIGMOID has separately qualified FLOAT32, BFLOAT16, and FLOAT16 answers.
 * The ledger contains no route, runtime, device, certificate, or generated-backward fact.</p>
 *
 * <p>This provider is immutable and performs no native loading, device discovery, allocation,
 * registration, or caching. The provider admits the exact seven-carrier movement, affine,
 * canonicalization, indexing, replacement, selection, ordering, state, predicate, cast, and
 * non-overlapping window occurrences declared below. Reads may use authenticated static affine or
 * transferred materialized layouts; outputs use the operation's exact logical view or canonical
 * materialization. External unsafe aliases, negative strides, overlap, empty or dynamic geometry,
 * and accumulation-requiring folds remain unsupported.</p>
 *
 * <p>For homogeneous BFLOAT16 and FLOAT16 occurrences, this predicate first checks the original
 * descriptors, then tests an exact FLOAT32 descriptor/attribute proxy against the baseline
 * occurrence predicate. This preserves the independently queried low-type support corresponding
 * to the frozen, formerly ACCELERATOR FLOAT32 baseline. EXP is instead checked by three exact
 * typed predicates, and SIGMOID by three exact typed predicates; neither inherits a low answer
 * from the FLOAT32 proxy. In particular, low
 * L1, ScatterAdd, and
 * singleton variance support does not select the separate FLOAT32-only Task 0069 specialized
 * kernels. Direct BFLOAT16/FLOAT16 mixed-low operations are rejected except explicit
 * {@code CAST}; explicit FLOAT32 casts are the sole mixed-low arithmetic boundary. Route selection
 * is separate, but every admitted occurrence containing a low carrier must use the fixed custom
 * program.</p>
 *
 * <p>All 49 ordered casts use the Model conversion and preserve legal floating gradient metadata.
 * Floating classification covers all four floating carriers. BOOL logic and the fourteen
 * non-mixed-low promoted floating {@code WHERE} signatures use exact right-aligned broadcasting;
 * WHERE differentiability is the branch-role OR. Arithmetic, scalar, reciprocal,
 * reduction, scan, MATMUL, MSE, convolution, pooling, dropout, L1, ScatterAdd, and singleton
 * variance occurrences retain the exact FLOAT32 predicate after homogeneous low projection.
 * Binary arithmetic output differentiability is the operand OR; scalar, reciprocal, reductions,
 * scans, and pooling preserve their declared input/output relation; comparisons publish
 * no-gradient BOOL while permitting differentiable floating inputs.</p>
 *
 * <p>MATMUL accepts canonical operands or authenticated local identity-prefix last-two-axis
 * transposes. The provider admits no-gradient INT32/INT64 pairs and every qualified
 * homogeneous floating geometry plus the existing one-low-plus-FLOAT32 widening rows. Direct
 * BFLOAT16/FLOAT16 mixing, FLOAT64 results, and disallowed gradient metadata remain false.
 * Reductions support the exact full, normalized-axis, keep-dimension, and binding-resolved
 * sum-to-Shape forms below. Index roles accept INT32 or INT64; complete bounds and replacement
 * destination uniqueness are proved before any write.</p>
 *
 * <p>Stable SORT, ARGSORT, and positive-K TOP_K accept all seven carriers; ARG_MAX and ARG_MIN
 * accept the six numeric carriers. Convolution and pooling require fully static positive geometry,
 * exact layouts and result Shape, unsigned-32-bit bounds, and the pooling-kernel cap. Generated
 * input cotangents are admitted only for non-overlapping effective windows. Conv3d backward,
 * overlapping generated folds, attention, recurrent execution, and convolution transpose remain
 * unsupported.</p>
 *
 * <p>INITIAL_STATE is no-gradient. DROPOUT preserves value
 * differentiability and publishes no-gradient mask/state roles. L1, ScatterAdd, and singleton
 * VARIANCE retain their narrow no-gradient predicates. LOG_SUM_EXP,
 * STANDARD_DEVIATION, L2_NORM, and every unlisted occurrence remain production-false.</p>
 */
public final class MetalCapabilityProvider implements BackendCapabilityProvider {
    private static final long UINT32_MAX = 0xffff_ffffL;
    private static final long TASK0064_MAX_POOL_KERNEL_POSITIONS = 65_536L;

    /**
     * Stable Planning ownership identity for the Metal backend.
     *
     * <p>The immutable value is shared by every provider instance and says nothing about device
     * availability or executable readiness.
     */
    public static final BackendId METAL_BACKEND_ID = new BackendId("metal");

    /**
     * Creates a stateless, immutable, and thread-safe Metal capability provider.
     *
     * <p>Construction performs no native loading, device discovery, registration, allocation, or
     * caching.
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
     * Reports support only for the exact prepared Metal occurrence domain.
     *
     * @param query the non-null immutable operation occurrence to classify without probing a device
     * or native library
     * @return {@code true} exactly for an admitted occurrence in the qualified Metal domain
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        return supportsOccurrence(query.operation(), query.inputs(), query.outputs());
    }

    /**
     * Validates one projected occurrence against the same domain used by Planning.
     *
     * @param operation non-null typed operation
     * @param inputs non-null ordered input descriptors
     * @param outputs non-null ordered output descriptors
     * @return {@code true} only if the original occurrence passes the exact current predicate or
     *     an independently qualified {@code EXP} or {@code SIGMOID} typed predicate, or a
     *     homogeneous low-type occurrence
     *     passes the frozen-baseline FLOAT32 proxy predicate; other FLOAT32-only additions do not
     *     gain low support through that proxy
     * @throws NullPointerException if {@code operation}, {@code inputs}, or {@code outputs} is
     *     {@code null}
     */
    static boolean supportsOccurrence(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        if (mixedLowPrecision(inputs, outputs) && operation.kind() != CastKind.CAST) {
            return false;
        }
        if (supportsFrozenBaselineOccurrence(operation, inputs, outputs)) {
            return true;
        }
        if (operation.kind() == UnaryElementwiseKind.SIGMOID && outputs.size() == 1) {
            try {
                return supportsFloat32ExpOrSigmoid(operation, inputs, outputs.getFirst())
                        || supportsLowExpOrSigmoid(operation, inputs, outputs.getFirst(),
                                DataType.BFLOAT16)
                        || supportsLowExpOrSigmoid(operation, inputs, outputs.getFirst(),
                                DataType.FLOAT16);
            } catch (IllegalArgumentException | ArithmeticException incompatible) {
                return false;
            }
        }
        if (operation.kind() == UnaryElementwiseKind.EXP && outputs.size() == 1) {
            try {
                return supportsFloat32ExpOrSigmoid(operation, inputs, outputs.getFirst())
                        || supportsLowExpOrSigmoid(operation, inputs, outputs.getFirst(),
                                DataType.BFLOAT16)
                        || supportsLowExpOrSigmoid(operation, inputs, outputs.getFirst(),
                                DataType.FLOAT16);
            } catch (IllegalArgumentException | ArithmeticException incompatible) {
                return false;
            }
        }
        DataType lowType = homogeneousLowPrecision(inputs, outputs);
        if (lowType == null || !lowScalarAttrsMatch(operation.attrs(), lowType)) {
            return false;
        }
        Operation proxy =
                new Operation(operation.kind(), float32ProxyAttrs(operation.attrs()));
        List<TensorDescriptor> proxyInputs =
                inputs.stream().map(MetalCapabilityProvider::float32Proxy).toList();
        List<TensorDescriptor> proxyOutputs =
                outputs.stream().map(MetalCapabilityProvider::float32Proxy).toList();
        return supportsFrozenBaselineOccurrence(proxy, proxyInputs, proxyOutputs);
    }

    /**
     * Evaluates only the preserved cutover domain, which is the sole authority for low proxies.
     * Post-cutover additions must be checked separately on their original typed descriptors.
     *
     * @param operation non-null typed operation
     * @param inputs non-null ordered input descriptors
     * @param outputs non-null ordered output descriptors
     * @return whether the frozen-baseline provider admits the occurrence
     */
    private static boolean supportsFrozenBaselineOccurrence(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        if (operation.kind() == GraphRngKind.INITIAL_STATE
                || operation.kind() == DropoutKind.DROPOUT) {
            try {
                return operation.kind() == GraphRngKind.INITIAL_STATE
                        ? supportsInitialState(operation, inputs, outputs)
                        : supportsDropout(operation, inputs, outputs);
            } catch (IllegalArgumentException | ArithmeticException incompatible) {
                return false;
            }
        }
        if (operation.kind() == TopKKind.TOP_K) {
            try {
                return supportsTopK(operation, inputs, outputs);
            } catch (IllegalArgumentException | ArithmeticException incompatible) {
                return false;
            }
        }
        if (outputs.size() != 1) {
            return false;
        }
        TensorDescriptor output = outputs.getFirst();
        try {
            if (operation.kind() instanceof OrderingKind ordering) {
                return supportsOrdering(operation, inputs, output, ordering);
            }
            if (operation.kind() == AggregateReductionKind.ARG_MAX
                    || operation.kind() == AggregateReductionKind.ARG_MIN) {
                return supportsArgExtrema(operation, inputs, output);
            }
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
            if (operation.kind() == Conv2dKind.CONV2D
                    || operation.kind() == Conv3dKind.CONV3D) {
                return supportsConvolution(operation, inputs, output);
            }
            if (operation.kind() instanceof Pool2dKind
                    || operation.kind() instanceof Pool3dKind) {
                return supportsPooling(operation, inputs, output);
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
            if (operation.kind() == AxisScatterKind.SCATTER_ADD) {
                return supportsTask0069ScatterAdd(
                        operation, inputs, output);
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
            if (operation.kind() == MatmulKind.MATMUL) {
                return supportsMatmul(operation, inputs, output);
            }
            if (operation.kind() instanceof BinaryComparisonKind comparison) {
                return supportsComparison(operation, inputs, output, comparison);
            }
            if (operation.kind() == LossKind.MEAN_SQUARED_ERROR) {
                return supportsMeanSquaredError(operation, inputs, output);
            }
            if (operation.kind() instanceof ScalarElementwiseKind scalar) {
                return supportsScalar(operation, inputs, output, scalar);
            }
            if (operation.kind() == UnaryElementwiseKind.RECIPROCAL) {
                return supportsReciprocal(operation, inputs, output);
            }
            if (operation.kind() instanceof CumulativeScanKind scan) {
                return supportsScan(operation, inputs, output, scan);
            }
            if (operation.kind() instanceof AggregateReductionKind reduction) {
                if (reduction == AggregateReductionKind.L1_NORM) {
                    return supportsTask0069L1Norm(operation, inputs, output);
                }
                if (reduction == AggregateReductionKind.VARIANCE) {
                    return supportsTask0069Variance(operation, inputs, output);
                }
                return supportsReduction(operation, inputs, output, reduction);
            }
            return supportsBinary(operation, inputs, output);
        } catch (IllegalArgumentException | ArithmeticException incompatible) {
            return false;
        }
    }
    private static boolean mixedLowPrecision(
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        boolean bfloat16 = false;
        boolean float16 = false;
        for (TensorDescriptor descriptor : concat(inputs, outputs)) {
            bfloat16 |= descriptor.dataType() == DataType.BFLOAT16;
            float16 |= descriptor.dataType() == DataType.FLOAT16;
        }
        return bfloat16 && float16;
    }

    private static DataType homogeneousLowPrecision(
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        DataType selected = null;
        for (TensorDescriptor descriptor : concat(inputs, outputs)) {
            DataType type = descriptor.dataType();
            if (type == DataType.BFLOAT16 || type == DataType.FLOAT16) {
                if (selected != null && selected != type) {
                    return null;
                }
                selected = type;
            } else if (type.isFloating()) {
                return null;
            }
        }
        return selected;
    }

    private static List<TensorDescriptor> concat(
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        var descriptors =
                new java.util.ArrayList<TensorDescriptor>(inputs.size() + outputs.size());
        descriptors.addAll(inputs);
        descriptors.addAll(outputs);
        return descriptors;
    }

    private static TensorDescriptor float32Proxy(TensorDescriptor descriptor) {
        DataType type = descriptor.dataType() == DataType.BFLOAT16
                        || descriptor.dataType() == DataType.FLOAT16
                ? DataType.FLOAT32
                : descriptor.dataType();
        return new TensorDescriptor(
                type, descriptor.shape(), descriptor.layout(), descriptor.requiresGrad());
    }

    private static io.github.pho001.synaptik.model.operation.OperationAttrs float32ProxyAttrs(
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs) {
        if (attrs instanceof ScalarValueAttrs scalar) {
            return new ScalarValueAttrs(float32Proxy(scalar.value()));
        }
        if (attrs instanceof ClampRangeAttrs clamp) {
            return new ClampRangeAttrs(
                    float32Proxy(clamp.minValue()), float32Proxy(clamp.maxValue()));
        }
        if (attrs instanceof PadAttrs pad) {
            return new PadAttrs(
                    pad.before(), pad.after(), float32Proxy(pad.constantValue()));
        }
        if (attrs instanceof Unfold2dAttrs unfold) {
            return new Unfold2dAttrs(
                    unfold.window(), float32Proxy(unfold.paddingValue()));
        }
        if (attrs instanceof Unfold3dAttrs unfold) {
            return new Unfold3dAttrs(
                    unfold.window(), float32Proxy(unfold.paddingValue()));
        }
        if (attrs instanceof CastAttrs cast) {
            DataType target = cast.targetDataType() == DataType.BFLOAT16
                            || cast.targetDataType() == DataType.FLOAT16
                    ? DataType.FLOAT32
                    : cast.targetDataType();
            return new CastAttrs(target);
        }
        return attrs;
    }

    private static ScalarValue float32Proxy(ScalarValue value) {
        return switch (value.dataType()) {
            case BFLOAT16 -> ScalarValue.float32(BFloat16Bits.toFloat(value.bfloat16Bits()));
            case FLOAT16 -> ScalarValue.float32(Float16Bits.toFloat(value.float16Bits()));
            default -> value;
        };
    }

    private static boolean lowScalarAttrsMatch(
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs, DataType selected) {
        if (attrs instanceof ScalarValueAttrs scalar) {
            return scalar.value().dataType() == selected;
        }
        if (attrs instanceof ClampRangeAttrs clamp) {
            return clamp.minValue().dataType() == selected
                    && clamp.maxValue().dataType() == selected;
        }
        if (attrs instanceof PadAttrs pad) {
            return pad.constantValue().dataType() == selected;
        }
        if (attrs instanceof Unfold2dAttrs unfold) {
            return unfold.paddingValue().dataType() == selected;
        }
        if (attrs instanceof Unfold3dAttrs unfold) {
            return unfold.paddingValue().dataType() == selected;
        }
        return true;
    }


    private static boolean supportsInitialState(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (operation.kind() != GraphRngKind.INITIAL_STATE
                || !(operation.attrs() instanceof GraphRngStateAttrs)
                || !inputs.isEmpty()
                || outputs.size() != 1) {
            return false;
        }
        TensorDescriptor output = outputs.getFirst();
        return task0065Canonical(output, DataType.INT64, false)
                && output.shape().equals(Shape.of(2))
                && !output.requiresGrad();
    }

    private static boolean supportsDropout(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (operation.kind() != DropoutKind.DROPOUT
                || !(operation.attrs() instanceof DropoutAttrs)
                || inputs.size() != 2
                || outputs.size() != 3) {
            return false;
        }
        TensorDescriptor value = inputs.get(0);
        TensorDescriptor state = inputs.get(1);
        TensorDescriptor output = outputs.get(0);
        TensorDescriptor mask = outputs.get(1);
        TensorDescriptor nextState = outputs.get(2);
        return task0065Canonical(value, DataType.FLOAT32, true)
                && task0065Canonical(output, DataType.FLOAT32, true)
                && task0065Canonical(mask, DataType.BOOL, true)
                && task0065Canonical(state, DataType.INT64, false)
                && task0065Canonical(nextState, DataType.INT64, false)
                && value.shape().equals(output.shape())
                && value.shape().equals(mask.shape())
                && state.shape().equals(Shape.of(2))
                && nextState.shape().equals(Shape.of(2))
                && value.requiresGrad() == output.requiresGrad()
                && !mask.requiresGrad()
                && !state.requiresGrad()
                && !nextState.requiresGrad();
    }

    private static boolean task0065Canonical(
            TensorDescriptor descriptor, DataType type, boolean allowScalar) {
        if (descriptor.dataType() != type || !canonicalAny(descriptor, allowScalar)) {
            return false;
        }
        long elements = 1L;
        for (long extent : descriptor.shape().toLongArray()) {
            if (extent > UINT32_MAX) return false;
            elements = Math.multiplyExact(elements, extent);
            if (elements > UINT32_MAX) return false;
        }
        return descriptor.layout().orElseThrow().referencedElementSpan() <= UINT32_MAX;
    }

    private static boolean supportsOrdering(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            OrderingKind kind) {
        if (!(operation.attrs() instanceof SortAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalOrdering(input, false)
                || !canonicalOrdering(output, false)
                || attrs.axis() >= input.shape().rank()
                || !input.shape().equals(output.shape())) {
            return false;
        }
        if (kind == OrderingKind.SORT) {
            return output.dataType() == input.dataType()
                    && output.requiresGrad() == input.requiresGrad();
        }
        return output.dataType() == DataType.INT64 && !output.requiresGrad();
    }

    private static boolean supportsTopK(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (!(operation.attrs() instanceof TopKAttrs attrs)
                || inputs.size() != 1 || outputs.size() != 2) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        TensorDescriptor values = outputs.get(0);
        TensorDescriptor indices = outputs.get(1);
        if (!canonicalOrdering(input, false)
                || !canonicalOrdering(values, false)
                || !canonicalOrdering(indices, false)
                || attrs.axis() >= input.shape().rank()) {
            return false;
        }
        long[] expected = input.shape().toLongArray();
        if (attrs.k() < 1L || attrs.k() > expected[attrs.axis()]
                || attrs.k() > UINT32_MAX) {
            return false;
        }
        expected[attrs.axis()] = attrs.k();
        Shape expectedShape = Shape.of(expected);
        return values.dataType() == input.dataType()
                && values.requiresGrad() == input.requiresGrad()
                && values.shape().equals(expectedShape)
                && indices.dataType() == DataType.INT64
                && !indices.requiresGrad()
                && indices.shape().equals(expectedShape);
    }

    private static boolean supportsArgExtrema(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof ArgExtremaAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonicalOrdering(input, false)
                || !canonicalOrdering(output, true)
                || input.dataType() == DataType.BOOL
                || attrs.axis() >= input.shape().rank()
                || output.dataType() != DataType.INT64
                || output.requiresGrad()
                || (attrs.tiePolicy() != ArgExtremaTiePolicy.FIRST_INDEX
                        && attrs.tiePolicy() != ArgExtremaTiePolicy.LAST_INDEX)) {
            return false;
        }
        long[] source = input.shape().toLongArray();
        long[] expected;
        if (attrs.keepDimensions()) {
            expected = source.clone();
            expected[attrs.axis()] = 1L;
        } else {
            expected = new long[source.length - 1];
            System.arraycopy(source, 0, expected, 0, attrs.axis());
            System.arraycopy(
                    source, attrs.axis() + 1, expected, attrs.axis(),
                    source.length - attrs.axis() - 1);
        }
        return output.shape().equals(Shape.of(expected));
    }

    private static boolean canonicalOrdering(
            TensorDescriptor descriptor, boolean allowScalar) {
        if (!canonicalAny(descriptor, allowScalar)) {
            return false;
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension > UINT32_MAX) {
                return false;
            }
            elements = Math.multiplyExact(elements, dimension);
            if (elements > UINT32_MAX) {
                return false;
            }
        }
        return true;
    }

    private static boolean supportsSelect(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof SelectAttrs attrs) || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!selectedAffineRead(input, false)
                || !selectedAffineRead(output, true)
                || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()) {
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
        if (!selectedAffineRead(input, true)
                || !selectedAffineRead(output, true)
                || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()
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
        boolean expectedGradient = input.requiresGrad() && isFloating( source) && isFloating( target);
        return selectedAffineRead(input, true)
                && canonicalAny(output, true)
                && output.dataType() == target
                && input.shape().equals(output.shape())
        && output.requiresGrad() == expectedGradient;
    }

    private static boolean supportsGatherElements(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (!(operation.attrs() instanceof IndexAxisAttrs attrs) || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        if (!selectedAffineRead(data, false)
                || !selectedIndexRead(indices, false)
                || !canonicalAny(output, false)
                || indices.requiresGrad()
                || output.requiresGrad() != data.requiresGrad()
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
        if (!selectedAffineRead(data, false)
                || !selectedIndexRead(indices, false)
                || !canonicalAny(output, true)
                || indices.requiresGrad()
                || output.requiresGrad() != data.requiresGrad()
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
        if (!selectedAffineRead(input, true)
                || !canonicalAny(output, true)
                || input.dataType() != output.dataType()
                || attrs.constantValue().dataType() != input.dataType()
                || input.requiresGrad() != output.requiresGrad()) {
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
                || !canonicalAny(output, kind == TensorCompositionKind.STACK)) {
            return false;
        }
        DataType type = output.dataType();
    boolean expectedGradient = false;
        for (TensorDescriptor input : inputs) {
            if (!selectedAffineRead(input, kind == TensorCompositionKind.STACK)
                    || input.dataType() != type) {
        return false;
      }
      expectedGradient |= input.requiresGrad();
    }
    if (output.requiresGrad() != expectedGradient) return false;
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
        if (!selectedAffineRead(input, true)
                || !canonicalAny(output, true)
                || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()) {
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
        if (!selectedAffineRead(input, false)
                || !canonicalAny(output, false)
                || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()
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
        if (!selectedAffineRead(data, false)
                || !selectedIndexRead(indices, false)
                || !selectedAffineRead(updates, true)
                || !canonicalAny(output, false)
                || indices.requiresGrad()
                || output.requiresGrad() != (data.requiresGrad()
                || updates.requiresGrad())
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
        if (!selectedAffineRead(data, true)
                || !selectedAffineRead(updates, true)
                || !canonicalAny(output, true)
                || output.requiresGrad() != ( data.requiresGrad()
                || updates.requiresGrad())
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
    boolean supportedType =
        operation.kind () == WindowTransformKind.FOLD_AXIS
            ?type != DataType.BOOL
            : isFloating( type);
    if ( !supportedType
                || !selectedAffineRead(input, false)
                || !canonicalAny(output, false)
                || input.requiresGrad() != output.requiresGrad()
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
        if (!selectedAffineRead(data, false)
                || !selectedIndexRead(indices, true)
                || !canonicalAny(output, true)
        || data.dataType() != output.dataType()
                || indices.requiresGrad()
                || data.requiresGrad() != output.requiresGrad()) {
            return false;
        }
        long[] dataShape = data.shape().toLongArray();
        long[] indexShape = indices.shape().toLongArray();
        int axis = attrs.axis();
        int outputRank = Math.addExact(Math.subtractExact(dataShape.length, 1), indexShape.length);
        if (axis >= dataShape.length || outputRank < 0 || outputRank > 16
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
        if (!selectedAffineRead(data, false)
                || !selectedIndexRead(indices, false)
                || !selectedAffineRead(updates, false)
                || !canonicalAny(output, false)
                || indices.requiresGrad()
                || output.requiresGrad() != (data.requiresGrad() || updates.requiresGrad())
        || data.dataType() != updates.dataType()
        || data.dataType() != output.dataType()
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
        if (!selectedIndexRead(indices, true)
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
        if (!selectedAffineRead(input, false)
                || !canonicalAny(output, false)
        || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()
                || rank < 1
                || rank > 15
                || output.shape().rank() != rank + 1
                || attrs.axis() >= rank
                || attrs.size() < 1L
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
        boolean allowScalar = operation.kind() == UnaryElementwiseKind.RELU;
        return input.dataType() == DataType.FLOAT32
                && output.dataType() == DataType.FLOAT32
                && canonicalAny(input, allowScalar)
                && canonicalAny(output, allowScalar)
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
        return isFloating(input.dataType())
                && selectedAffineRead(input, true)
        && output.dataType() == DataType.BOOL
        && canonicalAny(output, true)
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
            return selectedAffineRead(input, true)
          && input.dataType() == DataType.BOOL
                    && output.dataType() == DataType.BOOL
          && canonicalAny(output, true)
                    && !input.requiresGrad()
                    && !output.requiresGrad()
                    && input.shape().equals(output.shape());
        }
        if (inputs.size() != 2) return false;
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        return selectedAffineRead(left, true)
        && left.dataType() == DataType.BOOL
        && selectedAffineRead(right, true)
                && right.dataType() == DataType.BOOL
                && output.dataType() == DataType.BOOL
        && canonicalAny(output, true)
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
    DataType resultType =
        DataTypePromotion.promoteFloating(whenTrue.dataType(), whenFalse.dataType());
        if (!selectedAffineRead(condition, true)
        || condition.dataType() != DataType.BOOL
        || !selectedAffineRead(whenTrue, true)
                || !selectedAffineRead(whenFalse, true)
                || !canonicalAny(output, true)
                || output.dataType() != resultType
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
        boolean gradientsValid =
                output.requiresGrad() == (left.requiresGrad() || right.requiresGrad());
        boolean storageValid = left.dataType() == DataType.FLOAT32
                && right.dataType() == DataType.FLOAT32
                && output.dataType() == DataType.FLOAT32
                && selectedAffineRead(left, true)
                && selectedAffineRead(right, true)
                && canonicalAny(output, true);
        return storageValid
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
        return selectedAffineRead(left, true)
                && selectedAffineRead(right, true)
                && canonicalTyped(output, DataType.BOOL)
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
        boolean allowScalar =
                kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.CLAMP;
        return input.dataType() == DataType.FLOAT32
                && output.dataType() == DataType.FLOAT32
                && canonicalAny(input, allowScalar)
                && canonicalAny(output, allowScalar)
                && input.requiresGrad() == output.requiresGrad()
                && input.shape().equals(output.shape());
    }

    private static boolean supportsReciprocal(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        return canonical(input)
                && canonical(output)
                && input.requiresGrad() == output.requiresGrad()
                && input.shape().equals(output.shape());
    }

    /**
     * Checks a bounded no-gradient FLOAT32 EXP or SIGMOID occurrence for its fixed MPSGraph
     * route: direct for EXP and sign-guarded composition for SIGMOID. This predicate is
     * deliberately reached before the frozen low proxy, so neither kind inherits a low-valued
     * answer.
     *
     * @param operation non-null unary operation
     * @param inputs non-null ordered input descriptors
     * @param output non-null output descriptor
     * @return whether this exact FLOAT32 occurrence has its qualified MPSGraph route
     */
    private static boolean supportsFloat32ExpOrSigmoid(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        return canonical(input) && canonical(output)
                && !input.requiresGrad() && !output.requiresGrad()
                && input.shape().equals(output.shape())
                && input.layout().orElseThrow().referencedElementSpan()
                        <= UINT32_MAX / Float.BYTES;
    }

    /**
     * Checks one explicitly selected no-gradient low EXP or SIGMOID domain. BFLOAT16 and FLOAT16
     * call this predicate separately, each with a distinct fixed custom kernel and a checked
     * two-byte span.
     *
     * @param operation non-null unary operation with no attributes
     * @param inputs non-null ordered input descriptors
     * @param output non-null output descriptor
     * @param type exact BFLOAT16 or FLOAT16 storage type to query; never inferred from FLOAT32
     * @return whether this exact low-type occurrence has the qualified custom route
     */
    private static boolean supportsLowExpOrSigmoid(
            Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output,
            DataType type) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 1) return false;
        TensorDescriptor input = inputs.getFirst();
        return (type == DataType.BFLOAT16 || type == DataType.FLOAT16)
                && canonicalTyped(input, type) && canonicalTyped(output, type)
                && !input.requiresGrad() && !output.requiresGrad()
                && input.shape().equals(output.shape())
                && input.layout().orElseThrow().referencedElementSpan()
                        <= UINT32_MAX / Short.BYTES;
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
                && input.requiresGrad() == output.requiresGrad()
                && input.shape().equals(output.shape())
                && attrs.axis() < input.shape().rank();
    }

    private static boolean supportsConvolution(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        int spatial = operation.kind() == Conv2dKind.CONV2D ? 2 : 3;
        if (inputs.size() < 2 || inputs.size() > 3
                || output.dataType() != DataType.FLOAT32
                || !task0064Canonical(output, spatial + 2)) {
            return false;
        }
        if (spatial == 2 && !(operation.attrs() instanceof Conv2dAttrs)
                || spatial == 3 && !(operation.attrs() instanceof Conv3dAttrs)) {
            return false;
        }
        TensorDescriptor input = inputs.get(0);
        TensorDescriptor weight = inputs.get(1);
        if (!task0064Image(input, spatial + 2, spatial == 2)
                || !task0064Image(weight, spatial + 2, spatial == 2)
                || !task0064FloatCarrier(input.dataType())
                || !task0064FloatCarrier(weight.dataType())) {
            return false;
        }
        TensorDescriptor bias = inputs.size() == 3 ? inputs.get(2) : null;
        if (bias != null
                && (!task0064Canonical(bias, 1)
                        || !task0064FloatCarrier(bias.dataType()))) {
            return false;
        }
        boolean anyFloat32 = input.dataType() == DataType.FLOAT32
                || weight.dataType() == DataType.FLOAT32
                || bias != null && bias.dataType() == DataType.FLOAT32;
        if (!anyFloat32) return false;
        boolean anyBfloat16 = input.dataType() == DataType.BFLOAT16
                || weight.dataType() == DataType.BFLOAT16
                || bias != null && bias.dataType() == DataType.BFLOAT16;
        boolean inputGradient = inputs.stream().anyMatch(TensorDescriptor::requiresGrad);
        if (anyBfloat16) {
            if (inputGradient || output.requiresGrad()) return false;
        } else if (output.requiresGrad() != inputGradient) {
            return false;
        }
        long[] source = input.shape().toLongArray();
        long[] kernel = weight.shape().toLongArray();
        long[] target = output.shape().toLongArray();
        long groups;
        long[] strides = new long[spatial];
        long[] padding = new long[spatial];
        long[] dilation = new long[spatial];
        if (spatial == 2) {
            Conv2dAttrs attrs = (Conv2dAttrs) operation.attrs();
            groups = attrs.groups();
            strides[0] = attrs.strideHeight();
            strides[1] = attrs.strideWidth();
            padding[0] = attrs.paddingHeight();
            padding[1] = attrs.paddingWidth();
            dilation[0] = attrs.dilationHeight();
            dilation[1] = attrs.dilationWidth();
        } else {
            Conv3dAttrs attrs = (Conv3dAttrs) operation.attrs();
            groups = attrs.groups();
            strides[0] = attrs.strideDepth();
            strides[1] = attrs.strideHeight();
            strides[2] = attrs.strideWidth();
            padding[0] = attrs.paddingDepth();
            padding[1] = attrs.paddingHeight();
            padding[2] = attrs.paddingWidth();
            dilation[0] = attrs.dilationDepth();
            dilation[1] = attrs.dilationHeight();
            dilation[2] = attrs.dilationWidth();
        }
        if (!task0064Unsigned(groups)
                || source[0] != target[0]
                || kernel[0] != target[1]
                || source[1] % groups != 0
                || kernel[0] % groups != 0
                || kernel[1] != source[1] / groups
                || bias != null && (bias.shape().toLongArray()[0] != kernel[0])) {
            return false;
        }
        long contributors = kernel[1];
        for (int axis = 0; axis < spatial; axis++) {
            if (!task0064Unsigned(strides[axis])
                    || !task0064UnsignedOrZero(padding[axis])
                    || !task0064Unsigned(dilation[axis])
                    || target[axis + 2] != task0064WindowExtent(
                            source[axis + 2], kernel[axis + 2], strides[axis],
                            padding[axis], dilation[axis], false)) {
                return false;
            }
            contributors = Math.multiplyExact(contributors, kernel[axis + 2]);
            if (input.requiresGrad()
                    && !nonOverlapping(
                            kernel[axis + 2], dilation[axis], strides[axis])) {
                return false;
            }
        }
        return contributors <= UINT32_MAX;
    }

    private static boolean supportsPooling(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (inputs.size() != 1) return false;
        int spatial = operation.kind() instanceof Pool2dKind ? 2 : 3;
        boolean maximum = operation.kind() == Pool2dKind.MAX_POOL2D
                || operation.kind() == Pool3dKind.MAX_POOL3D;
        TensorDescriptor input = inputs.getFirst();
        if (!task0064Image(input, spatial + 2, spatial == 2)
                || !task0064Canonical(output, spatial + 2)
                || !input.shape().isFullyStatic()
                || input.dataType() != output.dataType()
                || input.requiresGrad() != output.requiresGrad()) {
            return false;
        }
        if (maximum) {
            if (input.dataType() != DataType.FLOAT64
                    && input.dataType() != DataType.FLOAT32
                    && input.dataType() != DataType.BFLOAT16) {
                return false;
            }
        } else if (input.dataType() != DataType.FLOAT32) {
            return false;
        }
        long[] kernel = new long[spatial];
        long[] stride = new long[spatial];
        long[] padding = new long[spatial];
        long[] dilation = new long[spatial];
        boolean ceil;
        if (spatial == 2 && maximum && operation.attrs() instanceof MaxPool2dAttrs attrs) {
            kernel[0] = attrs.kernelHeight(); kernel[1] = attrs.kernelWidth();
            stride[0] = attrs.strideHeight(); stride[1] = attrs.strideWidth();
            padding[0] = attrs.paddingHeight(); padding[1] = attrs.paddingWidth();
            dilation[0] = attrs.dilationHeight(); dilation[1] = attrs.dilationWidth();
            ceil = attrs.ceilMode();
        } else if (spatial == 2 && !maximum
                && operation.attrs() instanceof AveragePool2dAttrs attrs) {
            kernel[0] = attrs.kernelHeight(); kernel[1] = attrs.kernelWidth();
            stride[0] = attrs.strideHeight(); stride[1] = attrs.strideWidth();
            padding[0] = attrs.paddingHeight(); padding[1] = attrs.paddingWidth();
            dilation[0] = attrs.dilationHeight(); dilation[1] = attrs.dilationWidth();
            ceil = attrs.ceilMode();
        } else if (spatial == 3 && maximum
                && operation.attrs() instanceof MaxPool3dAttrs attrs) {
            kernel[0] = attrs.kernelDepth(); kernel[1] = attrs.kernelHeight();
            kernel[2] = attrs.kernelWidth();
            stride[0] = attrs.strideDepth(); stride[1] = attrs.strideHeight();
            stride[2] = attrs.strideWidth();
            padding[0] = attrs.paddingDepth(); padding[1] = attrs.paddingHeight();
            padding[2] = attrs.paddingWidth();
            dilation[0] = attrs.dilationDepth(); dilation[1] = attrs.dilationHeight();
            dilation[2] = attrs.dilationWidth();
            ceil = attrs.ceilMode();
        } else if (spatial == 3 && !maximum
                && operation.attrs() instanceof AveragePool3dAttrs attrs) {
            kernel[0] = attrs.kernelDepth(); kernel[1] = attrs.kernelHeight();
            kernel[2] = attrs.kernelWidth();
            stride[0] = attrs.strideDepth(); stride[1] = attrs.strideHeight();
            stride[2] = attrs.strideWidth();
            padding[0] = attrs.paddingDepth(); padding[1] = attrs.paddingHeight();
            padding[2] = attrs.paddingWidth();
            dilation[0] = attrs.dilationDepth(); dilation[1] = attrs.dilationHeight();
            dilation[2] = attrs.dilationWidth();
            ceil = attrs.ceilMode();
        } else {
            return false;
        }
        long[] source = input.shape().toLongArray();
        long[] target = output.shape().toLongArray();
        if (source[0] != target[0] || source[1] != target[1]) return false;
        long divisor = 1L;
        for (int axis = 0; axis < spatial; axis++) {
            if (!task0064Unsigned(kernel[axis])
                    || !task0064Unsigned(stride[axis])
                    || !task0064UnsignedOrZero(padding[axis])
                    || !task0064Unsigned(dilation[axis])
                    || target[axis + 2] != task0064WindowExtent(
                            source[axis + 2], kernel[axis], stride[axis],
                            padding[axis], dilation[axis], ceil)) {
                return false;
            }
            if (divisor > TASK0064_MAX_POOL_KERNEL_POSITIONS / kernel[axis]) {
                return false;
            }
            divisor *= kernel[axis];
            if (input.requiresGrad()
                    && !nonOverlapping(kernel[axis], dilation[axis], stride[axis])) {
                return false;
            }
        }
        return true;
    }

    private static boolean task0064FloatCarrier(DataType type) {
        return type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }

    private static boolean task0064Canonical(TensorDescriptor descriptor, int rank) {
        return descriptor.shape().rank() == rank
                && task0064Bounded(descriptor)
                && descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()));
    }

    private static boolean task0064Image(
            TensorDescriptor descriptor, int rank, boolean allowSingletonHeightView) {
        if (descriptor.shape().rank() != rank || !task0064Bounded(descriptor)) return false;
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        if (layout.equals(LayoutDescriptor.contiguous(descriptor.shape()))) return true;
        if (!allowSingletonHeightView || rank != 4
                || descriptor.shape().toLongArray()[2] != 1L
                || layout.storageOffset() != 0L || !layout.isView()) {
            return false;
        }
        long[] shape = descriptor.shape().toLongArray();
        long[] expected = new long[] {
            Math.multiplyExact(shape[1], shape[3]), shape[3], shape[3], 1L
        };
        return Arrays.equals(layout.strides(), expected)
                && layout.referencedElementSpan()
                        == Math.multiplyExact(Math.multiplyExact(shape[0], shape[1]), shape[3]);
    }

    private static boolean task0064Bounded(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty()) return false;
        long elements = 1L;
        for (long extent : descriptor.shape().toLongArray()) {
            if (!task0064Unsigned(extent)) return false;
            elements = Math.multiplyExact(elements, extent);
            if (elements > UINT32_MAX) return false;
        }
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        return layout.referencedElementSpan() <= UINT32_MAX;
    }

    private static long task0064WindowExtent(
            long input, long kernel, long stride, long padding, long dilation, boolean ceil) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1L), 1L);
    if (effective > UINT32_MAX) throw new ArithmeticException("effective kernel exceeds uint32");
        long padded = Math.addExact(input, Math.multiplyExact(2L, padding));
        if (padded > UINT32_MAX || padded < effective) {
            throw new ArithmeticException("window does not fit");
        }
        long numerator = padded - effective;
        long result = Math.addExact(
                numerator / stride + (ceil && numerator % stride != 0L ? 1L : 0L), 1L);
        if (result > UINT32_MAX) throw new ArithmeticException("window count exceeds uint32");
        if (result - 1L > UINT32_MAX / stride) {
            throw new ArithmeticException("window origin exceeds uint32");
        }
        return result;
    }

    private static boolean task0064Unsigned(long value) {
        return value > 0L && value <= UINT32_MAX;
    }

    private static boolean task0064UnsignedOrZero(long value) {
        return value >= 0L && value <= UINT32_MAX;
    }

    private static boolean supportsMatmul(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (operation.attrs() != NoOperationAttrs.INSTANCE || inputs.size() != 2) return false;
        TensorDescriptor left = inputs.get(0);
        TensorDescriptor right = inputs.get(1);
        if (!matmulInput(left)
                || !matmulInput(right)
                || !canonicalAny(output, true)
                || output.requiresGrad() != (left.requiresGrad() || right.requiresGrad())
                || !matmulShapeMatches(left, right, output)) {
            return false;
        }
        DataType leftType = left.dataType();
        DataType rightType = right.dataType();
        DataType outputType = output.dataType();
        boolean leftIntegral = leftType == DataType.INT32 || leftType == DataType.INT64;
        boolean rightIntegral = rightType == DataType.INT32 || rightType == DataType.INT64;
        if (leftIntegral && rightIntegral) {
            DataType promoted = leftType == DataType.INT64 || rightType == DataType.INT64
                    ? DataType.INT64 : DataType.INT32;
            return outputType == promoted
                    && !left.requiresGrad()
                    && !right.requiresGrad()
                    && !output.requiresGrad();
        }
        if (outputType != DataType.FLOAT32) {
            return false;
        }
        if (leftType == DataType.FLOAT32 && rightType == DataType.FLOAT32) return true;
        boolean mixedBfloat32 =
                leftType == DataType.BFLOAT16 && rightType == DataType.FLOAT32
                        || leftType == DataType.FLOAT32 && rightType == DataType.BFLOAT16;
        return mixedBfloat32
                && !left.requiresGrad()
                && !right.requiresGrad()
                && !output.requiresGrad();
    }

    private static boolean matmulShapeMatches(
            TensorDescriptor left, TensorDescriptor right, TensorDescriptor output) {
        long[] leftShape = left.shape().toLongArray();
        long[] rightShape = right.shape().toLongArray();
        long[] outputShape = output.shape().toLongArray();
        return matmulShapeMatches(leftShape, rightShape, outputShape);
    }

    static boolean matmulShapeMatches(
            long[] leftShape, long[] rightShape, long[] outputShape) {
        if (leftShape.length == 0 || rightShape.length == 0) return false;
        int leftBatch = Math.max(0, leftShape.length - 2);
        int rightBatch = Math.max(0, rightShape.length - 2);
        int batchRank = Math.max(leftBatch, rightBatch);
        int expectedRank = batchRank
                + (leftShape.length > 1 ? 1 : 0)
                + (rightShape.length > 1 ? 1 : 0);
        long leftContract = leftShape[leftShape.length - 1];
        long rightContract =
                rightShape[rightShape.length == 1 ? 0 : rightShape.length - 2];
        if (leftContract != rightContract || outputShape.length != expectedRank) return false;
        for (int axis = 0; axis < batchRank; axis++) {
            int leftAxis = axis - (batchRank - leftBatch);
            int rightAxis = axis - (batchRank - rightBatch);
            long leftExtent = leftAxis < 0 ? 1L : leftShape[leftAxis];
            long rightExtent = rightAxis < 0 ? 1L : rightShape[rightAxis];
            if (leftExtent != rightExtent && leftExtent != 1L && rightExtent != 1L) return false;
            if (outputShape[axis] != Math.max(leftExtent, rightExtent)) return false;
        }
        int outputAxis = batchRank;
        if (leftShape.length > 1 && outputShape[outputAxis++] != leftShape[leftShape.length - 2]) {
            return false;
        }
        return rightShape.length == 1
                || outputShape[outputAxis] == rightShape[rightShape.length - 1];
    }

    private static boolean matmulInput(TensorDescriptor descriptor) {
        int rank = descriptor.shape().rank();
        if (!geometry(descriptor, descriptor.dataType(), false)) return false;
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        if (layout.equals(LayoutDescriptor.contiguous(descriptor.shape()))) return true;
        if (rank < 2 || layout.storageOffset() != 0L || !layout.isView()) return false;
        long[] logical = descriptor.shape().toLongArray();
        long[] source = logical.clone();
        long swap = source[rank - 2];
        source[rank - 2] = source[rank - 1];
        source[rank - 1] = swap;
        long[] sourceStrides = LayoutDescriptor.contiguous(Shape.of(source)).strides();
        long[] expected = sourceStrides.clone();
        expected[rank - 2] = sourceStrides[rank - 1];
        expected[rank - 1] = sourceStrides[rank - 2];
        return layout.equals(LayoutDescriptor.of(
                descriptor.shape(), expected, 0L, true));
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



    private static boolean supportsMeanSquaredError(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (!(operation.attrs() instanceof MeanSquaredErrorAttrs attrs)
                || inputs.size() != 2) {
            return false;
        }
        TensorDescriptor prediction = inputs.get(0);
        TensorDescriptor target = inputs.get(1);
        if (!canonicalTyped(prediction, DataType.FLOAT32)
                || !canonicalTyped(target, DataType.FLOAT32)
                || !prediction.shape().equals(target.shape())
                || output.requiresGrad()
                        != (prediction.requiresGrad() || target.requiresGrad())) {
            return false;
        }
        if (attrs.reduction() == LossReduction.NONE) {
            return canonicalTyped(output, DataType.FLOAT32)
                    && output.shape().equals(prediction.shape());
        }
        return (attrs.reduction() == LossReduction.SUM
                        || attrs.reduction() == LossReduction.MEAN)
                && output.dataType() == DataType.FLOAT32
                && canonicalReductionOutput(output)
                && output.shape().rank() == 0;
    }

    private static boolean supportsTask0069ScatterAdd(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (!(operation.attrs() instanceof IndexAxisAttrs attrs)
                || attrs.axis() != 0
                || inputs.size() != 3) {
            return false;
        }
        TensorDescriptor data = inputs.get(0);
        TensorDescriptor indices = inputs.get(1);
        TensorDescriptor updates = inputs.get(2);
        if (!canonicalTyped(data, DataType.FLOAT32)
                || !canonicalAny(indices, false)
                || !canonicalTyped(updates, DataType.FLOAT32)
                || !canonicalTyped(output, DataType.FLOAT32)
                || data.shape().rank() != 1
                || indices.shape().rank() != 1
                || updates.shape().rank() != 1
                || output.shape().rank() != 1
                || indices.requiresGrad()
                || data.requiresGrad()
                || updates.requiresGrad()
                || output.requiresGrad()
                || !supportedStorageLayout(indices, 1)
                || (indices.dataType() != DataType.INT32
                        && indices.dataType() != DataType.INT64)) {
            return false;
        }
        long dataExtent = data.shape().toLongArray()[0];
        long updateExtent = updates.shape().toLongArray()[0];
        if (dataExtent != output.shape().toLongArray()[0]
                || updateExtent != indices.shape().toLongArray()[0]
                || dataExtent < 1L
                || updateExtent < 1L
                || dataExtent > UINT32_MAX / Float.BYTES
                || updateExtent > UINT32_MAX / Float.BYTES
                || updateExtent > UINT32_MAX / indices.dataType().byteWidth()) {
            return false;
        }
        long indexSpan = indices.layout().orElseThrow().referencedElementSpan();
        return indexSpan <= UINT32_MAX / indices.dataType().byteWidth();
    }

    private static boolean supportsTask0069L1Norm(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (!(operation.attrs() instanceof MultiAxisReductionAttrs attrs)
                || !attrs.axes().equals(List.of(0))
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonical(input)
                || !canonicalReductionOutput(output)
                || input.requiresGrad()
                || output.requiresGrad()
                || input.shape().rank() != 1) {
            return false;
        }
        long extent = input.shape().toLongArray()[0];
        if (extent < 1L || extent > UINT32_MAX / Float.BYTES) {
            return false;
        }
        long[] expected = attrs.keepDimensions() ? new long[] {1L} : new long[0];
        return Arrays.equals(expected, output.shape().toLongArray())
                && output.layout().orElseThrow().referencedElementSpan() == 1L;
    }

    private static boolean supportsTask0069Variance(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        if (!(operation.attrs() instanceof StatisticalReductionAttrs attrs)
                || !attrs.axes().equals(List.of(0))
                || attrs.correction() != 0
                || inputs.size() != 1) {
            return false;
        }
        TensorDescriptor input = inputs.getFirst();
        if (!canonical(input)
                || !canonicalReductionOutput(output)
                || input.requiresGrad()
                || output.requiresGrad()
                || input.shape().rank() != 1
                || input.shape().toLongArray()[0] != 1L) {
            return false;
        }
        long[] expected = attrs.keepDimensions() ? new long[] {1L} : new long[0];
        return Arrays.equals(expected, output.shape().toLongArray())
                && output.layout().orElseThrow().referencedElementSpan() == 1L;
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
        boolean gradientsValid = input.requiresGrad() == output.requiresGrad();
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
        if (!selectedAffineRead(input, true)
                || !selectedAffineRead(output, true)
                || input.dataType() != output.dataType()
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
                if (!inputLayout.isContiguous()
                        || input.shape().knownElementCount().orElseThrow()
                                != output.shape().knownElementCount().orElseThrow()) {
                    return false;
                }
                expected =
                        LayoutDescriptor.of(
                                output.shape(),
                                LayoutDescriptor.contiguous(output.shape()).strides(),
                                inputLayout.storageOffset(),
                                true);
            } else if (kind == ShapeTransformKind.EXPAND) {
                long[] inputShape = input.shape().toLongArray();
                long[] outputShape = output.shape().toLongArray();
                if (inputShape.length > outputShape.length) return false;
                long[] strides = new long[outputShape.length];
                int padding = outputShape.length - inputShape.length;
                for (int axis = 0; axis < inputShape.length; axis++) {
                    long source = inputShape[axis];
                    long target = outputShape[axis + padding];
                    if (source != target && source != 1L) return false;
                    strides[axis + padding] =
                            source == 1L && target != 1L ? 0L : inputLayout.stride(axis);
                }
                expected =
                        LayoutDescriptor.of(
                                output.shape(), strides, inputLayout.storageOffset(), true);
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
                    if (source < 0 || source >= inputShape.length || seen[source]) return false;
                    seen[source] = true;
                    expectedShape[axis] = inputShape[source];
                    strides[axis] = inputLayout.stride(source);
                }
                if (!Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                expected =
                        LayoutDescriptor.of(
                                output.shape(), strides, inputLayout.storageOffset(), true);
            } else {
                if (!(operation.attrs() instanceof AxisTransformAttrs attrs)) return false;
                int axis = attrs.axis();
                if (kind == AxisTransformKind.EXPAND_DIMS) {
                    if (axis < 0
                            || axis > inputShape.length
                            || output.shape().rank() != inputShape.length + 1) {
                        return false;
                    }
                    long[] expectedShape = new long[inputShape.length + 1];
                    long[] strides = new long[inputShape.length + 1];
                    for (int outputAxis = 0; outputAxis < expectedShape.length; outputAxis++) {
                        if (outputAxis == axis) {
                            expectedShape[outputAxis] = 1L;
                            strides[outputAxis] =
                                    outputAxis == inputShape.length
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
                    if (!Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                    expected =
                            LayoutDescriptor.of(
                                    output.shape(), strides, inputLayout.storageOffset(), true);
                } else if (kind == AxisTransformKind.SQUEEZE) {
                    if (axis < 0
                            || axis >= inputShape.length
                            || inputShape[axis] != 1L
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
                    if (!Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                    expected =
                            LayoutDescriptor.of(
                                    output.shape(), strides, inputLayout.storageOffset(), true);
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
        return selectedAffineRead(input, true)
                && canonicalAny(output, true)
                && input.dataType() == output.dataType()
                && input.shape().equals(output.shape())
                && input.requiresGrad() == output.requiresGrad();
    }

    private static boolean selectedIndexRead(TensorDescriptor descriptor, boolean allowScalar) {
        return !descriptor.requiresGrad()
                && (descriptor.dataType() == DataType.INT32
                        || descriptor.dataType() == DataType.INT64)
                && selectedAffineRead(descriptor, allowScalar);
    }

    private static boolean selectedAffineRead(TensorDescriptor descriptor, boolean allowScalar) {
        int rank = descriptor.shape().rank();
        if (!descriptor.shape().isFullyStatic()
                || rank < (allowScalar ? 0 : 1)
                || rank > MetalMpsGraphProgram.MAX_RANK
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) return false;
            elements = Math.multiplyExact(elements, dimension);
        }
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        Math.multiplyExact(elements, descriptor.dataType().byteWidth());
        Math.multiplyExact(layout.referencedElementSpan(), descriptor.dataType().byteWidth());
        if (layout.hasZeroStride()) {
            return layout.isView();
        }
        return supportedStorageLayout(descriptor, allowScalar ? 0 : 1);
    }

    private static boolean isFloating(DataType dataType) {
        return dataType == DataType.FLOAT64
                || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16
                || dataType == DataType.FLOAT16;
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
