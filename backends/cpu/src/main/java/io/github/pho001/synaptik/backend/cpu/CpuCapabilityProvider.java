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
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the executable semantic coverage currently delivered by the CPU backend.
 *
 * <p>The provider has a stable CPU ownership identity and advertises the bounded, fully static
 * pointwise matrix implemented by the portable route: selected same-type arithmetic including
 * extrema and floating Tensor power, exact scalar arithmetic and floating range clamp,
 * canonical-BOOL logic, all nineteen same-typed FLOAT32/FLOAT64 unary semantics, floating
 * classification, comparisons, floating {@code WHERE}, and same-type {@code CAST}. Every
 * descriptor has a resolved layout, and results obey the
 * Model family's shape rule. The provider also admits the exact one-input, one-output, fully
 * static and resolved-layout occurrences of {@code CONTIGUOUS}, {@code RESHAPE}, {@code EXPAND},
 * {@code PERMUTE}, {@code EXPAND_DIMS}, {@code SQUEEZE}, scalar {@code SELECT}, and positive-step
 * {@code SLICE}, including target-relative crop attributes. These affine rows preserve one data
 * type and must carry the exact layout implied by their Model semantics. The provider also
 * admits one fully static, resolved-layout {@code PAD}, {@code TILE}, {@code CONCAT}, or
 * {@code STACK} occurrence for all six represented data types. Movement inputs preserve their
 * exact semantic occurrence order, the output layout must be injective, and composition is
 * bounded to one through sixteen occurrences. The same movement route admits one
 * {@code UNFOLD_AXIS} occurrence for all six represented types or one floating
 * {@code UNFOLD2D} occurrence with direct positive-zero or exact typed padding. Both window
 * forms require their exact static result geometry and one distinct injective output. A separate
 * one-node indexing matrix admits {@code GATHER}, {@code GATHER_ELEMENTS}, {@code GATHER_ND}, and
 * {@code ONE_HOT} with INT32/INT64 indices, exact static result geometry, and an injective
 * output. A separate one-node functional-scatter matrix admits current {@code SCATTER_ELEMENTS},
 * Gather-compatible fixed-add {@code SCATTER_ADD}, and {@code SCATTER_ND}, with exact Model
 * shapes, permitted represented reductions, and a distinct injective data-shaped output.</p>
 *
 * <p>A distinct one-node fold matrix admits numeric {@code FOLD_AXIS} and floating
 * {@code FOLD2D}. Fold requires exact current static geometry, a distinct injective result,
 * canonical represented addition, and no implicit base, workspace, or padding value.</p>
 *
 * <p>A distinct one-node ordering matrix admits stable {@code SORT}, {@code ARGSORT}, and
 * two-output {@code TOP_K} for all six represented types. It requires exact static Shape and
 * output-role relationships, resolved non-negative layouts, and injective outputs. This
 * occurrence-local capability records Model order only; CPU lowering and binding remain
 * responsible for exact scratch, carrier compatibility, and physical non-overlap.</p>
 *
 * <p>A distinct one-node cumulative-scan matrix admits {@code CUM_SUM} and {@code CUM_PROD}
 * across FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64. It requires one non-scalar static Shape,
 * the same input/output type and Shape, a valid normalized axis, resolved non-negative layouts,
 * and an injective output. Capability covers inclusive/exclusive and forward/reverse modes;
 * lowering and binding remain responsible for exact sequential per-slice realization, carrier
 * compatibility, and complete physical non-overlap.</p>
 *
 * <p>A distinct one-node ordinary aggregate matrix admits MIN and MAX for FLOAT64, FLOAT32,
 * BFLOAT16, INT32, and INT64, plus ALL and ANY for canonical BOOL. Exact parameterless full,
 * single-axis, and multi-axis attributes are supported with fully static Model-derived output
 * Shapes, resolved non-negative layouts, and an injective output. Lowering and binding retain
 * responsibility for canonical selected-axis membership, complete-domain traversal, exact
 * carriers, and complete physical non-overlap.</p>
 *
 * <p>The movement route also admits exactly one fully static, resolved-layout
 * {@code SLICE_UPDATE} occurrence with ordered {@code [base, update]} inputs. Both normalized
 * signed finite-coordinate {@link SliceAttrs} and target-relative {@link CropToShapeAttrs}
 * placement are supported for all six represented types. The result retains the base Shape and
 * uses a distinct injective layout; its semantic effect is to copy base values and replace only
 * selected positions, without mutating either input.</p>
 *
 * <p>Complete-partition lowering remains stricter: it validates either one supported movement,
 * indexing, functional-scatter, overlap-fold, ordering, random, cumulative-scan, or ordinary
 * aggregate occurrence, a connected one-to-eight pointwise
 * chain, or a connected one-to-eight affine chain, then applies exact layout, alias, fan-out,
 * publication, and partition-boundary checks before resource declaration. Occurrence support
 * therefore does not promise that an arbitrary mixed or branched partition can be prepared.</p>
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
     * {@code WHERE} applies branch-first then condition broadcasting. Admitted affine operations
     * require one input, one output, the same data type, fully static Shapes, resolved layouts,
     * and their exact current attributes and descriptor relationship. Admitted movement
     * operations additionally require exact static PAD/TILE/composition/window/slice-update shape
     * relationships, an injective result layout, and at most sixteen composition occurrences.
     * General-axis unfold admits every represented type; NCHW two-dimensional unfold admits only
     * FLOAT64, FLOAT32, and BFLOAT16 with exact matching padding type. Indexing rows additionally
     * require INT32/INT64 indices and their exact current Shape
     * formulas; run-bound value checks remain an execution responsibility rather than capability
     * inspection. Functional scatter additionally requires ordered data/indices/updates, exact
     * data/update type equality, current family attributes and shape formulas, and rejects BOOL
     * arithmetic. Fold additionally requires exact current one-input Shapes and admits INT32/
     * INT64 only for general-axis fold. Ordering additionally requires one input, exact one- or
     * two-output roles, and an injective resolved output layout; execution supplies stable
     * NaN-last/signed-zero order, logical INT64 indices, and represented-bit value copies.
     * Cumulative scans additionally require a non-scalar input, the same five-type numeric input
     * and output descriptor, a valid normalized axis, and an injective output layout.
     * Ordinary aggregates additionally require the exact five-type extrema or BOOL fold matrix,
     * one exact ordinary attribute form, the Model-derived output Shape, and an injective output.
     * Cross-type casts, dynamic
     * or unresolved geometry, negative-step extraction slices, non-injective
     * movement outputs, and all rows outside the implemented matrix return {@code false} without
     * defining conversion or fallback behavior. Negative and non-unit steps are supported for
     * {@code SLICE_UPDATE}; they describe logical placement and do not create a negative storage
     * stride.
     *
     * @param query non-null immutable operation occurrence to validate structurally
     * @return {@code true} only for the exact implemented occurrence-local matrix; otherwise
     *     {@code false}; complete-partition eligibility may still be stricter
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        Object requestedKind = query.operation().kind();
        int expectedOutputs = requestedKind == TopKKind.TOP_K ? 2
                : requestedKind == DropoutKind.DROPOUT ? 3 : 1;
        if (query.outputs().size() != expectedOutputs || !query.inputs().stream().allMatch(CpuCapabilityProvider::staticResolved)
                || !query.outputs().stream().allMatch(CpuCapabilityProvider::staticResolved)) {
            return false;
        }
        var kind = query.operation().kind();
        var attrs = query.operation().attrs();
        TensorDescriptor output = query.outputs().getFirst();
        try {
            if (kind instanceof AggregateReductionKind aggregate)
                return supportsAggregate(query, output, aggregate);
            if (kind instanceof CumulativeScanKind) return supportsScan(query, output);
            if (kind == GraphRngKind.INITIAL_STATE) return supportsInitialState(query);
            if (kind == DropoutKind.DROPOUT) return supportsDropout(query);
            if (kind instanceof OrderingKind || kind == TopKKind.TOP_K)
                return supportsOrdering(query);
            if (kind instanceof AxisGatherKind || kind == GatherNdKind.GATHER_ND
                    || kind == OneHotKind.ONE_HOT) return supportsIndexing(query, output);
            if (kind instanceof AxisScatterKind || kind == ScatterNdKind.SCATTER_ND) {
                return supportsScatter(query, output);
            }
            if (kind == WindowTransformKind.FOLD_AXIS || kind == WindowTransformKind.FOLD2D) {
                return supportsFold(query, output);
            }
            if (movementKind(kind)) return supportsMovement(query, output);
            if (affineKind(kind)) return supportsAffine(query, output);
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
                        && floating(query.inputs().getFirst().dataType())
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
        } catch (IllegalArgumentException | ArithmeticException incompatible) { return false; }
        return false;
    }

    private static boolean supportsAggregate(OperationCapabilityQuery query,
            TensorDescriptor output, AggregateReductionKind kind) {
        if (query.inputs().size() != 1 || kind != AggregateReductionKind.MIN
                && kind != AggregateReductionKind.MAX && kind != AggregateReductionKind.ALL
                && kind != AggregateReductionKind.ANY) return false;
        TensorDescriptor input = query.inputs().getFirst();
        boolean bool = kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY;
        boolean numeric = input.dataType() == DataType.FLOAT64 || input.dataType() == DataType.FLOAT32
                || input.dataType() == DataType.BFLOAT16 || input.dataType() == DataType.INT32
                || input.dataType() == DataType.INT64;
        if (bool != (input.dataType() == DataType.BOOL) || !bool && !numeric
                || output.dataType() != input.dataType()) return false;
        Object attrs = query.operation().attrs(); int rank = input.shape().rank();
        boolean keep; int[] axes;
        if (attrs == NoOperationAttrs.INSTANCE) { keep = false; axes = new int[rank];
            for (int axis = 0; axis < rank; axis++) axes[axis] = axis;
        } else if (attrs instanceof AxisReductionAttrs axis) {
            if (axis.axis() < 0 || axis.axis() >= rank) return false;
            keep = axis.keepDimensions(); axes = new int[] {axis.axis()};
        } else if (attrs instanceof MultiAxisReductionAttrs multi) {
            keep = multi.keepDimensions(); axes = multi.axes().stream().mapToInt(Integer::intValue).toArray();
            boolean[] seen = new boolean[rank];
            for (int axis : axes) if (axis < 0 || axis >= rank || seen[axis]) return false;
                else seen[axis] = true;
        } else return false;
        boolean[] selected = new boolean[rank]; for (int axis : axes) selected[axis] = true;
        long[] in = input.shape().toLongArray(); var expected = new java.util.ArrayList<Long>();
        if (attrs != NoOperationAttrs.INSTANCE) for (int axis = 0; axis < rank; axis++) {
            if (!selected[axis]) expected.add(in[axis]); else if (keep) expected.add(1L);
        }
        if (!java.util.Arrays.equals(output.shape().toLongArray(),
                expected.stream().mapToLong(Long::longValue).toArray())) return false;
        return injective(output.shape().toLongArray(), output.layout().orElseThrow().strides());
    }

    private static boolean supportsScan(OperationCapabilityQuery query, TensorDescriptor output) {
        if (!(query.operation().attrs() instanceof CumulativeScanAttrs attrs)
                || query.inputs().size() != 1) return false;
        TensorDescriptor input = query.inputs().getFirst();
        if (input.shape().rank() == 0 || attrs.axis() >= input.shape().rank()
                || input.dataType() == DataType.BOOL || input.dataType() != output.dataType()
                || !input.shape().equals(output.shape())) return false;
        return injective(output.shape().toLongArray(), output.layout().orElseThrow().strides());
    }

    private static boolean supportsInitialState(OperationCapabilityQuery query) {
        return query.operation().attrs() instanceof GraphRngStateAttrs
                && query.inputs().isEmpty() && query.outputs().size() == 1
                && stateDescriptor(query.outputs().getFirst())
                && injective(query.outputs().getFirst().shape().toLongArray(),
                    query.outputs().getFirst().layout().orElseThrow().strides());
    }

    private static boolean supportsDropout(OperationCapabilityQuery query) {
        if (!(query.operation().attrs() instanceof DropoutAttrs) || query.inputs().size() != 2
                || query.outputs().size() != 3) return false;
        TensorDescriptor value = query.inputs().get(0), state = query.inputs().get(1);
        TensorDescriptor output = query.outputs().get(0), mask = query.outputs().get(1);
        TensorDescriptor next = query.outputs().get(2);
        if ((value.dataType() != DataType.FLOAT64 && value.dataType() != DataType.FLOAT32)
                || output.dataType() != value.dataType() || !output.shape().equals(value.shape())
                || mask.dataType() != DataType.BOOL || !mask.shape().equals(value.shape())
                || !stateDescriptor(state) || !stateDescriptor(next)) return false;
        return injective(state.shape().toLongArray(), state.layout().orElseThrow().strides())
                && query.outputs().stream().allMatch(descriptor -> injective(
                descriptor.shape().toLongArray(), descriptor.layout().orElseThrow().strides()));
    }

    private static boolean stateDescriptor(TensorDescriptor descriptor) {
        return descriptor.dataType() == DataType.INT64
                && java.util.Arrays.equals(descriptor.shape().toLongArray(), new long[] {2});
    }

    private static boolean supportsOrdering(OperationCapabilityQuery query) {
        if (query.inputs().size() != 1) return false;
        TensorDescriptor input = query.inputs().getFirst();
        int axis;
        long k;
        if (query.operation().kind() instanceof OrderingKind kind
                && query.operation().attrs() instanceof SortAttrs attrs) {
            axis = attrs.axis();
            if (axis >= input.shape().rank()) return false;
            k = input.shape().toLongArray()[axis];
            TensorDescriptor result = query.outputs().getFirst();
            if (!result.shape().equals(input.shape())
                    || kind == OrderingKind.SORT && result.dataType() != input.dataType()
                    || kind == OrderingKind.ARGSORT && result.dataType() != DataType.INT64) return false;
        } else if (query.operation().kind() == TopKKind.TOP_K
                && query.operation().attrs() instanceof TopKAttrs attrs) {
            axis = attrs.axis(); k = attrs.k();
            if (axis >= input.shape().rank() || k > input.shape().toLongArray()[axis]) return false;
            long[] expected = input.shape().toLongArray(); expected[axis] = k;
            if (query.outputs().get(0).dataType() != input.dataType()
                    || query.outputs().get(1).dataType() != DataType.INT64
                    || !java.util.Arrays.equals(query.outputs().get(0).shape().toLongArray(), expected)
                    || !java.util.Arrays.equals(query.outputs().get(1).shape().toLongArray(), expected)) return false;
        } else return false;
        if (axis < 0 || axis >= input.shape().rank()) return false;
        for (TensorDescriptor result : query.outputs()) if (!injective(result.shape().toLongArray(),
                result.layout().orElseThrow().strides())) return false;
        return k >= 0;
    }

    private static boolean staticResolved(TensorDescriptor descriptor) {
        return descriptor.shape().isFullyStatic() && descriptor.layout().isPresent()
                && descriptor.layout().orElseThrow().storageOffset() >= 0
                && java.util.Arrays.stream(descriptor.layout().orElseThrow().strides())
                        .allMatch(stride -> stride >= 0);
    }

    private static boolean supportsIndexing(OperationCapabilityQuery query,
            TensorDescriptor output) {
        if (!injective(output.shape().toLongArray(), output.layout().orElseThrow().strides())) {
            return false;
        }
        Object kind = query.operation().kind();
        Object attrs = query.operation().attrs();
        if (kind == OneHotKind.ONE_HOT) {
            if (!(attrs instanceof OneHotAttrs hot) || query.inputs().size() != 1
                    || !indexType(query.inputs().getFirst().dataType())
                    || output.dataType() != DataType.BOOL) return false;
            long[] input = query.inputs().getFirst().shape().toLongArray();
            long[] result = output.shape().toLongArray();
            if (result.length != input.length + 1 || result[result.length - 1] != hot.depth()) return false;
            for (int i = 0; i < input.length; i++) if (input[i] != result[i]) return false;
            return true;
        }
        if (query.inputs().size() != 2) return false;
        TensorDescriptor data = query.inputs().get(0), indices = query.inputs().get(1);
        if (!indexType(indices.dataType()) || output.dataType() != data.dataType()) return false;
        long[] d = data.shape().toLongArray(), i = indices.shape().toLongArray();
        long[] result = output.shape().toLongArray();
        if (kind instanceof AxisGatherKind axisKind && attrs instanceof IndexAxisAttrs axisAttrs) {
            int axis = axisAttrs.axis();
            if (axis >= d.length) return false;
            if (axisKind == AxisGatherKind.GATHER) {
                if (result.length != d.length - 1 + i.length) return false;
                int out = 0;
                for (int a = 0; a < axis; a++) if (result[out++] != d[a]) return false;
                for (long extent : i) if (result[out++] != extent) return false;
                for (int a = axis + 1; a < d.length; a++) if (result[out++] != d[a]) return false;
                return true;
            }
            if (i.length != d.length || !java.util.Arrays.equals(result, i)) return false;
            for (int a = 0; a < d.length; a++) if (a != axis && d[a] != i[a]) return false;
            return true;
        }
        if (kind == GatherNdKind.GATHER_ND && attrs instanceof GatherNdAttrs nd) {
            int batch = nd.batchDimensions();
            if (batch < 0 || batch >= Math.min(d.length, i.length) || i.length == 0) return false;
            long tupleLong = i[i.length - 1];
            if (tupleLong < 1 || tupleLong > d.length - batch) return false;
            int tuple = Math.toIntExact(tupleLong);
            for (int a = 0; a < batch; a++) if (d[a] != i[a]) return false;
            if (result.length != i.length - 1 + d.length - batch - tuple) return false;
            int out = 0;
            for (int a = 0; a < i.length - 1; a++) if (result[out++] != i[a]) return false;
            for (int a = batch + tuple; a < d.length; a++) if (result[out++] != d[a]) return false;
            return true;
        }
        return false;
    }

    private static boolean indexType(DataType type) {
        return type == DataType.INT32 || type == DataType.INT64;
    }

    private static boolean supportsScatter(OperationCapabilityQuery query,
            TensorDescriptor output) {
        if (query.inputs().size() != 3
                || !injective(output.shape().toLongArray(), output.layout().orElseThrow().strides())) {
            return false;
        }
        TensorDescriptor data = query.inputs().get(0);
        TensorDescriptor indices = query.inputs().get(1);
        TensorDescriptor updates = query.inputs().get(2);
        if (!indexType(indices.dataType()) || updates.dataType() != data.dataType()
                || output.dataType() != data.dataType() || !output.shape().equals(data.shape())) {
            return false;
        }
        Object kind = query.operation().kind();
        Object attrs = query.operation().attrs();
        ScatterReduction reduction;
        int axis = -1;
        if (kind == AxisScatterKind.SCATTER_ELEMENTS && attrs instanceof ScatterElementsAttrs a) {
            reduction = a.reduction(); axis = a.axis();
        } else if (kind == AxisScatterKind.SCATTER_ADD && attrs instanceof IndexAxisAttrs a) {
            reduction = ScatterReduction.ADD; axis = a.axis();
        } else if (kind == ScatterNdKind.SCATTER_ND && attrs instanceof ScatterNdAttrs a) {
            reduction = a.reduction();
        } else return false;
        if (data.dataType() == DataType.BOOL && reduction != ScatterReduction.NONE) return false;
        long[] d = data.shape().toLongArray();
        long[] i = indices.shape().toLongArray();
        long[] u = updates.shape().toLongArray();
        if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
            if (d.length == 0 || axis >= d.length || !java.util.Arrays.equals(i, u)
                    || i.length != d.length) return false;
            for (int a = 0; a < d.length; a++) if (a != axis && d[a] != i[a]) return false;
            return true;
        }
        if (kind == AxisScatterKind.SCATTER_ADD) {
            if (d.length == 0 || axis >= d.length || u.length != d.length - 1 + i.length) return false;
            int p = 0;
            for (int a = 0; a < axis; a++) if (u[p++] != d[a]) return false;
            for (long extent : i) if (u[p++] != extent) return false;
            for (int a = axis + 1; a < d.length; a++) if (u[p++] != d[a]) return false;
            return true;
        }
        int batch = ((ScatterNdAttrs) attrs).batchDimensions();
        if (i.length == 0 || batch >= Math.min(d.length, i.length)) return false;
        long tupleLong = i[i.length - 1];
        if (tupleLong < 1 || tupleLong > d.length - batch) return false;
        int tuple = Math.toIntExact(tupleLong);
        for (int a = 0; a < batch; a++) if (d[a] != i[a]) return false;
        if (u.length != i.length - 1 + d.length - batch - tuple) return false;
        int p = 0;
        for (int a = 0; a < i.length - 1; a++) if (u[p++] != i[a]) return false;
        for (int a = batch + tuple; a < d.length; a++) if (u[p++] != d[a]) return false;
        return true;
    }

    private static boolean supportsFold(OperationCapabilityQuery query, TensorDescriptor output) {
        if (query.inputs().size() != 1
                || !injective(output.shape().toLongArray(), output.layout().orElseThrow().strides())) {
            return false;
        }
        TensorDescriptor input = query.inputs().getFirst();
        if (input.dataType() != output.dataType() || input.dataType() == DataType.BOOL) return false;
        long[] source = input.shape().toLongArray();
        long[] result = output.shape().toLongArray();
        Object kind = query.operation().kind();
        Object attrs = query.operation().attrs();
        if (kind == WindowTransformKind.FOLD_AXIS && attrs instanceof FoldAxisAttrs fold) {
            if (source.length < 2 || result.length != source.length - 1
                    || fold.axis() >= result.length || result[fold.axis()] != fold.outputSize()) {
                return false;
            }
            for (int axis = 0; axis < result.length; axis++) {
                if (axis != fold.axis() && result[axis] != source[axis]) return false;
            }
            long windows = source[fold.axis()], size = source[source.length - 1];
            if (size <= 0) return false;
            if (fold.outputSize() == 0) return windows == 0;
            if (size > fold.outputSize()) return false;
            long expected = Math.addExact((fold.outputSize() - size) / fold.step(), 1);
            Math.addExact(Math.multiplyExact(expected - 1, fold.step()), size - 1);
            return windows == expected;
        }
        if (kind == WindowTransformKind.FOLD2D && attrs instanceof Fold2dAttrs fold) {
            if (!windowFloating(input.dataType()) || source.length != 3 || result.length != 4
                    || !output.shape().equals(fold.outputShape())) return false;
            Window2dAttrs window = fold.window();
            long effectiveHeight = Math.addExact(Math.multiplyExact(window.dilationHeight(),
                    window.kernelHeight() - 1), 1);
            long effectiveWidth = Math.addExact(Math.multiplyExact(window.dilationWidth(),
                    window.kernelWidth() - 1), 1);
            long numeratorHeight = Math.subtractExact(Math.addExact(result[2],
                    Math.multiplyExact(2, window.paddingHeight())), effectiveHeight);
            long numeratorWidth = Math.subtractExact(Math.addExact(result[3],
                    Math.multiplyExact(2, window.paddingWidth())), effectiveWidth);
            if (numeratorHeight < 0 || numeratorWidth < 0) return false;
            long columnsHeight = windowOutput(numeratorHeight, window.strideHeight(),
                    window.ceilMode());
            long columnsWidth = windowOutput(numeratorWidth, window.strideWidth(),
                    window.ceilMode());
            return source[0] == result[0]
                    && source[1] == Math.multiplyExact(Math.multiplyExact(result[1],
                            window.kernelHeight()), window.kernelWidth())
                    && source[2] == Math.multiplyExact(columnsHeight, columnsWidth);
        }
        return false;
    }

    private static boolean affineKind(Object kind) {
        return kind == ContiguousKind.CONTIGUOUS || kind instanceof ShapeTransformKind
                || kind instanceof AxisTransformKind || kind == SelectKind.SELECT
                || kind == SliceKind.SLICE;
    }

    private static boolean movementKind(Object kind) {
        return kind == PadKind.PAD || kind == TileKind.TILE
                || kind == TensorCompositionKind.CONCAT
                || kind == TensorCompositionKind.STACK
                || kind == WindowTransformKind.UNFOLD_AXIS
                || kind == WindowTransformKind.UNFOLD2D
                || kind == SliceKind.SLICE_UPDATE;
    }

    private static boolean supportsMovement(OperationCapabilityQuery query,
            TensorDescriptor output) {
        if (query.inputs().isEmpty() || query.inputs().size() > 16
                || !injective(output.shape().toLongArray(),
                    output.layout().orElseThrow().strides())) return false;
        if (query.inputs().stream().anyMatch(input -> input.dataType() != output.dataType())) {
            return false;
        }
        Object kind = query.operation().kind();
        Object attrs = query.operation().attrs();
        if (kind == SliceKind.SLICE_UPDATE) {
            return supportsSliceUpdate(query, output, attrs);
        }
        if (kind == WindowTransformKind.UNFOLD_AXIS && attrs instanceof UnfoldAxisAttrs unfold) {
            if (query.inputs().size() != 1) return false;
            long[] input = query.inputs().getFirst().shape().toLongArray();
            long[] result = output.shape().toLongArray();
            if (input.length == 0 || unfold.axis() >= input.length || unfold.size() > input[unfold.axis()]
                    || result.length != input.length + 1) return false;
            long positions = Math.addExact((input[unfold.axis()] - unfold.size()) / unfold.step(), 1);
            Math.addExact(Math.multiplyExact(positions - 1, unfold.step()), unfold.size() - 1);
            for (int axis = 0; axis < result.length; axis++) {
                long expected = axis == input.length ? unfold.size()
                        : axis == unfold.axis() ? positions : input[axis];
                if (result[axis] != expected) return false;
            }
            return true;
        }
        if (kind == WindowTransformKind.UNFOLD2D
                && (attrs instanceof Window2dAttrs || attrs instanceof Unfold2dAttrs)) {
            if (query.inputs().size() != 1 || !windowFloating(output.dataType())) return false;
            Window2dAttrs window;
            if (attrs instanceof Window2dAttrs direct) window = direct;
            else {
                Unfold2dAttrs configured = (Unfold2dAttrs) attrs;
                if (configured.paddingValue().dataType() != output.dataType()) return false;
                window = configured.window();
            }
            long[] input = query.inputs().getFirst().shape().toLongArray();
            long[] result = output.shape().toLongArray();
            if (input.length != 4 || result.length != 3) return false;
            long effectiveH = Math.addExact(Math.multiplyExact(window.dilationHeight(),
                    window.kernelHeight() - 1), 1);
            long effectiveW = Math.addExact(Math.multiplyExact(window.dilationWidth(),
                    window.kernelWidth() - 1), 1);
            long numeratorH = Math.subtractExact(Math.addExact(input[2],
                    Math.multiplyExact(2, window.paddingHeight())), effectiveH);
            long numeratorW = Math.subtractExact(Math.addExact(input[3],
                    Math.multiplyExact(2, window.paddingWidth())), effectiveW);
            if (numeratorH < 0 || numeratorW < 0) return false;
            long outH = windowOutput(numeratorH, window.strideHeight(), window.ceilMode());
            long outW = windowOutput(numeratorW, window.strideWidth(), window.ceilMode());
            return result[0] == input[0]
                    && result[1] == Math.multiplyExact(Math.multiplyExact(input[1],
                        window.kernelHeight()), window.kernelWidth())
                    && result[2] == Math.multiplyExact(outH, outW);
        }
        if (kind == PadKind.PAD && attrs instanceof PadAttrs pad) {
            if (query.inputs().size() != 1) return false;
            long[] input = query.inputs().getFirst().shape().toLongArray();
            long[] result = output.shape().toLongArray();
            if (pad.before().size() != input.length || pad.after().size() != input.length
                    || result.length != input.length
                    || pad.constantValue().dataType() != output.dataType()) return false;
            for (int axis = 0; axis < input.length; axis++) if (result[axis]
                    != Math.addExact(pad.before().get(axis),
                        Math.addExact(input[axis], pad.after().get(axis)))) return false;
            return true;
        }
        if (kind == TileKind.TILE && attrs instanceof TileAttrs tile) {
            if (query.inputs().size() != 1) return false;
            long[] input = query.inputs().getFirst().shape().toLongArray();
            long[] result = output.shape().toLongArray();
            if (tile.repeats().size() != input.length || result.length != input.length) return false;
            for (int axis = 0; axis < input.length; axis++) if (result[axis]
                    != Math.multiplyExact(input[axis], tile.repeats().get(axis))) return false;
            return true;
        }
        if (!(attrs instanceof CompositionAxisAttrs composition)) return false;
        int axis = composition.axis();
        long[] first = query.inputs().getFirst().shape().toLongArray();
        long[] result = output.shape().toLongArray();
        if (kind == TensorCompositionKind.CONCAT) {
            if (first.length == 0 || axis >= first.length || result.length != first.length) return false;
            long selected = 0;
            for (TensorDescriptor input : query.inputs()) {
                long[] shape = input.shape().toLongArray();
                if (shape.length != first.length) return false;
                for (int current = 0; current < first.length; current++) {
                    if (current != axis && shape[current] != first[current]) return false;
                }
                selected = Math.addExact(selected, shape[axis]);
            }
            for (int current = 0; current < result.length; current++) {
                if (result[current] != (current == axis ? selected : first[current])) return false;
            }
            return true;
        }
        if (kind == TensorCompositionKind.STACK) {
            if (axis > first.length || result.length != first.length + 1
                    || query.inputs().stream().anyMatch(input -> !input.shape()
                        .equals(query.inputs().getFirst().shape()))) return false;
            for (int outAxis = 0, inAxis = 0; outAxis < result.length; outAxis++) {
                if (result[outAxis] != (outAxis == axis ? query.inputs().size() : first[inAxis++])) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    private static boolean supportsSliceUpdate(OperationCapabilityQuery query,
            TensorDescriptor output, Object attrs) {
        if (query.inputs().size() != 2) return false;
        TensorDescriptor base = query.inputs().get(0), update = query.inputs().get(1);
        if (base.dataType() != update.dataType() || base.dataType() != output.dataType()
                || !base.shape().equals(output.shape())
                || base.shape().rank() != update.shape().rank()) return false;
        long[] baseShape = base.shape().toLongArray();
        long[] updateShape = update.shape().toLongArray();
        if (attrs instanceof SliceAttrs slice) {
            boolean[] selected = new boolean[baseShape.length];
            for (int index = 0; index < slice.axes().size(); index++) {
                int axis = slice.axes().get(index);
                if (axis >= baseShape.length || selected[axis]
                        || updateShape[axis] != slice.lengths().get(index)) return false;
                selected[axis] = true;
                long length = slice.lengths().get(index);
                if (length > 0) {
                    long start = slice.starts().get(index);
                    long last = Math.addExact(start,
                            Math.multiplyExact(length - 1, slice.steps().get(index)));
                    if (start >= baseShape[axis] || last < 0 || last >= baseShape[axis]) return false;
                }
            }
            for (int axis = 0; axis < baseShape.length; axis++) {
                if (!selected[axis] && updateShape[axis] != baseShape[axis]) return false;
            }
            return true;
        }
        if (attrs instanceof CropToShapeAttrs crop) {
            if (!crop.targetShape().isFullyStatic() || !crop.prefixShape().isFullyStatic()
                    || crop.targetShape().rank() != baseShape.length
                    || crop.prefixShape().rank() != baseShape.length
                    || !crop.targetShape().equals(update.shape())) return false;
            long[] prefix = crop.prefixShape().toLongArray();
            for (int axis = 0; axis < baseShape.length; axis++) {
                if (Math.addExact(prefix[axis], updateShape[axis]) > baseShape[axis]) return false;
            }
            return true;
        }
        return false;
    }

    private static long windowOutput(long numerator, long stride, boolean ceilMode) {
        long quotient = numerator / stride;
        if (ceilMode && numerator % stride != 0) quotient = Math.addExact(quotient, 1);
        return Math.addExact(quotient, 1);
    }

    private static boolean windowFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }

    private static boolean injective(long[] extents, long[] strides) {
        if (java.util.Arrays.stream(extents).anyMatch(extent -> extent == 0)) return true;
        long count = 1;
        for (int axis = 0; axis < extents.length; axis++) {
            if (strides[axis] == 0 && extents[axis] > 1) return false;
            count = Math.multiplyExact(count, extents[axis]);
        }
        if (count > 1_000_000) {
            var axes = new java.util.ArrayList<Integer>();
            for (int axis = 0; axis < extents.length; axis++) if (extents[axis] > 1) axes.add(axis);
            axes.sort(java.util.Comparator.comparingLong(axis -> strides[axis]));
            long covered = 1;
            for (int axis : axes) {
                if (strides[axis] < covered) return false;
                covered = Math.addExact(covered,
                        Math.multiplyExact(extents[axis] - 1, strides[axis]));
            }
            return true;
        }
        var seen = new java.util.HashSet<Long>();
        long[] coordinates = new long[extents.length];
        for (long logical = 0; logical < count; logical++) {
            long address = 0;
            for (int axis = 0; axis < extents.length; axis++) address = Math.addExact(address,
                    Math.multiplyExact(coordinates[axis], strides[axis]));
            if (!seen.add(address)) return false;
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                if (++coordinates[axis] < extents[axis]) break;
                coordinates[axis] = 0;
            }
        }
        return true;
    }

    private static boolean supportsAffine(OperationCapabilityQuery query, TensorDescriptor output) {
        if (query.inputs().size() != 1) return false;
        TensorDescriptor input = query.inputs().getFirst();
        if (input.dataType() != output.dataType()) return false;
        Object kind = query.operation().kind();
        Object attrs = query.operation().attrs();
        LayoutDescriptor inputLayout = input.layout().orElseThrow();
        LayoutDescriptor expected;
        if (kind == ContiguousKind.CONTIGUOUS) {
            return attrs == NoOperationAttrs.INSTANCE && input.shape().equals(output.shape())
                    && output.layout().orElseThrow().equals(LayoutDescriptor.contiguous(output.shape()));
        }
        if (kind instanceof ShapeTransformKind transform) {
            if (!(attrs instanceof TargetShapeAttrs target) || !target.targetShape().isFullyStatic()
                    || !target.targetShape().equals(output.shape())) return false;
            if (transform == ShapeTransformKind.RESHAPE) {
                if (!inputLayout.isContiguous() || input.shape().knownElementCount().orElseThrow()
                        != output.shape().knownElementCount().orElseThrow()) return false;
                LayoutDescriptor canonical = LayoutDescriptor.contiguous(output.shape());
                expected = LayoutDescriptor.of(output.shape(), canonical.strides(),
                        inputLayout.storageOffset(), true);
            } else {
                if (!expands(input.shape(), output.shape())) return false;
                long[] strides = new long[output.shape().rank()];
                int offset = output.shape().rank() - input.shape().rank();
                for (int axis = 0; axis < input.shape().rank(); axis++) {
                    long in = input.shape().toLongArray()[axis];
                    long out = output.shape().toLongArray()[axis + offset];
                    strides[axis + offset] = in == 1 && out != 1 ? 0 : inputLayout.stride(axis);
                }
                expected = LayoutDescriptor.of(output.shape(), strides,
                        inputLayout.storageOffset(), true);
            }
            return output.layout().orElseThrow().equals(expected);
        }
        if (kind instanceof AxisTransformKind transform) {
            long[] inputShape = input.shape().toLongArray();
            if (transform == AxisTransformKind.PERMUTE) {
                if (!(attrs instanceof PermutationAttrs permutation)
                        || permutation.axes().size() != inputShape.length
                        || output.shape().rank() != inputShape.length) return false;
                long[] strides = new long[inputShape.length];
                long[] expectedShape = new long[inputShape.length];
                for (int axis = 0; axis < inputShape.length; axis++) {
                    int source = permutation.axes().get(axis);
                    strides[axis] = inputLayout.stride(source);
                    expectedShape[axis] = inputShape[source];
                }
                if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                expected = LayoutDescriptor.of(output.shape(), strides, inputLayout.storageOffset(), true);
            } else {
                if (!(attrs instanceof AxisTransformAttrs axisAttrs)) return false;
                int axis = axisAttrs.axis();
                if (transform == AxisTransformKind.EXPAND_DIMS) {
                    if (axis > inputShape.length || output.shape().rank() != inputShape.length + 1) return false;
                    long[] expectedShape = new long[inputShape.length + 1];
                    long[] strides = new long[inputShape.length + 1];
                    for (int i = 0; i < expectedShape.length; i++) {
                        if (i == axis) { expectedShape[i] = 1; strides[i] = i == inputShape.length
                                ? 1 : Math.multiplyExact(inputLayout.stride(i), inputShape[i]); }
                        else { int source = i < axis ? i : i - 1;
                            expectedShape[i] = inputShape[source]; strides[i] = inputLayout.stride(source); }
                    }
                    if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                    expected = LayoutDescriptor.of(output.shape(), strides, inputLayout.storageOffset(), true);
                } else {
                    if (axis >= inputShape.length || inputShape[axis] != 1
                            || output.shape().rank() != inputShape.length - 1) return false;
                    long[] expectedShape = new long[inputShape.length - 1];
                    long[] strides = new long[inputShape.length - 1];
                    for (int i = 0, j = 0; i < inputShape.length; i++) if (i != axis) {
                        expectedShape[j] = inputShape[i]; strides[j++] = inputLayout.stride(i);
                    }
                    if (!java.util.Arrays.equals(expectedShape, output.shape().toLongArray())) return false;
                    expected = LayoutDescriptor.of(output.shape(), strides, inputLayout.storageOffset(), true);
                }
            }
            return output.layout().orElseThrow().equals(expected);
        }
        if (kind == SelectKind.SELECT) {
            if (!(attrs instanceof SelectAttrs select) || select.axis() >= input.shape().rank()) return false;
            long[] inShape = input.shape().toLongArray();
            if (select.index() >= inShape[select.axis()] || output.shape().rank() != inShape.length - 1) return false;
            long[] shape = new long[inShape.length - 1], strides = new long[inShape.length - 1];
            for (int i = 0, j = 0; i < inShape.length; i++) if (i != select.axis()) {
                shape[j] = inShape[i]; strides[j++] = inputLayout.stride(i);
            }
            if (!java.util.Arrays.equals(shape, output.shape().toLongArray())) return false;
            expected = LayoutDescriptor.of(output.shape(), strides, Math.addExact(inputLayout.storageOffset(),
                    Math.multiplyExact(select.index(), inputLayout.stride(select.axis()))), true);
            return output.layout().orElseThrow().equals(expected);
        }
        if (kind == SliceKind.SLICE) {
            long[] shape = input.shape().toLongArray();
            long[] strides = inputLayout.strides();
            long offset = inputLayout.storageOffset();
            if (attrs instanceof SliceAttrs slice) {
                for (int i = 0; i < slice.axes().size(); i++) {
                    int axis = slice.axes().get(i); long step = slice.steps().get(i);
                    if (axis >= shape.length || step <= 0) return false;
                    long length = slice.lengths().get(i);
                    if (length > 0 && Math.addExact(slice.starts().get(i),
                            Math.multiplyExact(length - 1, step)) >= shape[axis]) return false;
                    shape[axis] = length;
                    offset = Math.addExact(offset, Math.multiplyExact(slice.starts().get(i), strides[axis]));
                    strides[axis] = Math.multiplyExact(strides[axis], step);
                }
            } else if (attrs instanceof CropToShapeAttrs crop) {
                if (!crop.targetShape().isFullyStatic() || !crop.prefixShape().isFullyStatic()
                        || crop.targetShape().rank() != shape.length
                        || crop.prefixShape().rank() != shape.length) return false;
                long[] target = crop.targetShape().toLongArray();
                long[] prefix = crop.prefixShape().toLongArray();
                for (int axis = 0; axis < shape.length; axis++) {
                    if (Math.addExact(prefix[axis], target[axis]) > shape[axis]) return false;
                    offset = Math.addExact(offset, Math.multiplyExact(prefix[axis], strides[axis]));
                }
                shape = target;
            } else return false;
            if (!java.util.Arrays.equals(shape, output.shape().toLongArray())) return false;
            expected = LayoutDescriptor.of(output.shape(), strides, offset, true);
            return output.layout().orElseThrow().equals(expected);
        }
        return false;
    }

    private static boolean expands(Shape input, Shape output) {
        long[] in = input.toLongArray(), out = output.toLongArray();
        if (in.length > out.length) return false;
        for (int i = 1; i <= in.length; i++) if (in[in.length - i] != 1
                && in[in.length - i] != out[out.length - i]) return false;
        return true;
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
