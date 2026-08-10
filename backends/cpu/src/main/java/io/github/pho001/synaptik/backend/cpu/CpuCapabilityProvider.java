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
 * bounded to one through sixteen occurrences.</p>
 *
 * <p>Complete-partition lowering remains stricter: it validates either a connected one-to-eight
 * pointwise chain or a connected one-to-eight affine chain, then applies exact layout, alias,
 * fan-out, publication, and partition-boundary checks before resource declaration. Occurrence
 * support therefore does not promise that an arbitrary mixed or branched partition can be
 * prepared.</p>
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
     * operations additionally require exact static PAD/TILE/composition shape relationships,
     * an injective result layout, and at most sixteen composition occurrences. Cross-type casts,
     * dynamic or unresolved geometry, negative-step slices, non-injective movement outputs, and
     * all rows outside the implemented matrix return {@code false} without defining conversion
     * or fallback behavior.
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

    private static boolean staticResolved(TensorDescriptor descriptor) {
        return descriptor.shape().isFullyStatic() && descriptor.layout().isPresent();
    }

    private static boolean affineKind(Object kind) {
        return kind == ContiguousKind.CONTIGUOUS || kind instanceof ShapeTransformKind
                || kind instanceof AxisTransformKind || kind == SelectKind.SELECT
                || kind == SliceKind.SLICE;
    }

    private static boolean movementKind(Object kind) {
        return kind == PadKind.PAD || kind == TileKind.TILE
                || kind == TensorCompositionKind.CONCAT
                || kind == TensorCompositionKind.STACK;
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
