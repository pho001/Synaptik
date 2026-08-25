package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers exactly one fully static, resolved-layout one-axis arg-extrema occurrence.
 *
 * <p>The lowerer consumes the already-normalized Model axis unchanged, validates it against the
 * resolved input rank, and derives a domain of complete independent output cells. It declares one
 * numeric read boundary and one distinct INT64 write boundary, with no workspace,
 * materialization, partial state, or combine state.</p>
 */
public final class CpuArgExtremaLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless lowerer using the current truthful capability provider. */
    public CpuArgExtremaLowering() { }

    /**
     * Lowers one supported occurrence to immutable logical-index geometry.
     *
     * @param context non-null complete one-node CPU analysis projection
     * @return one two-boundary, zero-resource lowering whose range is output-cell ordinals
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence, axis, shape, layout, or boundary facts
     *     do not match the exact supported matrix
     * @throws ArithmeticException if checked geometry or span arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) {
            throw new IllegalArgumentException("CPU arg-extrema partition requires exactly one node");
        }
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) {
            throw new IllegalArgumentException(
                    "partition contains an unsupported CPU arg-extrema occurrence");
        }
        ValueId inputId = node.inputs().getFirst();
        ValueId outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) {
            throw new IllegalArgumentException("arg-extrema output must be distinct from its input");
        }
        GraphValue input = require(values, inputId);
        GraphValue output = require(values, outputId);
        Layout inputLayout = layout(input);
        Layout outputLayout = layout(output);
        validateInjective(outputLayout);
        ArgExtremaAttrs attrs = (ArgExtremaAttrs) node.operation().attrs();
        int axis = attrs.axis();
        if (axis < 0 || axis >= inputLayout.extents.length) {
            throw new IllegalArgumentException("arg-extrema axis is outside the resolved input rank");
        }
        long axisExtent = inputLayout.extents[axis];
        if (axisExtent == 0) {
            throw new IllegalArgumentException("arg-extrema selected axis must be non-empty");
        }
        long[] expected = resultShape(inputLayout.extents, axis, attrs.keepDimensions());
        if (!Arrays.equals(expected, outputLayout.extents)) {
            throw new IllegalArgumentException("arg-extrema output Shape disagrees with its axis form");
        }
        var inputBinding = binding(inputLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        DataType inputType = input.descriptor().dataType();
        CpuArgExtremaIr.Kind kind = node.operation().kind() == AggregateReductionKind.ARG_MIN
                ? CpuArgExtremaIr.Kind.ARG_MIN : CpuArgExtremaIr.Kind.ARG_MAX;
        long outputCount = elementCount(outputLayout.extents);
        var ir = new CpuArgExtremaIr(kind, inputType, axis, attrs.keepDimensions(),
                attrs.tiePolicy(), axisExtent <= Integer.MAX_VALUE,
                outputCount <= Integer.MAX_VALUE,
                inputBinding.plan(), outputBinding.plan());
        var geometry = new Geometry(kind, inputType, axis, attrs.keepDimensions(),
                attrs.tiePolicy(), inputLayout, outputLayout, outputCount, axisExtent);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding), List.of(
                        input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(inputType, DataType.INT64), List.of(), new long[] {outputCount},
                outputCount, "legal: one static resolved-layout one-axis arg-extrema occurrence",
                new long[0], Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(geometry), Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException(
                "partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0)) {
            throw new IllegalArgumentException("arg-extrema requires non-negative resolved layouts");
        }
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(),
                source.strides());
    }

    private static long[] resultShape(long[] input, int axis, boolean keep) {
        long[] result = new long[keep ? input.length : input.length - 1];
        for (int source = 0, target = 0; source < input.length; source++) {
            if (source == axis) {
                if (keep) result[target++] = 1;
            } else result[target++] = input[source];
        }
        return result;
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0;
        long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) {
            roles.add(layout.strides[axis] == 0 ? CpuAccessPlan.AxisRole.BROADCAST
                    : axis >= layout.extents.length - suffix
                            ? CpuAccessPlan.AxisRole.CONTIGUOUS
                            : CpuAccessPlan.AxisRole.STRIDED);
        }
        var plan = new CpuAccessPlan(kind, suffix == layout.extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                layout.extents.length, roles, suffix);
        long count = elementCount(layout.extents);
        return CpuAccessPlan.Binding.create(plan, layout.extents, layout.offset, layout.strides,
                count, 0, count, referencedSpan(layout));
    }

    private static long referencedSpan(Layout layout) {
        if (elementCount(layout.extents) == 0) return 0;
        long maximum = 0;
        for (int axis = 0; axis < layout.extents.length; axis++) {
            maximum = Math.addExact(maximum,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
        return Math.addExact(layout.offset, Math.addExact(maximum, 1));
    }

    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents);
        if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>();
            long[] coordinates = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = 0;
                for (int axis = 0; axis < layout.extents.length; axis++) {
                    address = Math.addExact(address,
                            Math.multiplyExact(coordinates[axis], layout.strides[axis]));
                }
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "arg-extrema output layout must be injective");
                for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                    if (++coordinates[axis] < layout.extents[axis]) break;
                    coordinates[axis] = 0;
                }
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < layout.extents.length; axis++) {
            if (layout.extents[axis] > 1) axes.add(axis);
        }
        axes.sort(java.util.Comparator.comparingLong(axis -> layout.strides[axis]));
        long covered = 1;
        for (int axis : axes) {
            if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                    "arg-extrema output layout must be injective");
            covered = Math.addExact(covered,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
    }

    /**
     * Immutable resolved non-negative layout geometry.
     *
     * @param extents non-null logical extents; copied defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-null non-negative element strides; copied defensively and required to
     *     have one entry per extent
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one resolved non-negative layout.
         *
         * @param extents non-null logical extents; copied defensively
         * @param offset non-negative carrier-relative element offset
         * @param strides non-null non-negative element strides; copied defensively and required
         *     to have one entry per extent
         * @throws NullPointerException if either array is {@code null}
         * @throws IllegalArgumentException if {@code offset} is negative or the array lengths
         *     differ
         */
        public Layout {
            extents = Objects.requireNonNull(extents, "extents").clone();
            strides = Objects.requireNonNull(strides, "strides").clone();
            if (offset < 0 || extents.length != strides.length) {
                throw new IllegalArgumentException("arg-extrema layout facts disagree");
            }
        }
        /**
         * Returns the logical extents without exposing the stored array.
         *
         * @return a defensive copy of logical extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns the element strides without exposing the stored array.
         *
         * @return a defensive copy of element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Immutable cold geometry for complete output-cell ranges.
     *
     * @param kind non-null arg-min or arg-max selection
     * @param inputType non-null supported numeric input type
     * @param axis normalized input axis
     * @param keepDimensions exact output Shape form
     * @param tiePolicy non-null logical-coordinate tie policy
     * @param input non-null resolved input layout
     * @param output non-null resolved INT64 output layout
     * @param outputCount non-negative number of independent output cells
     * @param axisExtent positive selected-axis extent
     */
    public record Geometry(CpuArgExtremaIr.Kind kind, DataType inputType, int axis,
            boolean keepDimensions, ArgExtremaTiePolicy tiePolicy, Layout input, Layout output,
            long outputCount, long axisExtent) {
        /**
         * Validates one immutable complete-output-cell geometry.
         *
         * @param kind non-null arg-min or arg-max selection
         * @param inputType non-null supported numeric input type
         * @param axis normalized non-negative input axis within {@code input}
         * @param keepDimensions exact output Shape form
         * @param tiePolicy non-null logical-coordinate tie policy
         * @param input non-null resolved input layout
         * @param output non-null resolved INT64 output layout
         * @param outputCount exact non-negative output-cell count
         * @param axisExtent exact positive selected-axis extent
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if the type, axis, extent, count, or output Shape
         *     disagrees with the resolved layouts
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(inputType, "inputType");
            Objects.requireNonNull(tiePolicy, "tiePolicy");
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            if (inputType == DataType.BOOL || axis < 0 || axis >= input.extents.length
                    || axisExtent <= 0 || axisExtent != input.extents[axis]
                    || outputCount != elementCount(output.extents)
                    || !Arrays.equals(resultShape(input.extents, axis, keepDimensions),
                            output.extents)) {
                throw new IllegalArgumentException("arg-extrema geometry facts disagree");
            }
        }

        /**
         * Packs carrier-relative bases and immutable mapping geometry for one invocation.
         *
         * @param bases two element-base offsets in input/output boundary order
         * @return a new packed primitive geometry array, never retained by this recipe
         * @throws IllegalArgumentException if {@code bases} does not contain exactly two entries
         */
        public long[] pack(long[] bases) {
            Objects.requireNonNull(bases, "bases");
            if (bases.length != 2) throw new IllegalArgumentException(
                    "arg-extrema requires two carrier bases");
            int inputRank = input.extents.length;
            int outputRank = output.extents.length;
            long[] packed = new long[8 + 2 + 2 * inputRank + 2 + 2 * outputRank];
            packed[0] = kind.ordinal();
            packed[1] = inputType.ordinal();
            packed[2] = inputRank;
            packed[3] = outputRank;
            packed[4] = axis;
            packed[5] = keepDimensions ? 1 : 0;
            packed[6] = tiePolicy.ordinal();
            packed[7] = axisExtent;
            int next = packLayout(packed, 8, input, bases[0]);
            packLayout(packed, next, output, bases[1]);
            return packed;
        }

        private static int packLayout(long[] target, int next, Layout layout, long base) {
            target[next++] = layout.extents.length;
            target[next++] = Math.addExact(base, layout.offset);
            for (long extent : layout.extents) target[next++] = extent;
            for (long stride : layout.strides) target[next++] = stride;
            return next;
        }
    }
}
