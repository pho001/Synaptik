package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
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
 * Lowers exactly one fully static, resolved-layout cumulative scan occurrence.
 *
 * <p>The lowerer preserves the Model-owned normalized axis and scan modes, derives a row-major
 * domain of independent non-axis slices, and declares exactly one read input and one distinct
 * injective output. It selects no workspace, materialization, partial scan, carry buffer, or
 * cross-slice combination. Concrete layout and slice geometry remain immutable cold facts.</p>
 */
public final class CpuScanLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless scan lowerer with the current truthful CPU capability provider. */
    public CpuScanLowering() { }

    /**
     * Lowers one supported scan to a two-boundary, zero-workspace slice-domain unit.
     * @param context non-null complete one-node CPU projection whose input and output descriptors
     *     and logical memory requirements remain borrowed for this cold analysis call
     * @return an immutable scan lowering with slice-count iteration extent, exact two-boundary
     *     declarations, and no workspace or virtual value; never {@code null}
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if ownership, occurrence cardinality, capability, value
     *     projection, output distinctness/injectivity, or resolved layout facts are invalid
     * @throws ArithmeticException if exact element, address-span, or slice geometry overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU scan partition requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU scan occurrence");
        ValueId inputId = node.inputs().getFirst(), outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) throw new IllegalArgumentException(
                "scan output must be distinct from its input");
        GraphValue input = require(values, inputId), output = require(values, outputId);
        Layout inputLayout = layout(input), outputLayout = layout(output);
        validateInjective(outputLayout);
        var inputBinding = binding(inputLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        CumulativeScanAttrs attrs = (CumulativeScanAttrs) node.operation().attrs();
        DataType type = input.descriptor().dataType();
        CpuScanIr.Kind kind = node.operation().kind() == CumulativeScanKind.CUM_SUM
                ? CpuScanIr.Kind.CUM_SUM : CpuScanIr.Kind.CUM_PROD;
        long axisExtent = inputLayout.extents[attrs.axis()];
        long total = elementCount(inputLayout.extents);
        long slices = axisExtent == 0 ? 0 : total / axisExtent;
        var ir = new CpuScanIr(kind, type, attrs.axis(), attrs.exclusive(), attrs.reverse(),
                inputBinding.plan(), outputBinding.plan(), CpuScanIr.SEQUENTIAL_TYPED_ROUNDING);
        var geometry = new Geometry(kind, type, attrs.axis(), attrs.exclusive(), attrs.reverse(),
                inputLayout, outputLayout, slices, axisExtent);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding), List.of(
                    input.descriptor().layout().orElseThrow().referencedElementSpan(),
                    output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(type, type), List.of(), new long[] {slices}, slices,
                "legal: one fully static resolved-layout cumulative scan occurrence", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("scan requires non-negative resolved layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(), source.strides());
    }
    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) roles.add(layout.strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST : axis >= layout.extents.length - suffix
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
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
        for (int i = 0; i < layout.extents.length; i++) maximum = Math.addExact(maximum,
                Math.multiplyExact(layout.extents[i] - 1, layout.strides[i]));
        return Math.addExact(layout.offset, Math.addExact(maximum, 1));
    }
    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
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
                        "scan output layout must be injective");
                for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                    if (++coordinates[axis] < layout.extents[axis]) break;
                    coordinates[axis] = 0;
                }
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int i = 0; i < layout.extents.length; i++) if (layout.extents[i] > 1) axes.add(i);
        axes.sort(java.util.Comparator.comparingLong(i -> layout.strides[i])); long covered = 1;
        for (int axis : axes) {
            if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                    "scan output layout must be injective");
            covered = Math.addExact(covered,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
    }

    /**
     * Resolved non-negative cold layout used to derive scan addresses.
     *
     * @param extents non-null fully static logical extents; copied defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-null non-negative element strides in axis order; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots the layout arrays.
         *
         * @throws NullPointerException if an array is {@code null}
         */
        public Layout { extents = extents.clone(); strides = strides.clone(); }
        /**
         * Returns the retained logical extents without exposing mutable record state.
         *
         * @return a new defensive copy of the non-null logical extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns the retained element strides without exposing mutable record state.
         *
         * @return a new defensive copy of the non-null element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Cold scan geometry whose execution domain is the independent slice count.
     *
     * @param kind non-null cumulative arithmetic matching the structural IR
     * @param dataType non-null supported represented numeric type
     * @param axis normalized logical axis in the input rank
     * @param exclusive whether output precedes incorporation of the current value
     * @param reverse whether axis traversal proceeds from the last coordinate downward
     * @param input non-null resolved input layout
     * @param output non-null resolved output layout with the same extents
     * @param sliceCount non-negative product of all non-axis extents, or zero for empty output
     * @param axisExtent non-negative extent of {@code axis}
     */
    public record Geometry(CpuScanIr.Kind kind, DataType dataType, int axis, boolean exclusive,
            boolean reverse, Layout input, Layout output, long sliceCount, long axisExtent) {
        /**
         * Validates one immutable geometry.
         *
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if type, axis, Shape, slice count, or axis extent
         *     disagrees with the scan contract
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            if (dataType == DataType.BOOL || axis < 0 || axis >= input.extents.length
                    || !Arrays.equals(input.extents, output.extents)
                    || sliceCount < 0 || axisExtent != input.extents[axis])
                throw new IllegalArgumentException("scan geometry facts disagree");
        }
        /**
         * Packs direct carrier bases and layout geometry for one bound invocation.
         *
         * <p>The returned mutable array is invocation-owned coordinate state. Callers must not
         * share it between concurrent ranges.</p>
         *
         * @param bases non-null two-element array containing input then output element bases;
         *     read but not retained or mutated
         * @return a new mutable packed geometry array owned by the caller; never {@code null}
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if {@code bases} does not contain exactly two entries
         * @throws ArithmeticException if adding a carrier base and layout offset overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != 2) throw new IllegalArgumentException("scan requires two carrier bases");
            int rank = input.extents.length;
            long[] packed = new long[8 + rank + 2 * (2 + 2 * rank)];
            packed[0] = kind.ordinal(); packed[1] = dataType.ordinal(); packed[2] = rank;
            packed[3] = axis; packed[4] = exclusive ? 1 : 0; packed[5] = reverse ? 1 : 0;
            packed[6] = sliceCount; packed[7] = axisExtent;
            int x = 8 + rank;
            x = packLayout(packed, x, input, bases[0]);
            packLayout(packed, x, output, bases[1]);
            return packed;
        }
        private static int packLayout(long[] target, int x, Layout layout, long base) {
            target[x++] = layout.extents.length; target[x++] = Math.addExact(base, layout.offset);
            for (long extent : layout.extents) target[x++] = extent;
            for (long stride : layout.strides) target[x++] = stride;
            return x;
        }
    }
}
