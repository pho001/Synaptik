package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one fully static, resolved-layout stable SORT, ARGSORT, or TOP_K occurrence.
 *
 * <p>Lowering revalidates the exact Model descriptor relationship, retains logical-axis geometry,
 * derives ordered input/value/index boundaries, proves each output layout injective, and sizes
 * two primitive INT64 index regions for every selected execution range. It does not compare
 * values, allocate scratch, assign slots, select carriers, or submit worker work.</p>
 */
public final class CpuOrderingLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless ordering lowerer with the truthful CPU capability boundary. */
    public CpuOrderingLowering() { }

    /**
     * Lowers one exact supported ordering occurrence and its complete slice geometry.
     *
     * @param context non-null CPU analysis projection containing exactly one supported node
     * @return one immutable lowered partition whose execution domain is complete independent
     *     logical-axis slices and whose optional ordering geometry is present; never {@code null}
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if node count, projected values, attributes, descriptors,
     *     boundary identity, layouts, output injectivity, or CPU capability is invalid
     * @throws ArithmeticException if checked element, address-span, or scratch sizing overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU ordering partition requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU ordering occurrence");
        GraphValue input = require(values, node.inputs().getFirst());
        var boundaryIds = new ArrayList<ValueId>();
        boundaryIds.add(node.inputs().getFirst());
        boundaryIds.addAll(node.outputs());
        if (new HashSet<>(boundaryIds).size() != boundaryIds.size()) throw new IllegalArgumentException(
                "ordering boundaries must be distinct");
        var layouts = new ArrayList<Layout>();
        layouts.add(layout(input));
        for (ValueId output : node.outputs()) {
            Layout candidate = layout(require(values, output));
            validateInjective(candidate.extents, candidate.strides);
            layouts.add(candidate);
        }
        CpuOrderingIr.Family family;
        int axis;
        long k;
        boolean descending;
        boolean sorted;
        if (node.operation().kind() instanceof OrderingKind kind) {
            SortAttrs attrs = (SortAttrs) node.operation().attrs();
            family = kind == OrderingKind.SORT ? CpuOrderingIr.Family.SORT
                    : CpuOrderingIr.Family.ARGSORT;
            axis = attrs.axis();
            k = layouts.getFirst().extents[axis];
            descending = attrs.descending();
            sorted = true;
        } else {
            TopKAttrs attrs = (TopKAttrs) node.operation().attrs();
            family = CpuOrderingIr.Family.TOP_K;
            axis = attrs.axis(); k = attrs.k(); descending = attrs.largest(); sorted = attrs.sorted();
        }
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        bindings.add(binding(layouts.getFirst(), CpuAccessPlan.AccessKind.READ));
        for (int i = 1; i < layouts.size(); i++) bindings.add(
                binding(layouts.get(i), CpuAccessPlan.AccessKind.WRITE));
        var ir = new CpuOrderingIr(family, input.descriptor().dataType(), descending, sorted,
                bindings.stream().map(CpuAccessPlan.Binding::plan).toList(),
                CpuOrderingIr.TWO_INDEX_MERGE_REGIONS);
        long axisExtent = layouts.getFirst().extents[axis];
        long inputCount = elementCount(layouts.getFirst().extents);
        long slices = axisExtent == 0 ? 0 : inputCount / axisExtent;
        long work = k == 0 ? 0 : slices;
        long scratchSliceBytes = work == 0 ? 0
                : Math.multiplyExact(Math.multiplyExact(axisExtent, 2), Long.BYTES);
        var geometry = new Geometry(family, input.descriptor().dataType(), axis, k, descending,
                sorted, layouts, scratchSliceBytes, slices);
        var spans = boundaryIds.stream().map(id -> require(values, id).descriptor().layout()
                .orElseThrow().referencedElementSpan()).toList();
        var types = boundaryIds.stream().map(id -> require(values, id).descriptor().dataType()).toList();
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), new long[] {work}, work,
                "legal: one fully static stable ordering occurrence", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(geometry));
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("ordering requires non-negative resolved layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(),
                source.strides());
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
        if (Arrays.stream(extents).anyMatch(v -> v == 0)) return 0;
        long count = 1; for (long extent : extents) count = Math.multiplyExact(count, extent);
        return count;
    }

    private static void validateInjective(long[] extents, long[] strides) {
        long count = elementCount(extents); if (count == 0) return;
        var axes = new ArrayList<Integer>();
        for (int i = 0; i < extents.length; i++) if (extents[i] > 1) axes.add(i);
        axes.sort(Comparator.comparingLong(i -> strides[i])); long covered = 1;
        boolean simple = true;
        for (int axis : axes) {
            if (strides[axis] < covered) { simple = false; break; }
            covered = Math.addExact(covered, Math.multiplyExact(extents[axis] - 1, strides[axis]));
        }
        if (simple) return;
        if (count > 1_000_000) throw new IllegalArgumentException("ordering output is not injective");
        var seen = new HashSet<Long>(); long[] coordinate = new long[extents.length];
        for (long ordinal = 0; ordinal < count; ordinal++) {
            long address = 0;
            for (int i = 0; i < coordinate.length; i++) address = Math.addExact(address,
                    Math.multiplyExact(coordinate[i], strides[i]));
            if (!seen.add(address)) throw new IllegalArgumentException("ordering output is not injective");
            for (int i = coordinate.length - 1; i >= 0; i--) {
                if (++coordinate[i] < extents[i]) break; coordinate[i] = 0;
            }
        }
    }

    /**
     * Immutable resolved layout used only by cold invocation geometry.
     *
     * @param extents non-null fully static logical extents; copied defensively
     * @param offset non-negative base element offset
     * @param strides non-null non-negative element strides in rank order; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one resolved layout.
         *
         * @param extents non-null logical extents
         * @param offset non-negative base element offset established by lowering
         * @param strides non-null element strides matching the extent rank
         * @throws NullPointerException if either array is null
         */
        public Layout { extents = extents.clone(); strides = strides.clone(); }

        /**
         * Returns the retained logical extents without exposing mutable state.
         *
         * @return a defensive copy of the logical extents
         */
        @Override public long[] extents() { return extents.clone(); }

        /**
         * Returns the retained element strides without exposing mutable state.
         *
         * @return a defensive copy of the element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Immutable ordering geometry and exact per-range scratch policy.
     *
     * @param family non-null ordering family
     * @param dataType non-null represented input/value type
     * @param axis normalized logical axis
     * @param k selected pair count for TOP_K or the complete axis extent otherwise
     * @param descending exact stable value direction; floating NaNs remain last
     * @param sorted exact TOP_K output-order flag
     * @param boundaries ordered input then value/index output layouts; copied defensively
     * @param scratchSliceBytes exact bytes for two axis-extent INT64 index regions per active range
     * @param sliceCount number of independent logical-axis slices
     */
    public record Geometry(CpuOrderingIr.Family family, DataType dataType, int axis, long k,
            boolean descending, boolean sorted, List<Layout> boundaries, long scratchSliceBytes,
            long sliceCount) {
        /**
         * Validates and snapshots complete cold ordering geometry.
         *
         * @param family non-null ordering family
         * @param dataType non-null represented type
         * @param axis normalized logical axis within the input rank
         * @param k non-negative selected count
         * @param descending exact requested direction
         * @param sorted exact requested output-order flag
         * @param boundaries ordered non-null input/output layouts
         * @param scratchSliceBytes non-negative exact per-range scratch size in bytes
         * @param sliceCount non-negative independent-slice count
         * @throws NullPointerException if a required reference or boundary is null
         * @throws IllegalArgumentException if axis, count, boundary cardinality, or scratch facts
         *     disagree with the family
         */
        public Geometry {
            Objects.requireNonNull(family, "family"); Objects.requireNonNull(dataType, "dataType");
            boundaries = List.copyOf(boundaries);
            if (axis < 0 || boundaries.isEmpty() || axis >= boundaries.getFirst().extents.length
                    || k < 0 || scratchSliceBytes < 0 || sliceCount < 0
                    || boundaries.size() != (family == CpuOrderingIr.Family.TOP_K ? 3 : 2))
                throw new IllegalArgumentException("ordering geometry facts disagree");
        }
        /**
         * Computes the run-owned workspace required for all selected ranges.
         *
         * @param ranges selected range count; preparation supplies a positive value
         * @return exact workspace bytes, with disjoint equal-sized regions for every range
         * @throws ArithmeticException if the product overflows {@code long}
         */
        public long workspaceBytes(int ranges) {
            return Math.multiplyExact(scratchSliceBytes, ranges);
        }

        /**
         * Packs one invocation range's immutable geometry and workspace-region identity.
         *
         * @param bases carrier-relative base element positions in boundary order
         * @param start inclusive independent-slice ordinal
         * @param end exclusive independent-slice ordinal
         * @param rangeIndex non-negative selected-range index used to choose disjoint scratch
         * @return a fresh primitive geometry array retained by one bound invocation
         * @throws NullPointerException if {@code bases} is null
         * @throws IllegalArgumentException if boundary count, range, or range index is invalid
         * @throws ArithmeticException if checked offset packing overflows
         */
        public long[] pack(long[] bases, long start, long end, int rangeIndex) {
            if (bases.length != boundaries.size() || start < 0 || end < start || end > sliceCount
                    || rangeIndex < 0) throw new IllegalArgumentException("ordering packed facts disagree");
            int size = 11;
            for (Layout layout : boundaries) size += 2 + 2 * layout.extents.length;
            long[] p = new long[size]; int x = 0;
            p[x++] = family.ordinal(); p[x++] = dataType.ordinal(); p[x++] = axis; p[x++] = k;
            p[x++] = descending ? 1 : 0; p[x++] = sorted ? 1 : 0;
            p[x++] = boundaries.size(); p[x++] = scratchSliceBytes; p[x++] = rangeIndex;
            p[x++] = start; p[x++] = end;
            for (int i = 0; i < boundaries.size(); i++) {
                Layout layout = boundaries.get(i); p[x++] = layout.extents.length;
                p[x++] = Math.addExact(bases[i], layout.offset);
                for (long extent : layout.extents) p[x++] = extent;
                for (long stride : layout.strides) p[x++] = stride;
            }
            return p;
        }
    }
}
