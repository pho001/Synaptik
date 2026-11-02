package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.StaticDimension;
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
 * Lowers one static resolved-layout or canonical single-named-B stable ordering occurrence.
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

    /**
     * Proves ordering shape/type/attribute relationships without substituting a batch size.
     * The selected axis and all slice axes retain their original logical positions.
     * @param context non-null one-node SORT, ARGSORT or TOP_K projection
     * @return ordinary structural IR with an unresolved numeric geometry recipe
     * @throws IllegalArgumentException if the original descriptors or layouts are inconsistent
     * @throws ArithmeticException if fixed geometry coefficients overflow
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Uses the complete lexical body's numeric physical layouts, preserving incoming views.
     * @param context non-null original one-node ordering projection
     * @param layouts non-null complete local boundary-layout map
     * @return unresolved ordinary ordering recipe
     * @throws IllegalArgumentException if ordinary descriptors or numeric layouts disagree
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1) throw invalidBatch();
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 1) throw invalidBatch();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(v -> values.put(v.id(), v));
        var input = require(values, node.inputs().getFirst()).descriptor();
        var shape = input.shape().dimensions();
        int axis;
        long k;
        boolean descending, sorted;
        CpuOrderingIr.Family family;
        if (node.operation().kind() instanceof OrderingKind kind
                && node.operation().attrs() instanceof SortAttrs attrs) {
            axis = attrs.axis();
            family = kind == OrderingKind.SORT ? CpuOrderingIr.Family.SORT : CpuOrderingIr.Family.ARGSORT;
            k = shape.get(axis) instanceof StaticDimension extent ? extent.size() : -1;
            descending = attrs.descending(); sorted = true;
        } else if (node.operation().kind() == TopKKind.TOP_K
                && node.operation().attrs() instanceof TopKAttrs attrs) {
            family = CpuOrderingIr.Family.TOP_K;
            axis = attrs.axis(); k = attrs.k();
            descending = attrs.largest(); sorted = attrs.sorted();
        } else throw invalidBatch();
        var ids = new ArrayList<>(node.inputs());
        ids.addAll(node.outputs());
        if (new HashSet<>(ids).size() != ids.size()) throw invalidBatch();
        var types = new ArrayList<DataType>();
        var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int b = 0; b < ids.size(); b++) {
            var access = layouts.get(ids.get(b));
            if (access == null) throw invalidBatch();
            var plan = access.plan();
            var permission = new CpuAccessPlan(b == 0 ? CpuAccessPlan.AccessKind.READ
                    : CpuAccessPlan.AccessKind.WRITE, plan.regime(), plan.iterationRank(),
                    plan.axisRoles(), plan.contiguousSuffix());
            accesses.add(new CpuAccessPlan.BatchBinding(permission, access.extentConstants(),
                    access.extentBatchFactors(), access.baseElementOffset(),
                    access.strideConstants(), access.strideBatchFactors()));
            types.add(require(values, ids.get(b)).descriptor().dataType());
        }
        var ir = new CpuOrderingIr(family, input.dataType(), descending, sorted,
                accesses.stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                CpuOrderingIr.TWO_INDEX_MERGE_REGIONS);
        return new CpuPartitionLowering.LoweredPartition(
                new Geometry.BatchGeometry(ir, ids, types, accesses, axis, k));
    }

    private static IllegalArgumentException invalidBatch() {
        return new IllegalArgumentException("inconsistent symbolic ordering geometry");
    }

    private static BatchByteSize batchSlices(CpuAccessPlan.BatchBinding input, int selected) {
        long constant = 1, factor = 0;
        for (int axis = 0; axis < input.extentConstants().size(); axis++) {
            if (axis == selected) continue;
            long c = input.extentConstants().get(axis), f = input.extentBatchFactors().get(axis);
            if (factor != 0 && f != 0) throw invalidBatch();
            factor = Math.addExact(Math.multiplyExact(factor, c), Math.multiplyExact(constant, f));
            constant = Math.multiplyExact(constant, c);
        }
        return new BatchByteSize(constant, factor, 1);
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
            if (!CpuOutputInjectivity.isInjective(extents, strides)) {
                throw new IllegalArgumentException("output layout injectivity is unproved");
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
         * Numeric ordering recipe; no source descriptors or actual batch enter generated identity.
         * A negative selected count denotes the complete symbolic sort axis, not a sampled k.
         * @param ir immutable prepared ordinary ordering IR
         * @param boundaries input followed by ordered output identities
         * @param boundaryTypes actual represented input/value/index types
         * @param bindings immutable affine boundary geometry
         * @param axis original selected logical axis
         * @param k fixed TOP_K count, fixed sort extent, or -1 for a symbolic sort extent
         */
        public record BatchGeometry(CpuOrderingIr ir, List<ValueId> boundaries,
                List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
                int axis, long k) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots the numeric preparation facts.
             * @param ir non-null structural IR
             * @param boundaries non-null ordered identities
             * @param boundaryTypes non-null represented types
             * @param bindings non-null access recipes
             * @param axis normalized selected axis
             * @param k non-negative fixed count or -1 for the symbolic complete axis
             */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries);
                boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
            }
            /**
             * @return affine independent-slice capacity; actual empty selected axes bind zero work
             */
            @Override public BatchByteSize iterationCount() {
                var input = bindings.getFirst();
                return k == 0 || input.extentConstants().get(axis) == 0
                        && input.extentBatchFactors().get(axis) == 0
                        ? new BatchByteSize(0, 0, 1) : batchSlices(input, axis);
            }
            /** @return checked capacity for two INT64 index regions of one complete selected axis */
            @Override public BatchByteSize scratchBytes() {
                var input = bindings.getFirst();
                var work = iterationCount();
                if (work.constantBytes() == 0 && work.bytesPerBatch() == 0)
                    return new BatchByteSize(0, 0, 1);
                return new BatchByteSize(Math.multiplyExact(16, input.extentConstants().get(axis)),
                        Math.multiplyExact(16, input.extentBatchFactors().get(axis)), 1);
            }
            /** @return no mutable cursor cells; ordinary ordering uses only local cursors */
            @Override public List<PackReset> resets() { return List.of(); }
            /** @return deterministic B-independent numeric recipe identity without value IDs */
            @Override public String identity() { return "ordering:" + bindings + ":" + axis + ":" + k; }
            /**
             * Resolves numeric axis/slice/scratch facts and keeps the identical prepared IR.
             * @param batch non-negative actual batch extent
             * @return concrete ordinary ordering geometry
             * @throws IllegalArgumentException if TOP_K exceeds the actual selected axis
             * @throws ArithmeticException if checked geometry arithmetic exceeds CPU limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var access = bindings.stream().map(b -> b.bind(batch)).toList();
                var layouts = access.stream().map(b -> new Layout(
                        b.extents().stream().mapToLong(Long::longValue).toArray(),
                        b.baseElementOffset(),
                        b.effectiveStrides().stream().mapToLong(Long::longValue).toArray())).toList();
                long extent = layouts.getFirst().extents[axis];
                long selected = k < 0 ? extent : k;
                if (selected > extent) throw invalidBatch();
                long slices = extent == 0 ? 0 : access.getFirst().elementCount() / extent;
                long work = selected == 0 ? 0 : slices;
                long scratch = work == 0 ? 0 : Math.multiplyExact(16, extent);
                var geometry = new Geometry(ir.family(), boundaryTypes.getFirst(), axis, selected,
                        ir.descending(), ir.sorted(), layouts, scratch, slices);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, access,
                        access.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {work}, work,
                        "legal: bound symbolic ordering", new long[0], Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
            }
        }

        /**
         * One fixed generated assignment of a boundary's element base plus a layout delta.
         * @param cell non-negative destination cell in this family's packed geometry
         * @param boundary boundary position in the pack's carrier-base order
         * @param elementDelta immutable element offset added with checked arithmetic
         */
        public record BaseCell(int cell, int boundary, long elementDelta) { }

        /**
         * One original cursor seed restored before every temporal helper call, including retry.
         * @param cell non-negative destination cell in this family's packed geometry
         * @param value original range-start cursor value
         */
        public record ResetCell(int cell, long value) { }

        /**
         * Cold, immutable facts for emitting fixed numeric assignments, never a runtime program.
         * Scratch range offsets and invariant geometry remain bound once per invocation.
         * @param bases non-null boundary-base assignments, defensively snapshotted
         * @param resets non-null mutable-cursor assignments, defensively snapshotted
         */
        public record TemporalRebind(List<BaseCell> bases, List<ResetCell> resets) {
            /**
             * Snapshots the generation-time assignments; retains no invocation storage.
             * @param bases non-null base facts without null entries
             * @param resets non-null reset facts without null entries
             * @throws NullPointerException if either list or an entry is null
             */
            public TemporalRebind {
                bases = List.copyOf(bases);
                resets = List.copyOf(resets);
            }
        }

        /**
         * Describes temporal base rebinding and restoration of this family's original cursors.
         * Construct these facts only during generation or cold numeric binding; generated code
         * emits the assignments directly, without calling pack, allocating, or iterating facts.
         * Each boundary base is normalized using that boundary's own element width beforehand.
         * @param start non-negative inclusive ordinary helper range start
         * @return immutable owner-defined base locations and original cursor seeds
         * @throws IllegalArgumentException if start is negative
         * @throws ArithmeticException if checked layout or cursor arithmetic overflows
         */
        public TemporalRebind temporalRebind(long start) {
            if (start < 0) throw new IllegalArgumentException("negative temporal range start");
            var bases = new ArrayList<BaseCell>(boundaries.size());
            int cell = 11;
            for (int i = 0; i < boundaries.size(); i++) {
                var layout = boundaries.get(i);
                bases.add(new BaseCell(cell + 1, i, layout.offset));
                cell += 2 + 2 * layout.extents.length;
            }
            return new TemporalRebind(bases, List.of());
        }

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
