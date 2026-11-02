package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
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
 * Lowers one resolved-layout static or canonical single-named-B gather or one-hot occurrence.
 * Lowering preserves Model-owned coordinate meaning while deriving unique CPU boundary order,
 * structural generated identity, and compact immutable geometry. It creates no carrier, slot,
 * workspace, per-index table, or per-output address table.
 */
public final class CpuIndexingLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless indexing lowerer. */
    public CpuIndexingLowering() { }

    /**
     * Lowers exactly one supported indexing occurrence.
     *
     * @param context non-null complete CPU partition projection
     * @return immutable indexing lowering with compact geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the projection is not exactly one supported indexing
     *     occurrence, contains a repeated output/input identity, or has non-injective output
     *     geometry
     * @throws ArithmeticException if exact count, span, or compact geometry arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU indexing partition requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU indexing occurrence");

        var unique = new LinkedHashMap<ValueId, Integer>();
        var occurrenceMap = new ArrayList<Integer>();
        for (ValueId id : node.inputs()) occurrenceMap.add(
                unique.computeIfAbsent(id, ignored -> unique.size()));
        ValueId outputId = node.outputs().getFirst();
        if (unique.containsKey(outputId)) throw new IllegalArgumentException(
                "indexing output must be distinct from every input");
        var boundaryIds = new ArrayList<>(unique.keySet());
        boundaryIds.add(outputId);
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var types = new ArrayList<DataType>();
        var layouts = new ArrayList<Geometry.Layout>();
        for (int i = 0; i < boundaryIds.size(); i++) {
            GraphValue value = require(values, boundaryIds.get(i));
            long[] extents = value.descriptor().shape().toLongArray();
            LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
            if (i + 1 == boundaryIds.size()) validateInjective(extents, layout.strides());
            bindings.add(binding(extents, layout, i + 1 == boundaryIds.size()
                    ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
            spans.add(layout.referencedElementSpan());
            types.add(value.descriptor().dataType());
            layouts.add(new Geometry.Layout(extents, layout.storageOffset(), layout.strides()));
        }

        Object kind = node.operation().kind();
        CpuIndexingIr.Family family;
        Geometry.Variant variant;
        if (kind instanceof AxisGatherKind axisKind) {
            int axis = ((IndexAxisAttrs) node.operation().attrs()).axis();
            family = axisKind == AxisGatherKind.GATHER ? CpuIndexingIr.Family.GATHER
                    : CpuIndexingIr.Family.GATHER_ELEMENTS;
            variant = new Geometry.Axis(axis);
        } else if (kind == GatherNdKind.GATHER_ND) {
            int batch = ((GatherNdAttrs) node.operation().attrs()).batchDimensions();
            long[] indexShape = layouts.get(occurrenceMap.get(1)).extents();
            variant = new Geometry.Nd(batch, Math.toIntExact(indexShape[indexShape.length - 1]));
            family = CpuIndexingIr.Family.GATHER_ND;
        } else if (kind == OneHotKind.ONE_HOT) {
            variant = new Geometry.Hot(((OneHotAttrs) node.operation().attrs()).depth());
            family = CpuIndexingIr.Family.ONE_HOT;
        } else throw new IllegalArgumentException("unsupported indexing family");
        var plans = bindings.stream().map(CpuAccessPlan.Binding::plan).toList();
        var ir = new CpuIndexingIr(family, occurrenceMap, types, plans);
        Geometry geometry = new Geometry(family, occurrenceMap, layouts, types, variant);
        long[] outputExtents = layouts.getLast().extents();
        long outputCount = elementCount(outputExtents);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), outputExtents, outputCount,
                "legal: one fully static resolved-layout indexing occurrence", new long[0],
                Optional.empty(), Optional.of(geometry));
    }

    /**
     * Proves the original symbolic descriptors and retains only numeric boundary recipes.
     * No representative batch, graph reconstruction, or invocation-time semantic lowering occurs.
     * Index bounds remain the ordinary validator's responsibility, including empty outputs.
     * @param context non-null one-node indexing projection with one common named batch symbol
     * @return unresolved ordinary IR and family-owned numeric geometry
     * @throws IllegalArgumentException if descriptors, attributes, identities or layouts disagree
     * @throws ArithmeticException if fixed geometry coefficients overflow
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers against the complete lexical body's already-proved physical view recipes.
     * @param context non-null original one-node occurrence projection
     * @param layouts non-null complete local value-to-layout map, including incoming views
     * @return ordinary IR and unresolved numeric indexing geometry
     * @throws IllegalArgumentException if ordinary descriptors or numeric layouts disagree
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1) throw invalidBatch();
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(v -> values.put(v.id(), v));
        Object kind = node.operation().kind();
        CpuIndexingIr.Family family;
        Geometry.Variant variant;
        if (kind == OneHotKind.ONE_HOT && node.operation().attrs() instanceof OneHotAttrs hot) {
            family = CpuIndexingIr.Family.ONE_HOT;
            variant = new Geometry.Hot(hot.depth());
        } else if (kind instanceof AxisGatherKind axisKind
                && node.operation().attrs() instanceof IndexAxisAttrs attrs) {
            family = axisKind == AxisGatherKind.GATHER ? CpuIndexingIr.Family.GATHER
                    : CpuIndexingIr.Family.GATHER_ELEMENTS;
            variant = new Geometry.Axis(attrs.axis());
        } else if (kind == GatherNdKind.GATHER_ND
                && node.operation().attrs() instanceof GatherNdAttrs attrs) {
            var shape = require(values, node.inputs().get(1)).descriptor().shape().dimensions();
            if (!(shape.getLast() instanceof StaticDimension depth)) throw invalidBatch();
            family = CpuIndexingIr.Family.GATHER_ND;
            variant = new Geometry.Nd(attrs.batchDimensions(), Math.toIntExact(depth.size()));
        } else throw invalidBatch();
        var unique = new LinkedHashMap<ValueId, Integer>();
        var map = new ArrayList<Integer>();
        for (var id : node.inputs()) map.add(unique.computeIfAbsent(id, ignored -> unique.size()));
        if (unique.containsKey(node.outputs().getFirst())) throw invalidBatch();
        var ids = new ArrayList<>(unique.keySet());
        ids.add(node.outputs().getFirst());
        var types = new ArrayList<DataType>();
        var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int b = 0; b < ids.size(); b++) {
            var access = layouts.get(ids.get(b));
            if (access == null) throw invalidBatch();
            var plan = access.plan();
            var permission = new CpuAccessPlan(b + 1 == ids.size() ? CpuAccessPlan.AccessKind.WRITE
                    : CpuAccessPlan.AccessKind.READ, plan.regime(), plan.iterationRank(),
                    plan.axisRoles(), plan.contiguousSuffix());
            accesses.add(new CpuAccessPlan.BatchBinding(permission, access.extentConstants(),
                    access.extentBatchFactors(), access.baseElementOffset(),
                    access.strideConstants(), access.strideBatchFactors()));
            types.add(require(values, ids.get(b)).descriptor().dataType());
        }
        var ir = new CpuIndexingIr(family, map, types,
                accesses.stream().map(CpuAccessPlan.BatchBinding::plan).toList());
        return new CpuPartitionLowering.LoweredPartition(
                new Geometry.BatchGeometry(ir, ids, types, accesses, variant));
    }


    private static IllegalArgumentException invalidBatch() {
        return new IllegalArgumentException("inconsistent symbolic indexing geometry");
    }

    private static BatchByteSize batchCount(CpuAccessPlan.BatchBinding access) {
        return CpuPartitionLowering.batchElements(access);
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static CpuAccessPlan.Binding binding(long[] extents, LayoutDescriptor layout,
            CpuAccessPlan.AccessKind kind) {
        long[] strides = layout.strides();
        int suffix = 0; long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                extents.length, roles, suffix);
        long count = elementCount(extents);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), strides,
                count, 0, count, layout.referencedElementSpan());
    }

    private static long elementCount(long[] extents) {
        if (Arrays.stream(extents).anyMatch(v -> v == 0)) return 0;
        long count = 1; for (long extent : extents) count = Math.multiplyExact(count, extent);
        return count;
    }

    private static void validateInjective(long[] extents, long[] strides) {
        if (Arrays.stream(extents).anyMatch(v -> v == 0)) return;
        long count = elementCount(extents);
        if (count > 1_000_000) {
            var axes = new ArrayList<Integer>();
            for (int i = 0; i < extents.length; i++) if (extents[i] > 1) axes.add(i);
            axes.sort(java.util.Comparator.comparingLong(i -> strides[i]));
            long covered = 1;
            for (int axis : axes) {
                if (strides[axis] < covered) throw new IllegalArgumentException(
                        "indexing output layout is not injective");
                covered = Math.addExact(covered,
                        Math.multiplyExact(extents[axis] - 1, strides[axis]));
            }
            return;
        }
        var seen = new HashSet<Long>(); long[] c = new long[extents.length];
        for (long n = 0; n < count; n++) {
            long a = 0; for (int i = 0; i < c.length; i++) a = Math.addExact(a,
                    Math.multiplyExact(c[i], strides[i]));
            if (!seen.add(a)) throw new IllegalArgumentException(
                    "indexing output layout is not injective");
            for (int i = c.length - 1; i >= 0; i--) { if (++c[i] < extents[i]) break; c[i] = 0; }
        }
    }

    /**
     * Compact immutable static geometry used by validation and generated output mapping.
     *
     * @param family non-null closed indexing family
     * @param occurrenceToBoundary non-null semantic input positions mapped to unique input
     *     boundaries; copied defensively
     * @param boundaries non-null unique-input then output layouts; copied defensively
     * @param boundaryTypes non-null data types aligned with {@code boundaries}; copied defensively
     * @param variant non-null family-specific axis, tuple, or depth parameter
     */
    public record Geometry(CpuIndexingIr.Family family, List<Integer> occurrenceToBoundary,
            List<Layout> boundaries, List<DataType> boundaryTypes, Variant variant) {

        /**
         * Numeric symbolic indexing recipe. Binding resolves ordinary layouts once, retaining the
         * exact prepared IR; bounds checks and numerical mapping keep their existing owners.
         * @param ir immutable ordinary structural indexing IR
         * @param boundaries unique inputs then output identities
         * @param boundaryTypes actual represented boundary types
         * @param bindings checked affine element recipes
         * @param variant numeric axis, tuple-depth or one-hot-depth facts
         */
        public record BatchGeometry(CpuIndexingIr ir, List<ValueId> boundaries,
                List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
                Variant variant) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots immutable numeric preparation facts, without retaining Model descriptors.
             * @param ir non-null ordinary IR
             * @param boundaries non-null ordered identities
             * @param boundaryTypes non-null represented types
             * @param bindings non-null affine access facts
             * @param variant non-null numeric mapping facts
             */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries);
                boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
                Objects.requireNonNull(variant, "variant");
            }
            /** @return exact affine output element count, in unit-width cells */
            @Override public BatchByteSize iterationCount() { return batchCount(bindings.getLast()); }
            /** @return zero numeric scratch bytes */
            @Override public BatchByteSize scratchBytes() { return new BatchByteSize(0, 0, 1); }
            /** @return original output-coordinate pack cells snapshotted by numeric binding */
            @Override public List<PackReset> resets() {
                int first = 11 + ir.occurrenceToBoundary().size();
                int rank = bindings.getLast().plan().iterationRank();
                var cells = new ArrayList<PackReset>(rank);
                for (int axis = 0; axis < rank; axis++) cells.add(new PackReset(first + axis, 0));
                return List.copyOf(cells);
            }
            /** @return batch-independent, identity-free numeric recipe encoding */
            @Override public String identity() { return "indexing:" + bindings + ":" + variant; }
            /**
             * Resolves numeric invocation geometry without invoking a semantic lowerer.
             * @param batch non-negative actual batch extent
             * @return concrete ordinary lowering with the identical prepared IR
             * @throws ArithmeticException if counts, spans or strides exceed CPU limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var access = bindings.stream().map(b -> b.bind(batch)).toList();
                var layouts = access.stream().map(b -> new Layout(
                        b.extents().stream().mapToLong(Long::longValue).toArray(),
                        b.baseElementOffset(),
                        b.effectiveStrides().stream().mapToLong(Long::longValue).toArray())).toList();
                var geometry = new Geometry(ir.family(), ir.occurrenceToBoundary(), layouts,
                        boundaryTypes, variant);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, access,
                        access.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), layouts.getLast().extents(),
                        access.getLast().elementCount(), "legal: bound symbolic indexing",
                        new long[0], Optional.empty(), Optional.of(geometry));
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
            int rank = boundaries.getLast().extents.length;
            int coordinates = 11 + occurrenceToBoundary.size();
            var resets = new ArrayList<ResetCell>(rank);
            long remainder = start;
            for (int axis = rank - 1; axis >= 0; axis--) {
                long extent = boundaries.getLast().extents[axis];
                long value = extent == 0 ? 0 : remainder % extent;
                if (extent != 0) remainder /= extent;
                resets.add(new ResetCell(coordinates + axis, value));
            }
            var bases = new ArrayList<BaseCell>(boundaries.size());
            int cell = coordinates + rank;
            for (int i = 0; i < boundaries.size(); i++) {
                var layout = boundaries.get(i);
                bases.add(new BaseCell(cell + 1, i, layout.offset));
                cell += 2 + 2 * layout.extents.length;
            }
            return new TemporalRebind(bases, resets);
        }

        /**
         * Static resolved layout of one unique input or output boundary.
         *
         * @param extents non-null fully static non-negative logical extents; copied defensively
         * @param offset non-negative element offset into the eventual direct carrier
         * @param strides non-null non-negative element strides aligned with {@code extents}; copied
         *     defensively
         */
        public record Layout(long[] extents, long offset, long[] strides) {
            /**
             * Snapshots the boundary arrays.
             *
             * @throws NullPointerException if either array is {@code null}
             */
            public Layout { extents = extents.clone(); strides = strides.clone(); }
            /**
             * Returns the logical extents without exposing retained mutable state.
             *
             * @return a new defensive copy of the logical extents
             */
            @Override public long[] extents() { return extents.clone(); }
            /**
             * Returns the element strides without exposing retained mutable state.
             *
             * @return a new defensive copy of the element strides
             */
            @Override public long[] strides() { return strides.clone(); }
        }
        /** Closed indexing-specific parameters. */
        public sealed interface Variant permits Axis, Nd, Hot { }
        /**
         * Axis-gather parameter.
         *
         * @param axis normalized zero-based data axis
         */
        public record Axis(int axis) implements Variant { }
        /**
         * Gather-ND parameters.
         *
         * @param batchDimensions non-negative shared leading batch-axis count
         * @param tupleDepth positive count of data axes selected by each coordinate tuple
         */
        public record Nd(int batchDimensions, int tupleDepth)
                implements Variant { }
        /**
         * One-hot parameter.
         *
         * @param depth positive trailing indicator-axis extent
         */
        public record Hot(long depth) implements Variant { }
        /**
         * Validates and snapshots the compact geometry.
         *
         * @throws NullPointerException if a component or element is {@code null}
         * @throws IllegalArgumentException if the type and boundary counts disagree
         */
        public Geometry {
            Objects.requireNonNull(family, "family");
            occurrenceToBoundary = List.copyOf(occurrenceToBoundary);
            boundaries = List.copyOf(boundaries);
            boundaryTypes = List.copyOf(boundaryTypes);
            Objects.requireNonNull(variant, "variant");
            if (boundaryTypes.size() != boundaries.size()) throw new IllegalArgumentException(
                    "indexing geometry types must match boundaries");
        }
        /**
         * Returns output extents as an isolated array.
         *
         * @return a new defensive copy of the final boundary's extents
         */
        public long[] outputExtents() { return boundaries.getLast().extents(); }

        /**
         * Packs carrier bases and all compact mapping facts for one generated range.
         * Range-start division and remainder initialize the output coordinate once; generated
         * iteration subsequently advances it with carry/reset state.
         *
         * @param carrierBases non-null element bases aligned with the unique boundary order
         * @param start non-negative inclusive output logical ordinal
         * @param end exclusive output logical ordinal, not less than {@code start}
         * @return a new mutable invocation-private packed geometry array; never {@code null}
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if the carrier count differs from the boundary count
         * @throws ArithmeticException if a carrier base plus layout offset overflows
         */
        public long[] pack(long[] carrierBases, long start, long end) {
            if (carrierBases.length != boundaries.size()) throw new IllegalArgumentException(
                    "indexing carrier count is inconsistent");
            int outputRank = boundaries.getLast().extents.length;
            int size = 11 + occurrenceToBoundary.size() + outputRank + boundaries.size();
            for (Layout layout : boundaries) size += 2 * layout.extents.length + 2;
            long[] p = new long[size]; int x = 0;
            p[x++] = family.ordinal(); p[x++] = boundaries.size();
            p[x++] = occurrenceToBoundary.size();
            p[x++] = variant instanceof Axis a ? a.axis : -1;
            p[x++] = variant instanceof Nd n ? n.batchDimensions : 0;
            p[x++] = variant instanceof Nd n ? n.tupleDepth
                    : variant instanceof Hot h ? h.depth : 0;
            p[x++] = start; p[x++] = end;
            p[x++] = boundaries.getLast().extents.length;
            p[x++] = occurrenceToBoundary.getLast();
            p[x++] = occurrenceToBoundary.getFirst();
            for (int value : occurrenceToBoundary) p[x++] = value;
            long remainder = start;
            long[] outputExtents = boundaries.getLast().extents;
            for (int axis = outputRank - 1; axis >= 0; axis--) {
                if (outputExtents[axis] != 0) {
                    p[x + axis] = remainder % outputExtents[axis];
                    remainder /= outputExtents[axis];
                }
            }
            x += outputRank;
            for (int i = 0; i < boundaries.size(); i++) {
                Layout layout = boundaries.get(i); p[x++] = layout.extents.length;
                p[x++] = Math.addExact(carrierBases[i], layout.offset);
                for (long v : layout.extents) p[x++] = v;
                for (long v : layout.strides) p[x++] = v;
            }
            for (DataType type : boundaryTypes) p[x++] = type.ordinal();
            return p;
        }
    }
}
