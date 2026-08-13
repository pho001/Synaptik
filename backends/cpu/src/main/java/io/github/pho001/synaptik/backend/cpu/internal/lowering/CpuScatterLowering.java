package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
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
 * Lowers exactly one fully static resolved-layout functional scatter occurrence.
 * This analysis owner revalidates the current Model signature and derives unique boundaries,
 * compact cold coordinate geometry, structural generated identity, and the exact optional
 * floating-product scratch slice. It does not read values, allocate Runtime resources, or choose
 * a native or fallback route.
 */
public final class CpuScatterLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless scatter lowerer. */
    public CpuScatterLowering() { }

    /**
     * Lowers one supported current scatter occurrence.
     *
     * @param context non-null complete CPU partition projection
     * @return immutable single-unit lowering with compact scatter geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence, boundary identity, or output layout is
     *     unsupported
     * @throws ArithmeticException if exact counts, spans, or scratch geometry overflow
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) {
            throw new IllegalArgumentException("CPU scatter partition requires exactly one node");
        }
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) {
            throw new IllegalArgumentException("partition contains an unsupported CPU scatter occurrence");
        }

        var unique = new LinkedHashMap<ValueId, Integer>();
        var occurrenceMap = new ArrayList<Integer>(3);
        for (ValueId id : node.inputs()) {
            occurrenceMap.add(unique.computeIfAbsent(id, ignored -> unique.size()));
        }
        ValueId outputId = node.outputs().getFirst();
        if (unique.containsKey(outputId)) {
            throw new IllegalArgumentException("scatter output must be distinct from every input");
        }
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
            if (layout.storageOffset() < 0 || Arrays.stream(layout.strides()).anyMatch(v -> v < 0)) {
                throw new IllegalArgumentException("scatter requires non-negative resolved layouts");
            }
            if (i + 1 == boundaryIds.size()) validateInjective(extents, layout.strides());
            bindings.add(binding(extents, layout, i + 1 == boundaryIds.size()
                    ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
            spans.add(layout.referencedElementSpan());
            types.add(value.descriptor().dataType());
            layouts.add(new Geometry.Layout(extents, layout.storageOffset(), layout.strides()));
        }

        Object kind = node.operation().kind();
        CpuScatterIr.Family family;
        ScatterReduction reduction;
        int axis = -1, batch = 0, tuple = 0;
        if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
            var attrs = (ScatterElementsAttrs) node.operation().attrs();
            family = CpuScatterIr.Family.SCATTER_ELEMENTS;
            reduction = attrs.reduction(); axis = attrs.axis();
        } else if (kind == AxisScatterKind.SCATTER_ADD) {
            family = CpuScatterIr.Family.SCATTER_ADD;
            reduction = ScatterReduction.ADD; axis = ((IndexAxisAttrs) node.operation().attrs()).axis();
        } else if (kind == ScatterNdKind.SCATTER_ND) {
            var attrs = (ScatterNdAttrs) node.operation().attrs();
            family = CpuScatterIr.Family.SCATTER_ND; reduction = attrs.reduction();
            batch = attrs.batchDimensions();
            long[] indices = layouts.get(occurrenceMap.get(1)).extents();
            tuple = Math.toIntExact(indices[indices.length - 1]);
        } else {
            throw new IllegalArgumentException("unsupported scatter family");
        }
        long outputCount = elementCount(layouts.getLast().extents());
        long updateCount = elementCount(layouts.get(occurrenceMap.get(2)).extents());
        boolean floatingMul = reduction == ScatterReduction.MUL
                && (types.getFirst() == DataType.FLOAT64 || types.getFirst() == DataType.FLOAT32
                    || types.getFirst() == DataType.BFLOAT16)
                && outputCount != 0 && updateCount != 0;
        long maximumUpdates = maximumUpdates(family, layouts, occurrenceMap, axis, batch);
        long sliceBytes = floatingMul ? scratchSliceBytes(types.getFirst(), maximumUpdates) : 0;
        var ir = new CpuScatterIr(family, reduction, occurrenceMap, types,
                bindings.stream().map(CpuAccessPlan.Binding::plan).toList(), floatingMul ? 1 : 0);
        var geometry = new Geometry(family, reduction, occurrenceMap, layouts, types, axis, batch,
                tuple, maximumUpdates, sliceBytes);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), layouts.getLast().extents(), outputCount,
                "legal: one fully static resolved-layout functional scatter occurrence",
                new long[0], Optional.empty(), Optional.empty(), Optional.of(geometry),
                Optional.empty(), Optional.empty());
    }

    private static long maximumUpdates(CpuScatterIr.Family family, List<Geometry.Layout> layouts,
            List<Integer> map, int axis, int batch) {
        long[] indices = layouts.get(map.get(1)).extents();
        if (elementCount(indices) == 0) return 0;
        if (family == CpuScatterIr.Family.SCATTER_ELEMENTS) return indices[axis];
        if (family == CpuScatterIr.Family.SCATTER_ADD) return elementCount(indices);
        long result = 1;
        for (int i = batch; i < indices.length - 1; i++) result = Math.multiplyExact(result, indices[i]);
        return result;
    }

    private static long scratchSliceBytes(DataType type, long maximumUpdates) {
        int precision = type == DataType.FLOAT64 ? 53 : type == DataType.FLOAT32 ? 24 : 8;
        long factorCount = Math.addExact(maximumUpdates, 1);
        Math.multiplyExact(factorCount, maximumExponentMagnitude(type));
        long bits = Math.multiplyExact((long) precision, factorCount);
        long limbs = Math.floorDiv(Math.addExact(bits, 63), 64);
        return Math.addExact(24, Math.multiplyExact(8, limbs));
    }

    private static long maximumExponentMagnitude(DataType type) {
        return type == DataType.FLOAT64 ? 1074 : type == DataType.FLOAT32 ? 149 : 133;
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
        for (int a = extents.length - 1; a >= 0; a--) {
            if (strides[a] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[a]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int a = 0; a < extents.length; a++) roles.add(strides[a] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : a >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
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
            for (int a : axes) {
                if (strides[a] < covered) throw new IllegalArgumentException(
                        "scatter output layout is not injective");
                covered = Math.addExact(covered, Math.multiplyExact(extents[a] - 1, strides[a]));
            }
            return;
        }
        var seen = new HashSet<Long>(); long[] coordinate = new long[extents.length];
        for (long n = 0; n < count; n++) {
            long address = 0;
            for (int i = 0; i < coordinate.length; i++) address = Math.addExact(address,
                    Math.multiplyExact(coordinate[i], strides[i]));
            if (!seen.add(address)) throw new IllegalArgumentException(
                    "scatter output layout is not injective");
            advance(coordinate, extents);
        }
    }

    private static void advance(long[] coordinate, long[] extents) {
        for (int i = coordinate.length - 1; i >= 0; i--) {
            if (++coordinate[i] < extents[i]) return;
            coordinate[i] = 0;
        }
    }

    /**
     * Compact immutable scatter mapping and scratch geometry retained by the prepared recipe.
     * @param family current coordinate family
     * @param reduction exact represented-value reduction
     * @param occurrenceToBoundary semantic data, indices, and updates boundary positions
     * @param boundaries unique inputs followed by the output layout
     * @param boundaryTypes data types aligned with {@code boundaries}
     * @param axis normalized selected axis, or {@code -1} for scatter-ND
     * @param batchDimensions non-negative shared scatter-ND batch count
     * @param tupleDepth positive scatter-ND tuple depth, or zero for axis families
     * @param maximumUpdatesPerTarget checked maximum target-group update count
     * @param scratchSliceBytes exact aligned per-range scratch bytes, or zero when absent
     */
    public record Geometry(CpuScatterIr.Family family, ScatterReduction reduction,
            List<Integer> occurrenceToBoundary, List<Layout> boundaries,
            List<DataType> boundaryTypes, int axis, int batchDimensions, int tupleDepth,
            long maximumUpdatesPerTarget, long scratchSliceBytes) {
        /**
         * Static resolved layout of one unique input or output boundary.
         * @param extents fully static non-negative logical extents
         * @param offset non-negative element offset
         * @param strides non-negative element strides aligned with {@code extents}
         */
        public record Layout(long[] extents, long offset, long[] strides) {
            /**
             * Snapshots arrays and validates rank agreement.
             * @throws NullPointerException if an array is {@code null}
             * @throws IllegalArgumentException if ranks differ or an extent, offset, or stride is
             *     negative
             */
            public Layout {
                extents = extents.clone(); strides = strides.clone();
                if (extents.length != strides.length || offset < 0
                        || Arrays.stream(extents).anyMatch(extent -> extent < 0)
                        || Arrays.stream(strides).anyMatch(stride -> stride < 0)) {
                    throw new IllegalArgumentException("scatter layout rank facts disagree");
                }
            }
            /** Returns a caller-owned snapshot of the logical extents.
             * @return a defensive copy of logical extents */
            @Override public long[] extents() { return extents.clone(); }
            /** Returns a caller-owned snapshot of the element strides.
             * @return a defensive copy of element strides */
            @Override public long[] strides() { return strides.clone(); }
        }

        /**
         * Validates and snapshots compact geometry.
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if occurrence mapping, types, family shape rules, or
         *     exact scratch facts disagree
         */
        public Geometry {
            Objects.requireNonNull(family, "family"); Objects.requireNonNull(reduction, "reduction");
            occurrenceToBoundary = List.copyOf(occurrenceToBoundary);
            boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
            if (occurrenceToBoundary.size() != 3 || boundaries.size() != boundaryTypes.size()
                    || boundaries.size() < 2 || boundaries.size() > 4
                    || maximumUpdatesPerTarget < 0 || scratchSliceBytes < 0) {
                throw new IllegalArgumentException("scatter geometry facts disagree");
            }
            int inputBoundaryCount = boundaries.size() - 1;
            var referencedInputs = new HashSet<Integer>();
            for (int boundary : occurrenceToBoundary) {
                if (boundary < 0 || boundary >= inputBoundaryCount) {
                    throw new IllegalArgumentException("scatter geometry occurrence is out of range");
                }
                referencedInputs.add(boundary);
            }
            if (referencedInputs.size() != inputBoundaryCount) {
                throw new IllegalArgumentException("scatter geometry must cover every input");
            }
            int dataBoundary = occurrenceToBoundary.get(0);
            int indexBoundary = occurrenceToBoundary.get(1);
            int updateBoundary = occurrenceToBoundary.get(2);
            DataType dataType = boundaryTypes.get(dataBoundary);
            DataType indexType = boundaryTypes.get(indexBoundary);
            if (indexType != DataType.INT32 && indexType != DataType.INT64
                    || boundaryTypes.get(updateBoundary) != dataType
                    || boundaryTypes.getLast() != dataType
                    || dataType == DataType.BOOL && reduction != ScatterReduction.NONE
                    || !Arrays.equals(boundaries.get(dataBoundary).extents,
                            boundaries.getLast().extents)) {
                throw new IllegalArgumentException("scatter geometry boundary facts disagree");
            }
            validateFamilyGeometry(family, reduction, boundaries, occurrenceToBoundary,
                    axis, batchDimensions, tupleDepth);
            long expectedMaximum = maximumUpdates(family, boundaries, occurrenceToBoundary,
                    axis, batchDimensions);
            boolean floatingProduct = reduction == ScatterReduction.MUL
                    && (dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                        || dataType == DataType.BFLOAT16)
                    && elementCount(boundaries.getLast().extents) != 0
                    && elementCount(boundaries.get(updateBoundary).extents) != 0;
            long expectedSlice = floatingProduct
                    ? CpuScatterLowering.scratchSliceBytes(dataType, expectedMaximum) : 0;
            if (maximumUpdatesPerTarget != expectedMaximum || scratchSliceBytes != expectedSlice) {
                throw new IllegalArgumentException("scatter geometry scratch facts disagree");
            }
        }

        /** Returns a caller-owned snapshot of the output extents.
         * @return a defensive copy of output extents */
        public long[] outputExtents() { return boundaries.getLast().extents(); }

        /**
         * Returns total exact scratch bytes for the selected range count.
         * @param rangeCount positive number of selected disjoint ranges
         * @return checked total workspace bytes, or zero when scratch is absent
         * @throws IllegalArgumentException if {@code rangeCount} is not positive
         * @throws ArithmeticException if multiplication overflows
         */
        public long workspaceBytes(int rangeCount) {
            if (rangeCount <= 0) throw new IllegalArgumentException("rangeCount must be positive");
            return Math.multiplyExact(scratchSliceBytes, rangeCount);
        }

        /**
         * Packs cold carrier bases, range-start state, and an optional disjoint scratch offset.
         *
         * @param carrierBases element bases aligned with unique boundaries
         * @param start inclusive output ordinal
         * @param end exclusive output ordinal
         * @param rangeIndex zero-based selected range index
         * @return new invocation-private primitive geometry
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if carrier count or range index is invalid
         * @throws ArithmeticException if a carrier base or scratch offset overflows
         */
        public long[] pack(long[] carrierBases, long start, long end, int rangeIndex) {
            if (carrierBases.length != boundaries.size() || rangeIndex < 0) {
                throw new IllegalArgumentException("scatter packed geometry facts disagree");
            }
            int outRank = boundaries.getLast().extents.length;
            int updateRank = boundaries.get(occurrenceToBoundary.get(2)).extents.length;
            int size = 16 + 2 * outRank + updateRank + boundaries.size();
            for (Layout layout : boundaries) size += 2 + 2 * layout.extents.length;
            long[] p = new long[size]; int x = 0;
            p[x++] = family.ordinal(); p[x++] = reduction.ordinal(); p[x++] = boundaries.size();
            p[x++] = occurrenceToBoundary.get(0); p[x++] = occurrenceToBoundary.get(1);
            p[x++] = occurrenceToBoundary.get(2); p[x++] = axis; p[x++] = batchDimensions;
            p[x++] = tupleDepth; p[x++] = start; p[x++] = end; p[x++] = outRank;
            p[x++] = updateRank; p[x++] = Math.multiplyExact(rangeIndex, scratchSliceBytes);
            p[x++] = scratchSliceBytes; p[x++] = maximumUpdatesPerTarget;
            long remainder = start;
            long[] output = boundaries.getLast().extents;
            for (int a = outRank - 1; a >= 0; a--) {
                if (output[a] != 0) { p[x + a] = remainder % output[a]; remainder /= output[a]; }
            }
            System.arraycopy(p, x, p, x + outRank, outRank);
            x += 2 * outRank + updateRank;
            for (int i = 0; i < boundaries.size(); i++) {
                Layout layout = boundaries.get(i); p[x++] = layout.extents.length;
                p[x++] = Math.addExact(carrierBases[i], layout.offset);
                for (long value : layout.extents) p[x++] = value;
                for (long value : layout.strides) p[x++] = value;
            }
            for (DataType type : boundaryTypes) p[x++] = type.ordinal();
            return p;
        }
    }

    private static void validateFamilyGeometry(CpuScatterIr.Family family,
            ScatterReduction reduction, List<Geometry.Layout> boundaries, List<Integer> map,
            int axis, int batch, int tuple) {
        long[] data = boundaries.get(map.get(0)).extents;
        long[] indices = boundaries.get(map.get(1)).extents;
        long[] updates = boundaries.get(map.get(2)).extents;
        if (family == CpuScatterIr.Family.SCATTER_ELEMENTS) {
            if (axis < 0 || axis >= data.length || batch != 0 || tuple != 0
                    || !Arrays.equals(indices, updates) || indices.length != data.length) {
                throw new IllegalArgumentException("scatter-elements geometry is inconsistent");
            }
            for (int current = 0; current < data.length; current++) {
                if (current != axis && data[current] != indices[current]) {
                    throw new IllegalArgumentException("scatter-elements geometry is inconsistent");
                }
            }
            return;
        }
        if (family == CpuScatterIr.Family.SCATTER_ADD) {
            if (reduction != ScatterReduction.ADD || axis < 0 || axis >= data.length
                    || batch != 0 || tuple != 0
                    || updates.length != data.length - 1 + indices.length) {
                throw new IllegalArgumentException("scatter-add geometry is inconsistent");
            }
            int position = 0;
            for (int current = 0; current < axis; current++) {
                if (updates[position++] != data[current]) {
                    throw new IllegalArgumentException("scatter-add geometry is inconsistent");
                }
            }
            for (long extent : indices) if (updates[position++] != extent) {
                throw new IllegalArgumentException("scatter-add geometry is inconsistent");
            }
            for (int current = axis + 1; current < data.length; current++) {
                if (updates[position++] != data[current]) {
                    throw new IllegalArgumentException("scatter-add geometry is inconsistent");
                }
            }
            return;
        }
        if (axis != -1 || indices.length == 0 || batch < 0
                || batch >= Math.min(data.length, indices.length)
                || tuple < 1 || tuple > data.length - batch
                || indices[indices.length - 1] != tuple
                || updates.length != indices.length - 1 + data.length - batch - tuple) {
            throw new IllegalArgumentException("scatter-ND geometry is inconsistent");
        }
        for (int current = 0; current < batch; current++) {
            if (data[current] != indices[current]) {
                throw new IllegalArgumentException("scatter-ND geometry is inconsistent");
            }
        }
        int position = 0;
        for (int current = 0; current < indices.length - 1; current++) {
            if (updates[position++] != indices[current]) {
                throw new IllegalArgumentException("scatter-ND geometry is inconsistent");
            }
        }
        for (int current = batch + tuple; current < data.length; current++) {
            if (updates[position++] != data[current]) {
                throw new IllegalArgumentException("scatter-ND geometry is inconsistent");
            }
        }
    }
}
