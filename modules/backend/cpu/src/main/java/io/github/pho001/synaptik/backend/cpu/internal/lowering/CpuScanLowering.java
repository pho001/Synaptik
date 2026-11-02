package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;

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
 * <p>The separate {@link #lowerBatch} path authenticates the original canonical named-B
 * occurrence and retains only checked numeric coefficients. Cold binding creates this owner's
 * existing concrete geometry and pack once; nonleading batch axes and exact boundary types are
 * preserved without a representative graph or a batch-specialized generated class.</p>
 */
public final class CpuScanLowering {
    /**
     * Lowers the original one-canonical-B occurrence to an unresolved numeric recipe.
     * No descriptor, graph or generated identity is specialized to a representative batch.
     * @param context complete original one-node analysis projection
     * @return immutable ordinary family recipe, resolved only during cold invocation binding
     * @throws IllegalArgumentException if original attributes, types, shapes or layouts disagree
     * @throws ArithmeticException if structural coefficient or scratch arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context,CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Authenticates an ordinary symbolic occurrence using complete enclosing physical layouts.
     * @param context original one-node projection with exact semantic descriptors
     * @param layouts immutable full-local numeric layout map, including views outside this unit
     * @return unresolved family recipe retaining those exact physical coefficients
     * @throws IllegalArgumentException if original semantics or physical geometry are unproved
     * @throws ArithmeticException if structural arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        CpuPartitionLowering.requireBatchContext(context);
        var query = CpuAggregateLowering.batchQuery(context);
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 1) throw new IllegalArgumentException("unary reduction required");
        var input = query.inputs().getFirst(); var output = query.outputs().getFirst();
        var type = input.dataType();
        var boundaries = List.of(node.inputs().getFirst(),node.outputs().getFirst());
        var types = List.of(type,output.dataType());
        var bindings = CpuAggregateLowering.batchBindings(context,boundaries,layouts);
        var in = bindings.getFirst(); var out = bindings.getLast();
        if (!(node.operation().kind() instanceof CumulativeScanKind modelKind)
                || !(node.operation().attrs() instanceof CumulativeScanAttrs attrs)
                || !input.shape().equals(output.shape()) || type != output.dataType()
                || attrs.axis() < 0 || attrs.axis() >= input.shape().rank())
            throw new IllegalArgumentException("scan batch semantics disagree");
        var kind = CpuScanIr.Kind.valueOf(modelKind.name());
        var ir = new CpuScanIr(kind,type,attrs.axis(),attrs.exclusive(),attrs.reverse(),
                in.plan(),out.plan(),CpuScanIr.SEQUENTIAL_TYPED_ROUNDING);
        var slices = CpuAggregateLowering.batchCount(in,java.util.stream.IntStream.range(0,input.shape().rank())
                .filter(a -> a != attrs.axis()).toArray());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                boundaries, types, bindings, slices, new BatchByteSize(0,0,1)));
    }

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
         * Immutable coefficient-only ordinary recipe; no graph, descriptor or invocation storage
         * survives lowering. The structural IR is reused for every actual batch.
         * @param ir authenticated structural ordinary IR
         * @param boundaries unique input/output IDs in carrier order
         * @param boundaryTypes exact represented type of every boundary
         * @param bindings numeric extent/stride coefficients in boundary order
         * @param iterationCount affine independent-range count
         * @param scratchBytes checked affine reserved scratch bytes
         */
        public record BatchGeometry(CpuScanIr ir, List<ValueId> boundaries,
                List<io.github.pho001.synaptik.model.datatype.DataType> boundaryTypes,
                List<CpuAccessPlan.BatchBinding> bindings, BatchByteSize iterationCount,
                BatchByteSize scratchBytes) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots the authenticated structural/numeric recipe without retaining a graph.
             * @param ir non-null ordinary structural IR
             * @param boundaries unique ordered boundary IDs, copied
             * @param boundaryTypes exact represented boundary types, copied
             * @param bindings complete physical coefficient recipes, copied
             * @param iterationCount checked affine capacity in complete helper ranges
             * @param scratchBytes checked affine scratch reservation in bytes
             * @throws NullPointerException if a required reference is null
             * @throws IllegalArgumentException if boundary cardinalities disagree
             */
            public BatchGeometry {
                Objects.requireNonNull(ir,"ir"); boundaries = List.copyOf(boundaries);
                boundaryTypes = List.copyOf(boundaryTypes); bindings = List.copyOf(bindings);
                Objects.requireNonNull(iterationCount,"iterationCount"); Objects.requireNonNull(scratchBytes,"scratchBytes");
                if (boundaries.size() != bindings.size() || boundaries.size() != boundaryTypes.size())
                    throw new IllegalArgumentException("batch boundary cardinality disagrees");
            }
            /** @return original zero-based mutable cursor seeds restored before every call */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset> resets() {
                int rank = ir.inputAccess().iterationRank();
                return java.util.stream.IntStream.range(0,rank).mapToObj(i -> new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(8+i,0)).toList();
            }
            /** @return deterministic coefficient identity, excluding actual B and storage */
            @Override public String identity() { return ir.structuralKey()+":bindings="+bindings+":ranges="+iterationCount+":scratch="+scratchBytes; }
            /**
             * Resolves existing concrete family geometry exactly once during cold binding.
             * @param batch non-negative actual int-representable batch
             * @return ordinary typed lowering with numeric geometry and unchanged structural key
             * @throws IllegalArgumentException if actual numerical domain policy is violated
             * @throws ArithmeticException if checked invocation geometry overflows
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var actual = bindings.stream().map(b -> b.bind(batch)).toList();
                scratchBytes.byteSize(batch);
                var inputBinding = actual.get(0);
                var input = new Layout(inputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        inputBinding.baseElementOffset(),inputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                var outputBinding = actual.get(1);
                var output = new Layout(outputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        outputBinding.baseElementOffset(),outputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                long extent = input.extents[ir.axis()];
                long work = extent == 0 ? 0 : iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(),ir.dataType(),ir.axis(),ir.exclusive(),ir.reverse(),input,output,work,extent);
                return new CpuPartitionLowering.LoweredPartition(ir,boundaries,actual,
                        actual.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes,List.of(),new long[] {work},work,"legal: bound ordinary batch geometry",new long[0],
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.of(geometry));
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
            int rank = input.extents.length;
            var resets = new ArrayList<ResetCell>(rank);
            for (int i = 0; i < rank; i++) resets.add(new ResetCell(8 + i, 0));
            return new TemporalRebind(List.of(new BaseCell(9 + rank, 0, input.offset),
                    new BaseCell(11 + 3 * rank, 1, output.offset)), resets);
        }

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
