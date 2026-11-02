package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one static resolved-layout masked floating sum or mean to complete output cells.
 *
 * <p>The lowerer preserves ordered {@code [data, mask, output]} boundaries, derives the Model's
 * directional right-aligned broadcast mapping without materialization, and sizes one fixed exact
 * sum state slice per simultaneously used output-cell range. Selected count remains generated
 * invocation-local state.</p>
 * <p>The separate {@link #lowerBatch} path authenticates the original canonical named-B
 * occurrence and retains only checked numeric coefficients. Cold binding creates this owner's
 * existing concrete geometry and pack once; nonleading batch axes and exact boundary types are
 * preserved without a representative graph or a batch-specialized generated class.</p>
 */
public final class CpuMaskedReductionLowering {
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
        if (node.inputs().size() != 2 || (node.operation().kind() != AggregateReductionKind.SUM
                    && node.operation().kind() != AggregateReductionKind.MEAN)
                || !(node.operation().attrs() instanceof MaskedReductionAttrs attrs))
            throw new IllegalArgumentException("masked batch attributes disagree");
        var data = query.inputs().getFirst(); var mask = query.inputs().getLast(); var output = query.outputs().getFirst();
        var type = data.dataType();
        if (type != output.dataType() || mask.dataType() != DataType.BOOL
                || !ShapeBroadcast.broadcast(data.shape(),mask.shape()).equals(data.shape()))
            throw new IllegalArgumentException("masked batch types or broadcast disagree");
        CpuAggregateLowering.normalizedAxes(data.shape(),output.shape(),
                new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(attrs.axis(),false));
        var boundaries = List.of(node.inputs().getFirst(),node.inputs().getLast(),node.outputs().getFirst());
        if (boundaries.stream().distinct().count() != 3) throw new IllegalArgumentException("masked boundaries must be distinct");
        var types = List.of(type,DataType.BOOL,type);
        var bindings = CpuAggregateLowering.batchBindings(context,boundaries,layouts);
        var in = bindings.getFirst(); var out = bindings.getLast();
        boolean[] singleton = new boolean[data.shape().rank()]; int omitted = data.shape().rank()-mask.shape().rank();
        for (int axis = 0; axis < mask.shape().rank(); axis++) singleton[omitted+axis] =
                mask.shape().dimension(axis).equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1));
        long budget = CpuAggregateLowering.batchDomainBudget(CpuAggregateLowering.batchCount(in,new int[] {attrs.axis()}));
        int limbs = stateLimbCount(type,budget); long scratch = Math.addExact(8,Math.multiplyExact(8L,limbs));
        var kind = node.operation().kind() == AggregateReductionKind.SUM ? CpuMaskedReductionIr.Kind.SUM : CpuMaskedReductionIr.Kind.MEAN;
        var ir = new CpuMaskedReductionIr(kind,type,attrs.axis(),mask.shape().rank(),singleton,budget,limbs,scratch,
                in.plan(),bindings.get(1).plan(),out.plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                boundaries, types, bindings, CpuAggregateLowering.batchCount(out), new BatchByteSize(scratch,0,8)));
    }

    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless focused masked-reduction lowerer. */
    public CpuMaskedReductionLowering() { }

    /**
     * Lowers one supported masked reduction.
     *
     * @param context non-null complete one-node CPU analysis projection
     * @return immutable three-boundary lowering with exact-state geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if semantic, layout, broadcast, or boundary facts disagree
     * @throws ArithmeticException if checked count, span, address, or state arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU masked reduction requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)
                || !(node.operation().attrs() instanceof MaskedReductionAttrs attrs)) {
            throw new IllegalArgumentException(
                    "partition contains an unsupported CPU masked reduction occurrence");
        }
        ValueId dataId = node.inputs().get(0), maskId = node.inputs().get(1);
        ValueId outputId = node.outputs().getFirst();
        if (outputId.equals(dataId) || outputId.equals(maskId)) throw new IllegalArgumentException(
                "masked-reduction output must be distinct from both inputs");
        GraphValue data = require(values, dataId), mask = require(values, maskId);
        GraphValue output = require(values, outputId);
        Layout dataLayout = layout(data), maskLayout = layout(mask), outputLayout = layout(output);
        validateInjective(outputLayout);
        int axis = attrs.axis();
        if (axis < 0 || axis >= dataLayout.extents.length) throw new IllegalArgumentException(
                "masked-reduction axis is outside the resolved data rank");
        if (!ShapeBroadcast.broadcast(data.descriptor().shape(), mask.descriptor().shape())
                .equals(data.descriptor().shape())) throw new IllegalArgumentException(
                "mask broadcast must equal the data Shape");
        long[] expectedOutput = remove(dataLayout.extents, axis);
        if (!Arrays.equals(expectedOutput, outputLayout.extents)) throw new IllegalArgumentException(
                "masked-reduction output Shape disagrees with its removed axis");
        int omitted = dataLayout.extents.length - maskLayout.extents.length;
        if (omitted < 0) throw new IllegalArgumentException("mask rank exceeds data rank");
        boolean[] singleton = new boolean[dataLayout.extents.length];
        for (int maskAxis = 0; maskAxis < maskLayout.extents.length; maskAxis++) {
            int dataAxis = omitted + maskAxis;
            long maskExtent = maskLayout.extents[maskAxis];
            if (maskExtent != 1 && maskExtent != dataLayout.extents[dataAxis]) {
                throw new IllegalArgumentException("aligned mask extent must be one or equal");
            }
            singleton[dataAxis] = maskExtent == 1;
        }
        long outputCount = elementCount(outputLayout.extents);
        long maximumDomain = dataLayout.extents[axis];
        int limbs = stateLimbCount(data.descriptor().dataType(), maximumDomain);
        long sliceBytes = Math.addExact(8L, Math.multiplyExact(8L, limbs));
        var dataBinding = binding(dataLayout, CpuAccessPlan.AccessKind.READ);
        var maskBinding = binding(maskLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        CpuMaskedReductionIr.Kind kind = node.operation().kind() == AggregateReductionKind.SUM
                ? CpuMaskedReductionIr.Kind.SUM : CpuMaskedReductionIr.Kind.MEAN;
        DataType type = data.descriptor().dataType();
        var ir = new CpuMaskedReductionIr(kind, type, axis, maskLayout.extents.length,
                singleton, maximumDomain, limbs, sliceBytes, dataBinding.plan(),
                maskBinding.plan(), outputBinding.plan());
        var geometry = new Geometry(kind, type, axis, dataLayout, maskLayout, outputLayout,
                outputCount, maximumDomain, limbs, sliceBytes);
        return new CpuPartitionLowering.LoweredPartition(ir,
                List.of(dataId, maskId, outputId),
                List.of(dataBinding, maskBinding, outputBinding),
                List.of(data.descriptor().layout().orElseThrow().referencedElementSpan(),
                        mask.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(type, DataType.BOOL, type), List.of(), new long[] {outputCount},
                outputCount, "legal: one static directional masked sum or mean", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(geometry), Optional.empty(), Optional.empty());
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("masked reduction requires non-negative layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(),
                source.strides());
    }

    private static int stateLimbCount(DataType type, long domainCount) {
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : 127;
        int cardinalityBits = domainCount <= 1 ? 0
                : 64 - Long.numberOfLeadingZeros(domainCount - 1);
        long signedBits = Math.addExact((long) emax + 1L - emin,
                Math.addExact(cardinalityBits, 1L));
        return Math.toIntExact(Math.addExact(signedBits, 63L) / 64L);
    }

    private static long[] remove(long[] source, int axis) {
        long[] result = new long[source.length - 1];
        for (int i = 0, j = 0; i < source.length; i++) if (i != axis) result[j++] = source[i];
        return result;
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) {
            roles.add(layout.strides[axis] == 0 ? CpuAccessPlan.AxisRole.BROADCAST
                    : axis >= layout.extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
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
        for (int axis = 0; axis < layout.extents.length; axis++) maximum = Math.addExact(maximum,
                Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        return Math.addExact(layout.offset, Math.addExact(maximum, 1));
    }

    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void validateInjective(Layout layout) {
            if (!CpuOutputInjectivity.isInjective(layout.extents, layout.strides)) {
                throw new IllegalArgumentException("output layout injectivity is unproved");
            }
        }

    /**
     * Resolved non-negative layout used only by cold masked-reduction geometry.
     *
     * <p>All arrays are snapshotted at construction. Accessors return defensive copies, so the
     * layout remains immutable and safe to retain in a reusable prepared recipe.</p>
     *
     * @param extents non-null logical axis extents in order; every value must be non-negative
     * @param offset non-negative carrier-relative element offset
     * @param strides non-null carrier-relative element strides in axis order; every value must be
     *     non-negative and the array length must equal {@code extents.length}
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Validates and snapshots one resolved layout.
         *
         * @throws NullPointerException if {@code extents} or {@code strides} is {@code null}
         * @throws IllegalArgumentException if the offset, an extent, or a stride is negative, or
         *     if the extent and stride ranks differ
         */
        public Layout {
            extents = Objects.requireNonNull(extents, "extents").clone();
            strides = Objects.requireNonNull(strides, "strides").clone();
            if (offset < 0 || extents.length != strides.length
                    || Arrays.stream(extents).anyMatch(v -> v < 0)
                    || Arrays.stream(strides).anyMatch(v -> v < 0)) {
                throw new IllegalArgumentException("masked-reduction layout facts disagree");
            }
        }
        /**
         * Returns the logical axis extents in data-axis order.
         *
         * @return a new array containing the non-negative logical extents; never {@code null}
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns the carrier-relative element strides in data-axis order.
         *
         * @return a new array containing the non-negative element strides; never {@code null}
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Immutable cold mapping and exact-state geometry for complete output-cell ranges.
     *
     * @param kind exact masked sum or mean meaning
     * @param dataType represented floating data/output type
     * @param axis normalized removed data axis
     * @param data resolved data layout
     * @param mask resolved directional-broadcast mask layout
     * @param output resolved injective output layout
     * @param outputCount number of independent result cells
     * @param maximumDomainCount static selected-axis extent
     * @param stateLimbCount exact fixed-width state limb count
     * @param scratchSliceBytes exact bytes per simultaneously used range
     */
    public record Geometry(CpuMaskedReductionIr.Kind kind, DataType dataType, int axis,
            Layout data, Layout mask, Layout output, long outputCount, long maximumDomainCount,
            int stateLimbCount, long scratchSliceBytes) {
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
        public record BatchGeometry(CpuMaskedReductionIr ir, List<ValueId> boundaries,
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
                return List.of();
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
                var dataBinding = actual.get(0);
                var data = new Layout(dataBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        dataBinding.baseElementOffset(),dataBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                var maskBinding = actual.get(1);
                var mask = new Layout(maskBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        maskBinding.baseElementOffset(),maskBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                var outputBinding = actual.get(2);
                var output = new Layout(outputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        outputBinding.baseElementOffset(),outputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                long work = iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(),ir.dataType(),ir.axis(),data,mask,output,work,data.extents[ir.axis()],
                        ir.stateLimbCount(),ir.scratchSliceBytes());
                return new CpuPartitionLowering.LoweredPartition(ir,boundaries,actual,
                        actual.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes,List.of(),new long[] {work},work,"legal: bound ordinary batch geometry",new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.of(geometry), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.<CpuPartitionLowering.LoweredPartition.MatmulFacts>empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
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
            int maskLayout = 13 + 2 * data.extents.length;
            int outputLayout = maskLayout + 2 + 2 * mask.extents.length;
            return new TemporalRebind(List.of(new BaseCell(12, 0, data.offset),
                    new BaseCell(maskLayout + 1, 1, mask.offset),
                    new BaseCell(outputLayout + 1, 2, output.offset)), List.of());
        }

        /**
         * Validates complete numeric geometry and sufficient exact-state capacity. Symbolic
         * recipes retain their all-admitted-count structural limb budget even when actual B
         * needs fewer limbs; a static occurrence still supplies its original exact width.
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(data, "data"); Objects.requireNonNull(mask, "mask");
            Objects.requireNonNull(output, "output");
            boolean floating = dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                    || dataType == DataType.BFLOAT16 || dataType == DataType.FLOAT16;
            if (!floating || axis < 0 || axis >= data.extents.length
                    || mask.extents.length > data.extents.length
                    || !Arrays.equals(remove(data.extents, axis), output.extents)
                    || outputCount != elementCount(output.extents)
                    || maximumDomainCount != data.extents[axis]
                    || stateLimbCount < CpuMaskedReductionLowering.stateLimbCount(
                            dataType, maximumDomainCount)
                    || scratchSliceBytes != Math.addExact(8L,
                            Math.multiplyExact(8L, stateLimbCount))) {
                throw new IllegalArgumentException("masked-reduction geometry facts disagree");
            }
            int omitted = data.extents.length - mask.extents.length;
            for (int maskAxis = 0; maskAxis < mask.extents.length; maskAxis++) {
                long extent = mask.extents[maskAxis];
                long dataExtent = data.extents[omitted + maskAxis];
                if (extent != 1 && extent != dataExtent) throw new IllegalArgumentException(
                        "mask geometry does not directionally broadcast to data");
            }
        }

        /**
         * Returns checked exact-state bytes for simultaneously used output-cell ranges.
         * @param rangeCount positive selected maximum range count
         * @return one slice per used range, or zero for an empty output
         */
        public long workspaceBytes(int rangeCount) {
            if (rangeCount <= 0) throw new IllegalArgumentException("rangeCount must be positive");
            return Math.multiplyExact(scratchSliceBytes,
                    Math.min((long) rangeCount, outputCount));
        }

        /**
         * Packs direct carrier bases and immutable mapping geometry for one invocation.
         * @param bases data, mask, and output element bases
         * @param rangeIndex non-negative worker-range ordinal
         * @return a new mutable invocation-owned primitive array
         */
        public long[] pack(long[] bases, int rangeIndex) {
            Objects.requireNonNull(bases, "bases");
            if (bases.length != 3 || rangeIndex < 0) throw new IllegalArgumentException(
                    "masked reduction requires three bases and a range index");
            int dataRank = data.extents.length, maskRank = mask.extents.length;
            int outputRank = output.extents.length;
            long[] packed = new long[11 + (2 + 2 * dataRank) + (2 + 2 * maskRank)
                    + (2 + 2 * outputRank)];
            packed[0] = kind.ordinal(); packed[1] = dataType.ordinal(); packed[2] = axis;
            packed[3] = dataRank; packed[4] = maskRank; packed[5] = outputRank;
            packed[6] = outputCount; packed[7] = maximumDomainCount;
            packed[8] = stateLimbCount; packed[9] = scratchSliceBytes;
            packed[10] = Math.multiplyExact(rangeIndex, scratchSliceBytes);
            int next = packLayout(packed, 11, data, bases[0]);
            next = packLayout(packed, next, mask, bases[1]);
            packLayout(packed, next, output, bases[2]);
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
