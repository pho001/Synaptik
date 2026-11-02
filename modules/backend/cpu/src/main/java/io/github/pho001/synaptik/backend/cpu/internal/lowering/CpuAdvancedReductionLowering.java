package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
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
 * Lowers one resolved advanced reduction to complete output-cell ranges.
 * <p>The separate {@link #lowerBatch} path authenticates the original canonical named-B
 * occurrence and retains only checked numeric coefficients. Cold binding creates this owner's
 * existing concrete geometry and pack once; nonleading batch axes and exact boundary types are
 * preserved without a representative graph or a batch-specialized generated class.</p>
 */
public final class CpuAdvancedReductionLowering {
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
        if (!(node.operation().kind() instanceof AggregateReductionKind modelKind) || type != output.dataType())
            throw new IllegalArgumentException("advanced batch type disagrees");
        var kind = CpuAdvancedReductionIr.Kind.valueOf(modelKind.name());
        List<Integer> axes; boolean keep; long correction;
        if (node.operation().attrs() instanceof StatisticalReductionAttrs attrs) {
            axes = attrs.axes(); keep = attrs.keepDimensions(); correction = attrs.correction();
        } else if (node.operation().attrs() instanceof MultiAxisReductionAttrs attrs) {
            axes = attrs.axes(); keep = attrs.keepDimensions(); correction = 0;
        } else throw new IllegalArgumentException("advanced batch attributes disagree");
        boolean statistics = kind == CpuAdvancedReductionIr.Kind.VARIANCE || kind == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION;
        if (statistics != (node.operation().attrs() instanceof StatisticalReductionAttrs))
            throw new IllegalArgumentException("statistical attributes disagree");
        CpuAggregateLowering.normalizedAxes(input.shape(),output.shape(),new MultiAxisReductionAttrs(axes,keep));
        int[] ordered = axes.stream().mapToInt(Integer::intValue).toArray();
        boolean[] selected = new boolean[input.shape().rank()]; for (int axis : ordered) selected[axis] = true;
        var domain = CpuAggregateLowering.batchCount(in,ordered);
        if (statistics && domain.bytesPerBatch() == 0 && domain.constantBytes() <= correction)
            throw new IllegalArgumentException("statistical selected count must exceed correction");
        long budget = CpuAggregateLowering.batchDomainBudget(domain);
        long scratch = kind == CpuAdvancedReductionIr.Kind.L1_NORM || statistics ? exactStateSliceBytes(type,budget) : 0;
        int limbs = scratch == 0 ? 0 : Math.toIntExact(scratch/8-1);
        var ir = new CpuAdvancedReductionIr(kind,type,ordered,selected,keep,correction,1,
                kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP || statistics ? 2 : 1,budget,limbs,scratch,in.plan(),out.plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                boundaries, types, bindings, CpuAggregateLowering.batchCount(out), new BatchByteSize(scratch,0,8)));
    }

    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless advanced-reduction lowerer. */
    public CpuAdvancedReductionLowering() { }

    /**
     * Lowers one supported occurrence and derives checked cold geometry.
     *
     * @param context non-null complete one-node CPU analysis projection
     * @return one immutable two-boundary lowering
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if semantic, descriptor, or layout facts disagree
     * @throws ArithmeticException if count, span, or resource arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU advanced reduction requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU advanced reduction occurrence");
        ValueId inputId = node.inputs().getFirst(), outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) throw new IllegalArgumentException(
                "advanced-reduction output must be distinct from input");
        GraphValue input = require(values, inputId), output = require(values, outputId);
        Layout inputLayout = layout(input), outputLayout = layout(output);
        validateInjective(outputLayout);
        int[] orderedAxes; boolean keep; long correction;
        if (node.operation().attrs() instanceof StatisticalReductionAttrs attrs) {
            orderedAxes = attrs.axes().stream().mapToInt(Integer::intValue).toArray();
            keep = attrs.keepDimensions(); correction = attrs.correction();
        } else {
            MultiAxisReductionAttrs attrs =
                    (MultiAxisReductionAttrs) node.operation().attrs();
            orderedAxes = attrs.axes().stream().mapToInt(Integer::intValue).toArray();
            keep = attrs.keepDimensions(); correction = 0;
        }
        boolean[] selected = new boolean[inputLayout.extents.length];
        for (int axis : orderedAxes) selected[axis] = true;
        long[] expected = reduced(inputLayout.extents, selected, keep);
        if (!Arrays.equals(expected, outputLayout.extents)) throw new IllegalArgumentException(
                "advanced-reduction output Shape disagrees with selected axes");
        long domainCount = 1;
        for (int axis = 0; axis < selected.length; axis++) if (selected[axis])
            domainCount = Math.multiplyExact(domainCount, inputLayout.extents[axis]);
        AggregateReductionKind modelKind = (AggregateReductionKind) node.operation().kind();
        CpuAdvancedReductionIr.Kind kind = CpuAdvancedReductionIr.Kind.valueOf(modelKind.name());
        boolean statistics = kind == CpuAdvancedReductionIr.Kind.VARIANCE
                || kind == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION;
        if (statistics && domainCount <= correction) throw new IllegalArgumentException(
                "statistical selected-domain count must exceed correction");
        long outputCount = elementCount(outputLayout.extents);
        long scratchSliceBytes = (kind == CpuAdvancedReductionIr.Kind.L1_NORM || statistics)
                && outputCount > 0 ? exactStateSliceBytes(input.descriptor().dataType(), domainCount)
                : 0;
        int stateLimbCount = scratchSliceBytes == 0 ? 0
                : Math.toIntExact(scratchSliceBytes / Long.BYTES - 1);
        var inputBinding = binding(inputLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuAdvancedReductionIr(kind, input.descriptor().dataType(), orderedAxes,
                selected, keep, correction, 1,
                kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP || statistics ? 2 : 1,
                domainCount, stateLimbCount, scratchSliceBytes, inputBinding.plan(),
                outputBinding.plan());
        var geometry = new Geometry(kind, input.descriptor().dataType(), orderedAxes, selected,
                keep, correction, inputLayout, outputLayout, outputCount, domainCount,
                scratchSliceBytes);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding),
                List.of(input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(input.descriptor().dataType(), output.descriptor().dataType()), List.of(),
                new long[] {outputCount}, outputCount,
                "legal: one static advanced floating reduction", new long[0], Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(geometry), Optional.empty());
    }

    private static long exactStateSliceBytes(DataType type, long count) {
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149
                : type == DataType.FLOAT16 ? -24 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : type == DataType.FLOAT16 ? 15 : 127;
        int countBits = count <= 1 ? 0 : 64 - Long.numberOfLeadingZeros(count - 1);
        long bits = Math.addExact((long) emax + 1 - emin, Math.addExact(countBits, 1));
        long limbs = Math.addExact(bits, 63) / 64;
        return Math.addExact(8, Math.multiplyExact(8, limbs));
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("advanced reduction requires non-negative layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(),
                source.strides());
    }

    private static long[] reduced(long[] source, boolean[] selected, boolean keep) {
        long[] result = new long[keep ? source.length : source.length - count(selected)];
        for (int input = 0, output = 0; input < source.length; input++) {
            if (selected[input]) { if (keep) result[output++] = 1; }
            else result[output++] = source[input];
        }
        return result;
    }

    private static int count(boolean[] values) { int result = 0; for (boolean v : values) if (v) result++; return result; }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) roles.add(
                layout.strides[axis] == 0 ? CpuAccessPlan.AxisRole.BROADCAST
                        : axis >= layout.extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                        : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == layout.extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                layout.extents.length, roles, suffix);
        long elements = elementCount(layout.extents);
        return CpuAccessPlan.Binding.create(plan, layout.extents, layout.offset, layout.strides,
                elements, 0, elements, referencedSpan(layout));
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
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void validateInjective(Layout layout) {
            if (!CpuOutputInjectivity.isInjective(layout.extents, layout.strides)) {
                throw new IllegalArgumentException("output layout injectivity is unproved");
            }
        }

    /**
     * Resolved non-negative cold layout.
     *
     * @param extents non-null static non-negative Shape extents, copied defensively
     * @param offset non-negative storage offset in represented elements
     * @param strides non-null non-negative element strides, copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /** Validates references and snapshots both arrays. */
        public Layout {
            extents = Objects.requireNonNull(extents, "extents").clone();
            strides = Objects.requireNonNull(strides, "strides").clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(value -> value < 0)
                    || Arrays.stream(strides).anyMatch(value -> value < 0)) {
                throw new IllegalArgumentException("advanced-reduction layout is invalid");
            }
        }
        /**
         * Returns the statically resolved extents for this layout.
         * @return a new copy of the static extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns the statically resolved element strides for this layout.
         * @return a new copy of the element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete cold geometry for one advanced reduction.
     *
     * @param kind non-null exact advanced meaning
     * @param dataType non-null identical floating input/output type
     * @param orderedAxes non-null normalized ordered selected axes, copied defensively
     * @param selectedAxes non-null canonical input-axis membership, copied defensively
     * @param keepDimensions whether selected axes remain as extent-one output axes
     * @param correction non-negative statistical denominator correction, otherwise zero
     * @param input non-null resolved input layout
     * @param output non-null resolved injective output layout
     * @param outputCount checked number of complete output cells
     * @param domainCount checked number of represented inputs per output cell
     * @param scratchSliceBytes exact per-range state bytes, or zero
     */
    public record Geometry(CpuAdvancedReductionIr.Kind kind, DataType dataType, int[] orderedAxes,
            boolean[] selectedAxes, boolean keepDimensions, long correction, Layout input,
            Layout output, long outputCount, long domainCount, long scratchSliceBytes) {
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
        public record BatchGeometry(CpuAdvancedReductionIr ir, List<ValueId> boundaries,
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
                var inputBinding = actual.get(0);
                var input = new Layout(inputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        inputBinding.baseElementOffset(),inputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                var outputBinding = actual.get(1);
                var output = new Layout(outputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        outputBinding.baseElementOffset(),outputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                long domain = 1; for (int axis : ir.orderedAxes()) domain = Math.multiplyExact(domain,input.extents[axis]);
                if ((ir.kind() == CpuAdvancedReductionIr.Kind.VARIANCE || ir.kind() == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION)
                        && domain <= ir.correction()) throw new IllegalArgumentException("statistical selected count must exceed correction");
                long work = iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(),ir.dataType(),ir.orderedAxes(),ir.selectedAxes(),ir.keepDimensions(),
                        ir.correction(),input,output,work,domain,ir.scratchSliceBytes());
                return new CpuPartitionLowering.LoweredPartition(ir,boundaries,actual,
                        actual.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes,List.of(),new long[] {work},work,"legal: bound ordinary batch geometry",new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.of(geometry), Optional.empty(),
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
            return new TemporalRebind(List.of(new BaseCell(0, 0, input.offset),
                    new BaseCell(1, 1, output.offset)), List.of());
        }

        /** Validates references and snapshots axis arrays. */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            orderedAxes = Objects.requireNonNull(orderedAxes, "orderedAxes").clone();
            selectedAxes = Objects.requireNonNull(selectedAxes, "selectedAxes").clone();
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            if (selectedAxes.length != input.extents.length || correction < 0 || outputCount < 0
                    || domainCount < 0 || scratchSliceBytes < 0
                    || scratchSliceBytes % Long.BYTES != 0) {
                throw new IllegalArgumentException("advanced-reduction geometry is invalid");
            }
        }
        /**
         * Returns the normalized axes without changing their Model-owned order.
         * @return a new copy of normalized ordered axes
         */
        @Override public int[] orderedAxes() { return orderedAxes.clone(); }
        /**
         * Returns canonical membership flags indexed by input axis.
         * @return a new copy of canonical selected-axis membership
         */
        @Override public boolean[] selectedAxes() { return selectedAxes.clone(); }
        /**
         * Packs invocation-local carrier bases and retained static layout geometry.
         * @param bases exact input/output carrier element bases
         * @return a new primitive geometry array consumed only by the generated entry
         */
        public long[] pack(long[] bases) {
            if (bases.length != 2) throw new IllegalArgumentException(
                    "advanced-reduction geometry requires two carrier bases");
            int inputRank = input.extents.length, outputRank = output.extents.length;
            long[] packed = new long[11 + 2 * inputRank + 2 * outputRank];
            packed[0] = Math.addExact(bases[0], input.offset);
            packed[1] = Math.addExact(bases[1], output.offset);
            packed[2] = domainCount;
            packed[3] = correction;
            packed[4] = inputRank;
            packed[5] = outputRank;
            packed[7] = domainCount;
            packed[9] = scratchSliceBytes;
            packed[10] = 0;
            System.arraycopy(input.extents, 0, packed, 11, inputRank);
            System.arraycopy(input.strides, 0, packed, 11 + inputRank, inputRank);
            int outputExtents = 11 + 2 * inputRank;
            System.arraycopy(output.extents, 0, packed, outputExtents, outputRank);
            System.arraycopy(output.strides, 0, packed, outputExtents + outputRank, outputRank);
            return packed;
        }
        /**
         * Computes the exact run-owned state size.
         *
         * @param ranges non-negative number of simultaneously selected complete-cell ranges
         * @return checked bytes required for all selected ranges
         * @throws ArithmeticException if the product exceeds {@code long}
         */
        public long workspaceBytes(int ranges) { return Math.multiplyExact(scratchSliceBytes, ranges); }
    }
}
