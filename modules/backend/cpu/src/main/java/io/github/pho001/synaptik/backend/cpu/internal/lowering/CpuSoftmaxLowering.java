package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
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
 * Lowers one supported static, resolved softmax occurrence to complete normalization-slice
 * ranges and a version-two CPU-private kernel identity. The identity binds the selected axis,
 * representation, three-pass shape, and access plans; run-time values remain for the separate
 * pre-write validator rather than becoming lowering facts.
 * <p>General Scan may retain a single named-axis numeric recipe through {@link #lowerBatch};
 * invocation binding resolves its complete slice geometry without semantic relowering.</p>
 */
public final class CpuSoftmaxLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless softmax lowerer. */
    public CpuSoftmaxLowering() { }

    /**
     * Lowers one supported occurrence and derives checked cold slice geometry.
     * @param context non-null complete one-node CPU analysis projection
     * @return one immutable two-boundary lowering ranged by complete slices
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if semantic, descriptor, or layout facts disagree
     * @throws ArithmeticException if count or address arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU softmax requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU softmax occurrence");
        ValueId inputId = node.inputs().getFirst(), outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) throw new IllegalArgumentException(
                "softmax output must be distinct from input");
        GraphValue input = require(values, inputId), output = require(values, outputId);
        Layout in = layout(input), out = layout(output);
        validateInjective(out);
        int axis = ((SoftmaxAttrs) node.operation().attrs()).axis();
        long width = in.extents[axis];
        if (width <= 0) throw new IllegalArgumentException(
                "softmax selected-axis extent must be positive");
        long elements = elementCount(in.extents);
        long slices = elements == 0 ? 0 : elements / width;
        var inputBinding = binding(in, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(out, CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuSoftmaxIr((SoftmaxKind) node.operation().kind(),
                input.descriptor().dataType(), axis, 2, 3,
                inputBinding.plan(), outputBinding.plan());
        var geometry = new Geometry(ir.kind(), ir.dataType(), axis, in, out, slices, width,
                elements);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding),
                List.of(input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(ir.dataType(), ir.dataType()), List.of(), new long[] {slices}, slices,
                "legal: one static stable softmax", new long[0], Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    /**
     * Qualifies original single-B softmax descriptors and retains symbolic slice geometry.
     * The selected axis may itself carry B; its ordinary positive-width guard is checked at bind.
     * @param context complete one-node ordinary projection, with canonical named-axis layouts
     * @return unresolved numeric geometry, never a sampled concrete lowering
     * @throws NullPointerException if the context or a required fact is null
     * @throws IllegalArgumentException if original descriptors or attributes disagree
     * @throws ArithmeticException if fixed coefficient arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers using the complete local graph's already proved physical access mappings.
     * No subset graph is used to reconstruct view strides, origins, or symbolic axis positions.
     * @param context original complete one-node descriptor projection
     * @param layouts complete local value-to-layout recipes; neither retained nor mutated
     * @return unresolved family geometry retaining only immutable numeric access facts
     * @throws NullPointerException if either argument or a required fact is null
     * @throws IllegalArgumentException if a mapping, descriptor or family predicate disagrees
     * @throws ArithmeticException if fixed coefficients exceed supported geometry limits
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        Objects.requireNonNull(layouts, "layouts");
        Objects.requireNonNull(context, "context");
        CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1) throw new IllegalArgumentException("softmax needs one node");
        var node = context.nodes().getFirst();
        if (!(node.operation().kind() instanceof SoftmaxKind kind)
                || !(node.operation().attrs() instanceof SoftmaxAttrs attrs)
                || node.inputs().size() != 1 || node.outputs().size() != 1
                || node.inputs().getFirst().equals(node.outputs().getFirst()))
            throw new IllegalArgumentException("unsupported symbolic softmax occurrence");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var input = require(values, node.inputs().getFirst());
        var output = require(values, node.outputs().getFirst());
        var in = input.descriptor(); var out = output.descriptor(); int axis = attrs.axis();
        var read = batchBinding(input, CpuAccessPlan.AccessKind.READ, layouts);
        var write = batchBinding(output, CpuAccessPlan.AccessKind.WRITE, layouts);
        if (read.extentConstants().get(axis) == 0 && read.extentBatchFactors().get(axis) == 0)
            throw new IllegalArgumentException("softmax width must be positive");
        long c = 1, f = 0;
        for (int a = 0; a < in.shape().rank(); a++) if (a != axis) {
            long ec = read.extentConstants().get(a), ef = read.extentBatchFactors().get(a);
            f = Math.addExact(Math.multiplyExact(f, ec), Math.multiplyExact(c, ef));
            c = Math.multiplyExact(c, ec);
        }
        var ir = new CpuSoftmaxIr(kind, in.dataType(), axis, 2, 3, read.plan(), write.plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                List.of(input.id(), output.id()), List.of(in.dataType(), out.dataType()),
                List.of(read, write), new BatchByteSize(c, f, 1), new BatchByteSize(0, 0, 1)));
    }


    private static CpuAccessPlan.BatchBinding batchBinding(GraphValue value,
            CpuAccessPlan.AccessKind kind, Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        var source = layouts.get(value.id());
        if (source == null || source.plan().iterationRank() != value.descriptor().shape().rank())
            throw new IllegalArgumentException("missing or inconsistent full-local normalization layout");
        var exact = CpuPartitionLowering.batchAccess(kind, source.extentConstants(),
                source.extentBatchFactors(), source.strideConstants(), source.strideBatchFactors());
        return new CpuAccessPlan.BatchBinding(exact.plan(), exact.extentConstants(),
                exact.extentBatchFactors(), source.baseElementOffset(), exact.strideConstants(),
                exact.strideBatchFactors());
    }


    private static Layout boundLayout(CpuAccessPlan.Binding binding) {
        return new Layout(binding.extents().stream().mapToLong(Long::longValue).toArray(),
                binding.baseElementOffset(), binding.effectiveStrides().stream()
                        .mapToLong(Long::longValue).toArray());
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("softmax requires non-negative layouts");
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
        long count = elementCount(layout.extents); if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>(); long[] coordinates = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = layout.offset;
                for (int axis = 0; axis < coordinates.length; axis++) address = Math.addExact(
                        address, Math.multiplyExact(coordinates[axis], layout.strides[axis]));
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "softmax output layout must be injective");
                for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                    if (++coordinates[axis] < layout.extents[axis]) break; coordinates[axis] = 0;
                }
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < layout.extents.length; axis++) if (layout.extents[axis] > 1) axes.add(axis);
        axes.sort(java.util.Comparator.comparingLong(axis -> layout.strides[axis]));
        long covered = 1;
        for (int axis : axes) { if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                "softmax output layout must be injective");
            covered = Math.addExact(covered, Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis])); }
    }

    /**
     * Resolved non-negative shape and element-stride geometry.
     * @param extents static non-negative Shape extents; copied defensively
     * @param offset non-negative storage element offset
     * @param strides non-negative storage element strides; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /** Validates and snapshots the resolved layout. */
        public Layout {
            extents = Objects.requireNonNull(extents, "extents").clone();
            strides = Objects.requireNonNull(strides, "strides").clone();
            if (extents.length == 0 || extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(v -> v < 0)
                    || Arrays.stream(strides).anyMatch(v -> v < 0))
                throw new IllegalArgumentException("softmax layout is invalid");
        }
        /** Returns the retained Shape extents without exposing mutable state.
         * @return a new copy of the static extents */
        @Override public long[] extents() { return extents.clone(); }
        /** Returns the retained element strides without exposing mutable state.
         * @return a new copy of the resolved element strides */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete cold geometry for shape-preserving normalization slices.
     * @param kind exact first-class normalization meaning
     * @param dataType identical represented input/output type
     * @param axis normalized selected axis
     * @param input resolved input layout
     * @param output resolved injective output layout
     * @param sliceCount number of independent complete normalization slices
     * @param sliceWidth positive selected-axis extent
     * @param elementCount shape-preserving represented element count
     */
    public record Geometry(SoftmaxKind kind, DataType dataType, int axis, Layout input,
            Layout output, long sliceCount, long sliceWidth, long elementCount) {

        /**
         * Numeric single-B geometry, retaining no graph, descriptor, or invocation storage.
         * Coefficients are resolved only at the cold invocation boundary; the compiled IR is
         * identical for every admitted B and each output retains its canonical semantic slot.
         * @param ir structurally qualified family IR, independent of actual B
         * @param boundaries unique reads followed by every ordered semantic output
         * @param boundaryTypes exact represented types in boundary order
         * @param bindings immutable per-boundary affine element geometry
         * @param iterationCount affine complete-range count, in items rather than bytes
         * @param scratchBytes fixed exact-state capacity for one simultaneous range, in bytes
         */
        public record BatchGeometry(CpuSoftmaxIr ir, List<ValueId> boundaries,
                List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
                BatchByteSize iterationCount, BatchByteSize scratchBytes)
                implements CpuPartitionLowering.BatchUnitGeometry {
            /** Snapshots the immutable numeric recipe and its structural boundary inventory. */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
                Objects.requireNonNull(iterationCount, "iterationCount");
                Objects.requireNonNull(scratchBytes, "scratchBytes");
                if (boundaries.size() != bindings.size() || boundaries.size() != boundaryTypes.size())
                    throw new IllegalArgumentException("normalization boundary recipe disagrees");
            }
            /**
             * Declares that these normalization packs have no mutable ordinary cursor cells.
             * @return an immutable empty reset inventory
             */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset>
                    resets() { return List.of(); }
            /**
             * Returns deterministic coefficients and family identity, never resolved B.
             * @return immutable specialization recipe identity
             */
            @Override public String identity() {
                return ir.structuralKey() + ":normalization-batch-v1:" + bindings
                        + ":items=" + iterationCount + ":scratch=" + scratchBytes;
            }
            /**
             * Resolves checked numeric geometry without invoking semantic lowering.
             * @param batch non-negative invocation B, checked against access count/address limits
             * @return ordinary concrete family geometry with the identical compiled IR
             * @throws IllegalArgumentException if a positive-axis or training-domain rule fails
             * @throws ArithmeticException if resolved geometry exceeds supported limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var resolved = bindings.stream().map(binding -> binding.bind(batch)).toList();
                var layouts = resolved.stream().map(CpuSoftmaxLowering::boundLayout).toList();
                var input = layouts.getFirst(); var output = layouts.getLast();
                long width = input.extents[ir.axis()];
                if (width <= 0) throw new IllegalArgumentException("softmax width must be positive");
                long count = resolved.getFirst().elementCount();
                long slices = iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(), ir.dataType(), ir.axis(), input, output,
                        slices, width, count);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {slices}, slices,
                        "legal: symbolic stable softmax", new long[0], Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.of(geometry));
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

        /** Validates one complete static geometry. */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            if (axis < 0 || axis >= input.extents.length || sliceCount < 0 || sliceWidth <= 0
                    || elementCount < 0 || !Arrays.equals(input.extents, output.extents)
                    || input.extents[axis] != sliceWidth
                    || elementCount != Math.multiplyExact(sliceCount, sliceWidth))
                throw new IllegalArgumentException("softmax geometry is invalid");
        }
        /**
         * Packs bases, ranks, axis, counts, extents, and strides for the generated entry.
         * @param bases exact input/output carrier element bases
         * @return a new primitive geometry array consumed only by the generated entry
         * @throws IllegalArgumentException if there are not exactly two bases
         * @throws ArithmeticException if a base plus layout offset overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != 2) throw new IllegalArgumentException(
                    "softmax geometry requires two carrier bases");
            int rank = input.extents.length;
            long[] packed = new long[7 + 3 * rank];
            packed[0] = Math.addExact(bases[0], input.offset);
            packed[1] = Math.addExact(bases[1], output.offset);
            packed[2] = axis; packed[3] = sliceCount; packed[4] = sliceWidth;
            packed[5] = rank; packed[6] = elementCount;
            System.arraycopy(input.extents, 0, packed, 7, rank);
            System.arraycopy(input.strides, 0, packed, 7 + rank, rank);
            System.arraycopy(output.strides, 0, packed, 7 + 2 * rank, rank);
            return packed;
        }
    }
}
