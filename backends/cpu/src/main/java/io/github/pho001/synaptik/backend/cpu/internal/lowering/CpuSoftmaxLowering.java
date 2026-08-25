package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
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

/** Lowers one static resolved softmax occurrence to complete normalization-slice ranges. */
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
                input.descriptor().dataType(), axis, 1, 3,
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
