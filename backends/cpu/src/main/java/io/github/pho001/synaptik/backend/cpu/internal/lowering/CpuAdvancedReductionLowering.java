package io.github.pho001.synaptik.backend.cpu.internal.lowering;

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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Lowers one static resolved advanced floating reduction to complete output-cell ranges. */
public final class CpuAdvancedReductionLowering {
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
                Optional.empty(), Optional.of(geometry));
    }

    private static long exactStateSliceBytes(DataType type, long count) {
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : 127;
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
        long count = elementCount(layout.extents); if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>(); long[] coordinates = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = layout.offset;
                for (int axis = 0; axis < coordinates.length; axis++) address = Math.addExact(
                        address, Math.multiplyExact(coordinates[axis], layout.strides[axis]));
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "advanced-reduction output layout must be injective");
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
                "advanced-reduction output layout must be injective");
            covered = Math.addExact(covered, Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis])); }
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
