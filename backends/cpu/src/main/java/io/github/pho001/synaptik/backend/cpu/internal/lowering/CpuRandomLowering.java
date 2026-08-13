package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
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

/** Lowers one exact fully static explicit-state RNG initializer or dropout occurrence. */
public final class CpuRandomLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless lowerer with no generator or mutable state. */
    public CpuRandomLowering() { }

    /**
     * Lowers one exact supported random occurrence with zero workspace.
     *
     * @param context non-null complete one-node CPU projection
     * @return immutable lowering with initializer output or dropout boundaries ordered as
     *     {@code [value,state,output,keepMask,nextState]}
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if the occurrence, descriptors, layouts, or boundaries are
     *     outside the exact supported matrix
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU random partition requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU random occurrence");

        var boundaryIds = new ArrayList<ValueId>();
        boundaryIds.addAll(node.inputs()); boundaryIds.addAll(node.outputs());
        if (new HashSet<>(boundaryIds).size() != boundaryIds.size()) throw new IllegalArgumentException(
                "random input and output value identities must be distinct");
        var layouts = boundaryIds.stream().map(id -> layout(require(values, id))).toList();
        int inputCount = node.inputs().size();
        for (int i = inputCount; i < layouts.size(); i++) validateInjective(layouts.get(i));
        if (node.operation().kind() == DropoutKind.DROPOUT) validateInjective(layouts.get(1));
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int i = 0; i < layouts.size(); i++) bindings.add(binding(layouts.get(i),
                i < inputCount ? CpuAccessPlan.AccessKind.READ : CpuAccessPlan.AccessKind.WRITE));

        CpuRandomIr ir;
        long elementCount;
        if (node.operation().kind() == GraphRngKind.INITIAL_STATE) {
            GraphRngStateAttrs attrs = (GraphRngStateAttrs) node.operation().attrs();
            ir = new CpuRandomIr(CpuRandomIr.Family.INITIAL_STATE, DataType.INT64,
                    attrs.key(), attrs.counter(), 0, List.of(bindings.getFirst().plan()));
            elementCount = 0;
        } else {
            DropoutAttrs attrs = (DropoutAttrs) node.operation().attrs();
            elementCount = elementCount(layouts.getFirst().extents());
            ir = new CpuRandomIr(CpuRandomIr.Family.DROPOUT,
                    require(values, node.inputs().getFirst()).descriptor().dataType(), 0, 0,
                    Double.doubleToRawLongBits(attrs.probability()),
                    bindings.stream().map(CpuAccessPlan.Binding::plan).toList());
        }
        var geometry = new Geometry(ir.family(), ir.valueType(), layouts, elementCount);
        var spans = boundaryIds.stream().map(id -> require(values, id).descriptor().layout()
                .orElseThrow().referencedElementSpan()).toList();
        var types = boundaryIds.stream().map(id -> require(values, id).descriptor().dataType()).toList();
        long[] domain = ir.family() == CpuRandomIr.Family.INITIAL_STATE
                ? new long[] {0} : layouts.getFirst().extents();
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), domain, elementCount,
                "legal: one fully static explicit-state CPU random occurrence", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(geometry));
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("random lowering requires non-negative resolved layouts");
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
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents); if (count == 0) return;
        var axes = new ArrayList<Integer>();
        for (int i = 0; i < layout.extents.length; i++) if (layout.extents[i] > 1) axes.add(i);
        axes.sort(Comparator.comparingLong(i -> layout.strides[i])); long covered = 1;
        for (int axis : axes) {
            if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                    "random output layout must be injective");
            covered = Math.addExact(covered,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
    }

    /**
     * Resolved cold boundary layout retained independently of a concrete carrier.
     *
     * @param extents non-null logical axis extents; copied defensively
     * @param offset non-negative carrier-relative base element offset
     * @param strides non-null non-negative element strides in axis order; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one resolved layout.
         *
         * @param extents non-null logical axis extents; copied defensively
         * @param offset non-negative carrier-relative base element offset
         * @param strides non-null non-negative element strides in axis order; copied defensively
         * @throws NullPointerException if {@code extents} or {@code strides} is null
         */
        public Layout { extents = extents.clone(); strides = strides.clone(); }
        /**
         * Returns a defensive snapshot of the logical extents.
         *
         * @return fresh logical-extents array
         */
        @Override public long[] extents() { return extents.clone(); }

        /**
         * Returns a defensive snapshot of the element strides.
         *
         * @return fresh element-strides array
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Cold invocation geometry for one initializer or dropout.
     *
     * @param family exact random family
     * @param valueType represented dropout type or INT64 initializer marker
     * @param boundaries ordered resolved layouts
     * @param elementCount exact logical dropout draw count, or zero for initialization
     */
    public record Geometry(CpuRandomIr.Family family, DataType valueType,
            List<Layout> boundaries, long elementCount) {
        /**
         * Validates and snapshots geometry.
         *
         * @param family non-null exact random family
         * @param valueType non-null represented dropout type or INT64 initializer marker
         * @param boundaries non-null ordered resolved layouts; copied defensively
         * @param elementCount non-negative logical dropout draw count, or zero for initialization
         * @throws NullPointerException if a required reference or boundary element is null
         * @throws IllegalArgumentException if the boundary count or element count disagrees with
         *     the family
         */
        public Geometry {
            Objects.requireNonNull(family, "family"); Objects.requireNonNull(valueType, "valueType");
            boundaries = List.copyOf(boundaries);
            int expected = family == CpuRandomIr.Family.INITIAL_STATE ? 1 : 5;
            if (boundaries.size() != expected || elementCount < 0)
                throw new IllegalArgumentException("random geometry facts disagree");
        }

        /**
         * Packs exact carrier bases and boundary layouts for generated execution.
         *
         * @param bases ordered carrier base element offsets
         * @return fresh packed primitive geometry
         * @throws NullPointerException if {@code bases} is null
         * @throws IllegalArgumentException if the base count differs from the boundary count
         * @throws ArithmeticException if adding a carrier base and layout offset overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != boundaries.size()) throw new IllegalArgumentException(
                    "random carrier bases must match boundary count");
            int size = 4;
            for (Layout layout : boundaries) size += 2 + 2 * layout.extents.length;
            long[] packed = new long[size];
            packed[0] = family.ordinal(); packed[1] = valueType.ordinal();
            packed[2] = elementCount; packed[3] = boundaries.size();
            int next = 4;
            for (int i = 0; i < boundaries.size(); i++) {
                Layout layout = boundaries.get(i); packed[next++] = layout.extents.length;
                packed[next++] = Math.addExact(bases[i], layout.offset);
                for (long extent : layout.extents) packed[next++] = extent;
                for (long stride : layout.strides) packed[next++] = stride;
            }
            return packed;
        }
    }
}
