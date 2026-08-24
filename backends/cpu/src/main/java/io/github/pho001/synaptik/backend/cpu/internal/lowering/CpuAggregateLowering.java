package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
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
 * Lowers one fully static supported ordinary aggregate or binding-aware SUM-to-Shape occurrence.
 *
 * <p>The lowerer canonicalizes selected-axis membership, derives complete output-cell and
 * selected-domain geometry, and declares exactly one input and one distinct injective output.
 * SUM-to-Shape derives leading and unequal target-one axes from exact right alignment while
 * preserving equal aligned extents before considering target-one reduction.
 * Floating SUM, MEAN, and PROD derive exact fixed-width state; other rows select no workspace.
 * No row creates a partial reduction or combine state.</p>
 */
public final class CpuAggregateLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless ordinary-aggregate lowerer. */
    public CpuAggregateLowering() { }

    /**
     * Lowers one supported occurrence into a two-boundary output-cell unit. A SUM-to-Shape
     * occurrence must already have fully static source, target, and result Shapes; this method
     * validates their exact right-aligned relationship and performs no runtime binding.
     * @param context non-null complete one-node CPU projection, borrowed for this cold call
     * @return immutable aggregate lowering with exact output-cell and optional state geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence, descriptors, layouts, axes, or output
     *     injectivity are outside the exact supported matrix
     * @throws ArithmeticException if checked geometry or referenced-span arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU aggregate partition requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU aggregate occurrence");
        ValueId inputId = node.inputs().getFirst(), outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) throw new IllegalArgumentException(
                "aggregate output must be distinct from its input");
        GraphValue input = require(values, inputId), output = require(values, outputId);
        Layout inputLayout = layout(input), outputLayout = layout(output);
        validateInjective(outputLayout);

        Object attrs = node.operation().attrs();
        CpuAggregateIr.Form form; boolean keep; int[] selected;
        if (attrs instanceof SumToShapeAttrs sumTo) {
            form = CpuAggregateIr.Form.SUM_TO_SHAPE; keep = false;
            long[] source = inputLayout.extents, target = sumTo.targetShape().toLongArray();
            if (!Arrays.equals(target, outputLayout.extents) || target.length > source.length)
                throw new IllegalArgumentException("sum-to-Shape target and output must agree");
            int leading = source.length - target.length;
            var axes = new ArrayList<Integer>();
            for (int axis = 0; axis < leading; axis++) axes.add(axis);
            for (int targetAxis = 0; targetAxis < target.length; targetAxis++) {
                int inputAxis = leading + targetAxis;
                if (source[inputAxis] == target[targetAxis]) continue;
                if (target[targetAxis] != 1) throw new IllegalArgumentException(
                        "sum-to-Shape aligned extents must be equal or target one");
                axes.add(inputAxis);
            }
            selected = axes.stream().mapToInt(Integer::intValue).toArray();
        } else if (attrs == NoOperationAttrs.INSTANCE) {
            form = CpuAggregateIr.Form.FULL; keep = false;
            selected = new int[inputLayout.extents.length];
            for (int axis = 0; axis < selected.length; axis++) selected[axis] = axis;
        } else if (attrs instanceof AxisReductionAttrs axis) {
            form = CpuAggregateIr.Form.SINGLE_AXIS; keep = axis.keepDimensions();
            selected = new int[] {axis.axis()};
        } else {
            var axes = (MultiAxisReductionAttrs) attrs;
            form = CpuAggregateIr.Form.MULTI_AXIS; keep = axes.keepDimensions();
            selected = axes.axes().stream().mapToInt(Integer::intValue).sorted().toArray();
        }
        boolean[] membership = new boolean[inputLayout.extents.length];
        for (int axis : selected) {
            if (axis < 0 || axis >= membership.length || membership[axis])
                throw new IllegalArgumentException("aggregate axes must be normalized and distinct");
            membership[axis] = true;
        }
        long[] expected = form == CpuAggregateIr.Form.SUM_TO_SHAPE
                ? outputLayout.extents : outputExtents(inputLayout.extents, membership, keep, form);
        if (!Arrays.equals(expected, outputLayout.extents))
            throw new IllegalArgumentException("aggregate output Shape disagrees with Model semantics");
        long outputCount = elementCount(outputLayout.extents);
        long domainCount = 1;
        for (int axis : selected) if (inputLayout.extents[axis] == 0) { domainCount = 0; break; }
        if (domainCount != 0) for (int axis : selected)
            domainCount = Math.multiplyExact(domainCount, inputLayout.extents[axis]);
        var inputBinding = binding(inputLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        AggregateReductionKind modelKind = (AggregateReductionKind) node.operation().kind();
        CpuAggregateIr.Kind kind = CpuAggregateIr.Kind.valueOf(modelKind.name());
        DataType type = input.descriptor().dataType();
        boolean actualReduction = form != CpuAggregateIr.Form.SUM_TO_SHAPE || selected.length != 0;
        boolean exactFloating = actualReduction && (kind == CpuAggregateIr.Kind.SUM
                || kind == CpuAggregateIr.Kind.MEAN || kind == CpuAggregateIr.Kind.PROD)
                && (type == DataType.FLOAT64 || type == DataType.FLOAT32
                    || type == DataType.BFLOAT16);
        int stateLimbCount = exactFloating ? stateLimbCount(kind, type, domainCount) : 0;
        long scratchSliceBytes = exactFloating ? Math.addExact(
                kind == CpuAggregateIr.Kind.PROD ? 24L : 8L,
                Math.multiplyExact(8L, stateLimbCount)) : 0;
        var ir = new CpuAggregateIr(kind, type, form, selected, keep, inputBinding.plan(),
                outputBinding.plan(), form == CpuAggregateIr.Form.SUM_TO_SHAPE
                    ? inputLayout.extents : new long[0],
                form == CpuAggregateIr.Form.SUM_TO_SHAPE
                    ? outputLayout.extents : new long[0],
                domainCount, stateLimbCount, scratchSliceBytes,
                CpuAggregateIr.FIRST_LOGICAL_NAN_AND_SIGNED_ZERO,
                CpuAggregateIr.COMPLETE_OUTPUT_CELLS, exactFloating
                    ? CpuAggregateIr.EXACT_FLOATING_STATE : CpuAggregateIr.ZERO_WORKSPACE);
        var geometry = new Geometry(kind, type, form, selected, keep, inputLayout, outputLayout,
                outputCount, domainCount, stateLimbCount, scratchSliceBytes);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding), List.of(
                    input.descriptor().layout().orElseThrow().referencedElementSpan(),
                    output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(type, type), List.of(), new long[] {outputCount}, outputCount,
                "legal: one fully static ordinary extrema or Boolean reduction", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    private static int stateLimbCount(CpuAggregateIr.Kind kind, DataType type, long domainCount) {
        int precision = type == DataType.FLOAT64 ? 53 : type == DataType.FLOAT32 ? 24 : 8;
        if (kind == CpuAggregateIr.Kind.PROD) {
            int maximumExponentMagnitude = type == DataType.FLOAT64 ? 1074
                    : type == DataType.FLOAT32 ? 149 : 133;
            Math.multiplyExact((long) maximumExponentMagnitude, Math.max(1L, domainCount));
            long bits = Math.multiplyExact((long) precision, Math.max(1L, domainCount));
            return Math.toIntExact(Math.max(1L, Math.addExact(bits, 63L) / 64L));
        }
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : 127;
        int cardinalityBits = domainCount <= 1 ? 0
                : 64 - Long.numberOfLeadingZeros(domainCount - 1);
        long signedBits = Math.addExact((long) emax + 1L - emin,
                Math.addExact(cardinalityBits, 1L));
        return Math.toIntExact(Math.addExact(signedBits, 63L) / 64L);
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("aggregate requires non-negative resolved layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(), source.strides());
    }
    private static long[] outputExtents(long[] input, boolean[] selected, boolean keep,
            CpuAggregateIr.Form form) {
        if (form == CpuAggregateIr.Form.FULL) return new long[0];
        var result = new ArrayList<Long>();
        for (int axis = 0; axis < input.length; axis++) {
            if (!selected[axis]) result.add(input[axis]); else if (keep) result.add(1L);
        }
        return result.stream().mapToLong(Long::longValue).toArray();
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
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }
    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents);
        if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>(); long[] coordinates = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = 0;
                for (int axis = 0; axis < coordinates.length; axis++) address = Math.addExact(address,
                        Math.multiplyExact(coordinates[axis], layout.strides[axis]));
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "aggregate output layout must be injective");
                increment(coordinates, layout.extents);
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int i = 0; i < layout.extents.length; i++) if (layout.extents[i] > 1) axes.add(i);
        axes.sort(java.util.Comparator.comparingLong(i -> layout.strides[i])); long covered = 1;
        for (int axis : axes) {
            if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                    "aggregate output layout must be injective");
            covered = Math.addExact(covered,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
    }
    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (++coordinates[axis] < extents[axis]) break; coordinates[axis] = 0;
        }
    }

    /**
     * Resolved non-negative layout used only as immutable cold aggregate geometry.
     * @param extents fully static logical extents; copied defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-negative element strides; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
    /**
     * Validates and snapshots one layout.
     *
     * @param extents fully static non-negative logical extents; not {@code null} and copied
     *     defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-negative element strides with the same rank as {@code extents}; not
     *     {@code null} and copied defensively
     * @throws NullPointerException if an array is {@code null}
     * @throws IllegalArgumentException if rank, extent, offset, or stride facts are invalid
         */
        public Layout {
            extents = extents.clone(); strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(value -> value < 0)
                    || Arrays.stream(strides).anyMatch(value -> value < 0))
                throw new IllegalArgumentException("aggregate layout facts disagree");
        }
        /**
         * Returns the logical Shape without exposing retained layout state.
         *
         * @return a new array containing the fully static logical extents; never {@code null}
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns carrier-relative addressing strides without exposing retained layout state.
         *
         * @return a new array containing carrier-relative element strides; never {@code null}
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Immutable cold geometry for complete output-cell reductions.
     * @param kind non-null aggregate kind matching the IR
     * @param dataType non-null represented input/output type
     * @param form non-null ordinary attribute form
     * @param selectedAxes increasing normalized membership; copied defensively
     * @param keepDimensions whether selected axes remain with extent one
     * @param input non-null resolved input layout
     * @param output non-null resolved output layout
     * @param outputCount non-negative number of independent output cells
     * @param domainCount non-negative number of values in every selected domain
     * @param stateLimbCount exact floating numerical limb count, otherwise zero
     * @param scratchSliceBytes exact bytes per floating numerical worker slice, otherwise zero
     */
    public record Geometry(CpuAggregateIr.Kind kind, DataType dataType, CpuAggregateIr.Form form,
            int[] selectedAxes, boolean keepDimensions, Layout input, Layout output,
            long outputCount, long domainCount, int stateLimbCount, long scratchSliceBytes) {
        /**
         * Validates and snapshots the cold geometry.
         *
         * @param kind aggregate meaning matching the canonical IR; not {@code null}
         * @param dataType represented input/output type matching the canonical IR; not
         *     {@code null}
         * @param form exact ordinary attribute form; not {@code null}
         * @param selectedAxes increasing normalized selected-axis membership; not {@code null}
         *     and copied defensively
         * @param keepDimensions whether selected axes remain as extent-one output dimensions
         * @param input resolved non-negative input layout; not {@code null}
         * @param output resolved injective output layout; not {@code null}
         * @param outputCount number of independent output cells; non-negative
         * @param domainCount number of represented input values per selected domain;
         *     non-negative, with one for an empty multi-axis selection
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if counts, axes, or form disagree
         * @throws ArithmeticException if checked domain or output geometry overflows
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(form, "form"); Objects.requireNonNull(selectedAxes, "selectedAxes");
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            selectedAxes = selectedAxes.clone();
            boolean[] membership = new boolean[input.extents.length];
            for (int index = 0; index < selectedAxes.length; index++) {
                int axis = selectedAxes[index];
                if (axis < 0 || axis >= membership.length || membership[axis]
                        || index > 0 && selectedAxes[index - 1] >= axis)
                    throw new IllegalArgumentException("aggregate geometry axes disagree");
                membership[axis] = true;
            }
            long expectedDomain = 1;
            for (int axis : selectedAxes) if (input.extents[axis] == 0) { expectedDomain = 0; break; }
            if (expectedDomain != 0) for (int axis : selectedAxes)
                expectedDomain = Math.multiplyExact(expectedDomain, input.extents[axis]);
            long[] expectedOutput = form == CpuAggregateIr.Form.SUM_TO_SHAPE
                    ? sumToShapeOutput(input.extents, output.extents, membership)
                    : outputExtents(input.extents, membership, keepDimensions, form);
            boolean actualReduction = form != CpuAggregateIr.Form.SUM_TO_SHAPE
                    || selectedAxes.length != 0;
            boolean exactFloating = actualReduction && (kind == CpuAggregateIr.Kind.SUM
                    || kind == CpuAggregateIr.Kind.MEAN || kind == CpuAggregateIr.Kind.PROD)
                    && (dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                        || dataType == DataType.BFLOAT16);
            int expectedLimbs = exactFloating ? CpuAggregateLowering.stateLimbCount(
                    kind, dataType, domainCount) : 0;
            long expectedSlice = exactFloating ? Math.addExact(
                    kind == CpuAggregateIr.Kind.PROD ? 24L : 8L,
                    Math.multiplyExact(8L, expectedLimbs)) : 0;
            if (!Arrays.equals(expectedOutput, output.extents)
                    || outputCount != elementCount(output.extents) || domainCount != expectedDomain
                    || stateLimbCount != expectedLimbs || scratchSliceBytes != expectedSlice)
                throw new IllegalArgumentException("aggregate geometry counts or Shapes disagree");
        }

        private static long[] sumToShapeOutput(long[] input, long[] output,
                boolean[] selected) {
            if (output.length > input.length) throw new IllegalArgumentException(
                    "sum-to-Shape target rank exceeds source rank");
            int leading = input.length - output.length;
            for (int axis = 0; axis < input.length; axis++) {
                boolean expectedSelected;
                if (axis < leading) expectedSelected = true;
                else {
                    long sourceExtent = input[axis], targetExtent = output[axis - leading];
                    if (sourceExtent == targetExtent) expectedSelected = false;
                    else if (targetExtent == 1) expectedSelected = true;
                    else throw new IllegalArgumentException(
                            "sum-to-Shape aligned extents must be equal or target one");
                }
                if (selected[axis] != expectedSelected) throw new IllegalArgumentException(
                        "sum-to-Shape selected axes disagree with alignment");
            }
            return output.clone();
        }

        /**
         * Returns the exact run-owned workspace needed by the simultaneously selected ranges.
         * Empty outputs require no state even though their floating numerical code shape retains
         * a nonzero slice size.
         *
         * @param rangeCount positive maximum number of simultaneously executing worker ranges
         * @return checked byte count for one slice per used range, never negative
         * @throws IllegalArgumentException if {@code rangeCount} is not positive
         * @throws ArithmeticException if the exact byte count overflows {@code long}
         */
        public long workspaceBytes(int rangeCount) {
            if (rangeCount <= 0) throw new IllegalArgumentException("rangeCount must be positive");
            return Math.multiplyExact(scratchSliceBytes,
                    Math.min((long) rangeCount, outputCount));
        }
        /**
         * Returns canonical selected-axis membership without exposing retained geometry state.
         *
         * @return a new array containing increasing normalized selected-axis membership; never
         *     {@code null}
         */
        @Override public int[] selectedAxes() { return selectedAxes.clone(); }
        /**
         * Packs direct bases and invocation-private coordinate state.
         * @param bases exact input and output element bases; read but not retained
         * @param rangeIndex non-negative worker-range ordinal used to derive its scratch slice
         * @return new mutable invocation-owned primitive geometry
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if exactly two bases are not supplied
         * @throws ArithmeticException if a base plus offset overflows
         */
        public long[] pack(long[] bases, int rangeIndex) {
            Objects.requireNonNull(bases, "bases");
            if (bases.length != 2 || rangeIndex < 0)
                throw new IllegalArgumentException("aggregate requires two bases and a range index");
            int inRank = input.extents.length, outRank = output.extents.length;
            long[] packed = new long[11 + inRank + inRank + outRank
                    + 2 + 2 * inRank + 2 + 2 * outRank];
            packed[0] = kind.ordinal(); packed[1] = dataType.ordinal(); packed[2] = form.ordinal();
            packed[3] = keepDimensions ? 1 : 0; packed[4] = inRank; packed[5] = outRank;
            packed[6] = outputCount; packed[7] = domainCount;
            packed[8] = stateLimbCount; packed[9] = scratchSliceBytes;
            packed[10] = Math.multiplyExact(rangeIndex, scratchSliceBytes);
            for (int axis : selectedAxes) packed[11 + axis] = 1;
            int x = 11 + inRank + inRank + outRank;
            x = packLayout(packed, x, input, bases[0]); packLayout(packed, x, output, bases[1]);
            return packed;
        }
        /**
         * Packs direct boundary bases for the first selected worker range.
         *
         * @param bases exact input and output element bases; read but not retained
         * @return new mutable invocation-owned primitive geometry with scratch offset zero
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if exactly two bases are not supplied
         * @throws ArithmeticException if a base plus layout offset overflows
         */
        public long[] pack(long[] bases) { return pack(bases, 0); }
        private static int packLayout(long[] target, int x, Layout layout, long base) {
            target[x++] = layout.extents.length; target[x++] = Math.addExact(base, layout.offset);
            for (long extent : layout.extents) target[x++] = extent;
            for (long stride : layout.strides) target[x++] = stride;
            return x;
        }
    }
}
