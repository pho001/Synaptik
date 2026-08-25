package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Lowers one explicit static trailing Layer or RMS normalization to complete-slice ranges.
 *
 * <p>The lowerer admits only the first-class Model occurrence. It snapshots resolved non-negative
 * layouts, deduplicates repeated logical inputs in first-use order, derives checked leading and
 * normalized counts, and declares Layer-only exact-mean scratch geometry. It does not recognize
 * decomposed graphs or choose run-time carriers, worker ranges, or physical resources.</p>
 */
public final class CpuTrailingNormalizationLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless trailing-normalization lowerer. */
    public CpuTrailingNormalizationLowering() { }

    /**
     * Lowers one supported occurrence and derives all checked cold geometry.
     *
     * @param context non-null projection containing exactly one supported first-class occurrence
     * @return one immutable lowering whose ranges count complete leading slices; never
     *     {@code null}
     * @throws NullPointerException if {@code context} or a required projected fact is null
     * @throws IllegalArgumentException if occurrence cardinality, descriptors, layouts, output
     *     identity, promotion, or normalization geometry is unsupported or inconsistent
     * @throws ArithmeticException if Shape, span, address, or exact-state sizing overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU trailing normalization requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU trailing-normalization occurrence");
        ValueId outputId = node.outputs().getFirst();
        if (node.inputs().contains(outputId)) throw new IllegalArgumentException(
                "trailing-normalization output must be distinct from every input");

        CpuTrailingNormalizationIr.Kind kind;
        CpuTrailingNormalizationIr.Form form;
        io.github.pho001.synaptik.model.shape.Shape normalizedShape;
        ScalarValue epsilon;
        if (node.operation().attrs() instanceof LayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER;
            form = CpuTrailingNormalizationIr.Form.LAYER;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        } else if (node.operation().attrs() instanceof AffineLayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER;
            form = CpuTrailingNormalizationIr.Form.LAYER_AFFINE;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        } else {
            RmsNormAttrs attrs = (RmsNormAttrs) node.operation().attrs();
            kind = CpuTrailingNormalizationIr.Kind.RMS;
            form = node.inputs().size() == 1 ? CpuTrailingNormalizationIr.Form.RMS
                    : CpuTrailingNormalizationIr.Form.RMS_SCALED;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        }
        GraphValue input = require(values, node.inputs().getFirst());
        GraphValue output = require(values, outputId);
        Layout inputLayout = layout(input), outputLayout = layout(output);
        validateInjective(outputLayout);
        int normalizedRank = normalizedShape.rank();
        long normalizedCount = elementCount(normalizedShape.toLongArray());
        long leadingCount = prefixCount(inputLayout.extents, inputLayout.extents.length - normalizedRank);
        long outputCount = elementCount(outputLayout.extents);
        if (outputCount != Math.multiplyExact(leadingCount, normalizedCount))
            throw new IllegalArgumentException("trailing-normalization Shape counts disagree");

        var uniqueIds = new ArrayList<ValueId>();
        var positionMap = new ArrayList<Integer>();
        for (ValueId id : node.inputs()) {
            int index = uniqueIds.indexOf(id);
            if (index < 0) { index = uniqueIds.size(); uniqueIds.add(id); }
            positionMap.add(index);
        }
        var layouts = new ArrayList<Layout>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var types = new ArrayList<DataType>();
        for (ValueId id : uniqueIds) {
            GraphValue value = require(values, id); Layout layout = layout(value);
            layouts.add(layout); bindings.add(binding(layout, CpuAccessPlan.AccessKind.READ));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            types.add(value.descriptor().dataType());
        }
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        bindings.add(outputBinding);
        spans.add(output.descriptor().layout().orElseThrow().referencedElementSpan());
        types.add(output.descriptor().dataType());
        var boundaryIds = new ArrayList<>(uniqueIds); boundaryIds.add(outputId);
        DataType exactType = output.descriptor().dataType() == DataType.BFLOAT16
                ? DataType.FLOAT32 : output.descriptor().dataType();
        long scratch = kind == CpuTrailingNormalizationIr.Kind.LAYER && outputCount > 0
                ? exactStateSliceBytes(exactType, normalizedCount) : 0;
        int limbs = scratch == 0 ? 0 : Math.toIntExact(scratch / Long.BYTES - 1);
        var semanticTypes = node.inputs().stream().map(id -> require(values, id).descriptor().dataType())
                .toList();
        var inputPlans = bindings.subList(0, uniqueIds.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        var ir = new CpuTrailingNormalizationIr(kind, form, semanticTypes,
                output.descriptor().dataType(), epsilonBits(epsilon), normalizedRank, 1,
                kind == CpuTrailingNormalizationIr.Kind.LAYER ? 3 : 2, normalizedCount,
                limbs, scratch, positionMap, inputPlans, outputBinding.plan());
        var geometry = new Geometry(kind, form, semanticTypes, output.descriptor().dataType(),
                epsilonBits(epsilon), normalizedRank, normalizedCount, leadingCount,
                positionMap, layouts, outputLayout, scratch);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), new long[] {outputCount == 0 ? 0 : leadingCount},
                outputCount == 0 ? 0 : leadingCount,
                "legal: one static trailing normalization", new long[0], Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    private static long epsilonBits(ScalarValue epsilon) {
        return switch (epsilon.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(epsilon.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(epsilon.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> epsilon.bfloat16Bits() & 0xffffL;
            default -> throw new IllegalArgumentException("normalization epsilon must be floating");
        };
    }
    private static long exactStateSliceBytes(DataType type, long count) {
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : 127;
        int countBits = count <= 1 ? 0 : 64 - Long.numberOfLeadingZeros(count - 1);
        long bits = Math.addExact((long) emax + 1 - emin, Math.addExact(countBits, 1));
        return Math.addExact(8, Math.multiplyExact(8, Math.addExact(bits, 63) / 64));
    }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("trailing normalization requires non-negative layouts");
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
        long max = layout.offset;
        for (int axis = 0; axis < layout.extents.length; axis++) max = Math.addExact(max,
                Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        return Math.addExact(max, 1);
    }
    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }
    private static long prefixCount(long[] extents, int end) {
        long result = 1; for (int i = 0; i < end; i++) result = Math.multiplyExact(result, extents[i]);
        return result;
    }
    private static long suffixCount(long[] extents, int rank) {
        long result = 1;
        for (int axis = extents.length - rank; axis < extents.length; axis++)
            result = Math.multiplyExact(result, extents[axis]);
        return result;
    }
    private static boolean supported(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }
    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents); if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>(); long[] c = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = layout.offset;
                for (int axis = 0; axis < c.length; axis++) address = Math.addExact(address,
                        Math.multiplyExact(c[axis], layout.strides[axis]));
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "trailing-normalization output layout must be injective");
                for (int axis = c.length - 1; axis >= 0; axis--)
                    if (++c[axis] < layout.extents[axis]) break; else c[axis] = 0;
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < layout.extents.length; axis++) if (layout.extents[axis] > 1) axes.add(axis);
        axes.sort(Comparator.comparingLong(axis -> layout.strides[axis])); long covered = 1;
        for (int axis : axes) { if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                "trailing-normalization output layout must be injective");
            covered = Math.addExact(covered, Math.multiplyExact(layout.extents[axis] - 1,
                    layout.strides[axis])); }
    }

    /**
     * Resolved non-negative Shape and element-stride geometry.
     *
     * @param extents non-negative static extents; copied defensively
     * @param offset non-negative element offset from the carrier base
     * @param strides non-negative element strides matching {@code extents}; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Validates and snapshots one layout.
         *
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if ranks differ or any extent, offset, or stride is
         *     negative
         */
        public Layout { extents = extents.clone(); strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(v -> v < 0)
                    || Arrays.stream(strides).anyMatch(v -> v < 0))
                throw new IllegalArgumentException("trailing-normalization layout is invalid"); }
        /**
         * Returns a defensive copy of the resolved static extents.
         *
         * @return a new copy of the non-negative static extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns a defensive copy of the resolved element strides.
         *
         * @return a new copy of the non-negative element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete cold multi-boundary trailing-slice geometry retained by preparation and binding.
     *
     * @param kind exact Layer or RMS family
     * @param form exact semantic operand form
     * @param inputTypes semantic input types in occurrence order; copied defensively
     * @param resultType exact ordered-promotion result type
     * @param epsilonBits exact raw result-type epsilon bits
     * @param normalizedRank positive trailing normalized rank
     * @param normalizedCount non-negative elements per normalized slice
     * @param leadingCount non-negative number of complete leading slices
     * @param positionToBoundary semantic-position to unique-input-boundary mapping; copied
     *     defensively
     * @param inputs unique resolved input layouts in first-use order; copied defensively
     * @param output distinct resolved injective output layout
     * @param scratchSliceBytes exact bytes per Layer range, or zero for RMS and empty work
     */
    public record Geometry(CpuTrailingNormalizationIr.Kind kind,
            CpuTrailingNormalizationIr.Form form, List<DataType> inputTypes, DataType resultType,
            long epsilonBits, int normalizedRank, long normalizedCount, long leadingCount,
            List<Integer> positionToBoundary, List<Layout> inputs, Layout output,
            long scratchSliceBytes) {
        /**
         * Validates and snapshots one geometry.
         *
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if promotion, mapping, Shape, family, form, or workspace
         *     facts disagree
         * @throws ArithmeticException if exact element or workspace geometry overflows
         */
        public Geometry { inputTypes = List.copyOf(inputTypes);
            positionToBoundary = List.copyOf(positionToBoundary); inputs = List.copyOf(inputs);
            int expected = switch (form) { case LAYER, RMS -> 1; case RMS_SCALED -> 2;
                case LAYER_AFFINE -> 3; };
            boolean floatingResult = supported(resultType);
            int unique = positionToBoundary.stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
            DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
            for (int index = 1; index < inputTypes.size(); index++)
                promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
            if (normalizedRank <= 0 || normalizedCount < 0 || leadingCount < 0
                    || inputs.isEmpty() || output == null || scratchSliceBytes < 0
                    || inputTypes.size() != expected || positionToBoundary.size() != expected
                    || inputs.size() != unique || positionToBoundary.getFirst() != 0
                    || !floatingResult || inputTypes.stream().anyMatch(type -> !supported(type))
                    || promoted != resultType
                    || (kind == CpuTrailingNormalizationIr.Kind.LAYER)
                        != (form == CpuTrailingNormalizationIr.Form.LAYER
                            || form == CpuTrailingNormalizationIr.Form.LAYER_AFFINE)
                    || kind == CpuTrailingNormalizationIr.Kind.RMS && scratchSliceBytes != 0
                    || kind == CpuTrailingNormalizationIr.Kind.LAYER && normalizedCount > 0
                        && leadingCount > 0 && scratchSliceBytes == 0
                    || output.extents.length < normalizedRank
                    || suffixCount(output.extents, normalizedRank) != normalizedCount
                    || prefixCount(output.extents, output.extents.length - normalizedRank)
                        != leadingCount
                    || elementCount(output.extents) != Math.multiplyExact(
                        normalizedCount, leadingCount))
                throw new IllegalArgumentException("trailing-normalization geometry is invalid");
            int nextBoundary = 0;
            for (int position = 0; position < positionToBoundary.size(); position++) {
                int boundary = positionToBoundary.get(position);
                if (boundary < 0 || boundary >= inputs.size() || boundary > nextBoundary)
                    throw new IllegalArgumentException(
                            "trailing-normalization boundary map is invalid");
                if (boundary == nextBoundary) nextBoundary++;
                int first = positionToBoundary.indexOf(boundary);
                if (inputTypes.get(position) != inputTypes.get(first))
                    throw new IllegalArgumentException(
                            "repeated trailing-normalization boundary type disagrees");
            }
            for (int boundary = 0; boundary < inputs.size(); boundary++) {
                Layout layout = inputs.get(boundary);
                int position = positionToBoundary.indexOf(boundary);
                if (position == 0) {
                    if (!Arrays.equals(layout.extents, output.extents))
                        throw new IllegalArgumentException(
                                "trailing-normalization input Shape disagrees");
                } else {
                    if (layout.extents.length != normalizedRank
                            || !Arrays.equals(layout.extents, Arrays.copyOfRange(output.extents,
                                output.extents.length - normalizedRank, output.extents.length)))
                        throw new IllegalArgumentException(
                                "trailing-normalization parameter Shape disagrees");
                }
            }
            validateInjective(output);
        }
        /**
         * Returns exact workspace bytes for the stated simultaneous complete-slice ranges.
         *
         * @param ranges non-negative number of simultaneous ranges
         * @return exact Layer workspace bytes, or zero for RMS and empty work
         * @throws ArithmeticException if multiplication overflows
         */
        public long workspaceBytes(int ranges) { return Math.multiplyExact(scratchSliceBytes, ranges); }
        /**
         * Packs carrier-relative bases and resolved layouts for one direct generated entry.
         *
         * @param bases carrier-relative element bases for every unique input followed by output;
         *     not retained or mutated
         * @return a new primitive geometry array owned by the caller
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if the base count does not match the boundaries
         * @throws ArithmeticException if applying a resolved layout offset overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != inputs.size() + 1) throw new IllegalArgumentException(
                    "trailing-normalization carrier base count disagrees");
            int size = 11 + bases.length;
            for (Layout layout : inputs) size += 1 + layout.extents.length * 2;
            size += 1 + output.extents.length * 2;
            long[] packed = new long[size]; int p = 0;
            packed[p++] = inputs.size(); packed[p++] = normalizedRank;
            packed[p++] = normalizedCount; packed[p++] = leadingCount;
            packed[p++] = output.extents.length; packed[p++] = scratchSliceBytes;
            packed[p++] = 0; packed[p++] = normalizedCount; packed[p++] = 0;
            packed[p++] = scratchSliceBytes; packed[p++] = 0;
            for (int i = 0; i < inputs.size(); i++) packed[p++] = Math.addExact(bases[i], inputs.get(i).offset);
            packed[p++] = Math.addExact(bases[bases.length - 1], output.offset);
            for (Layout layout : inputs) { packed[p++] = layout.extents.length;
                for (long v : layout.extents) packed[p++] = v;
                for (long v : layout.strides) packed[p++] = v; }
            packed[p++] = output.extents.length;
            for (long v : output.extents) packed[p++] = v;
            for (long v : output.strides) packed[p++] = v;
            return packed;
        }
    }
}
