package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
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

/**
 * Lowers one explicit static batch-normalization inference occurrence.
 *
 * <p>The lowerer derives arbitrary-axis prefix/channel/suffix geometry, snapshots exact resolved
 * layouts, and deduplicates repeated logical inputs in first-use order. It declares no workspace,
 * materialization, denominator table, saved statistic, or training state.</p>
 */
public final class CpuBatchNormInferenceLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless inference lowerer. */
    public CpuBatchNormInferenceLowering() { }

    /**
     * Lowers one supported five-input/one-output occurrence.
     *
     * @param context non-null one-node CPU preparation projection
     * @return immutable lowering with checked geometry and unique boundaries
     * @throws NullPointerException if {@code context} or a required fact is null
     * @throws IllegalArgumentException if the occurrence, descriptors, layouts, or geometry
     *     disagree with the implemented static inference subset
     * @throws ArithmeticException if count, span, or address arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) {
            throw new IllegalArgumentException(
                    "CPU batch-normalization inference requires exactly one node");
        }
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)
                || node.operation().kind() != BatchNormKind.BATCH_NORM_INFERENCE
                || !(node.operation().attrs() instanceof BatchNormInferenceAttrs attrs)
                || node.inputs().size() != 5 || node.outputs().size() != 1) {
            throw new IllegalArgumentException(
                    "partition contains an unsupported CPU batch-normalization occurrence");
        }
        ValueId outputId = node.outputs().getFirst();
        if (node.inputs().contains(outputId)) {
            throw new IllegalArgumentException(
                    "batch-normalization output must be distinct from every input");
        }
        GraphValue input = require(values, node.inputs().getFirst());
        GraphValue output = require(values, outputId);
        Layout inputLayout = layout(input);
        Layout outputLayout = layout(output);
        validateInjective(outputLayout);
        int axis = attrs.channelAxis();
        long prefix = product(inputLayout.extents, 0, axis);
        long channels = inputLayout.extents[axis];
        long suffix = product(inputLayout.extents, axis + 1, inputLayout.extents.length);
        long nonChannel = Math.multiplyExact(prefix, suffix);
        long outputCount = Math.multiplyExact(channels, nonChannel);
        if (elementCount(outputLayout.extents) != outputCount) {
            throw new IllegalArgumentException("batch-normalization Shape counts disagree");
        }

        var uniqueIds = new ArrayList<ValueId>();
        var map = new ArrayList<Integer>();
        for (ValueId id : node.inputs()) {
            int boundary = uniqueIds.indexOf(id);
            if (boundary < 0) { boundary = uniqueIds.size(); uniqueIds.add(id); }
            map.add(boundary);
        }
        var layouts = new ArrayList<Layout>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var boundaryTypes = new ArrayList<DataType>();
        for (ValueId id : uniqueIds) {
            GraphValue value = require(values, id);
            Layout valueLayout = layout(value);
            layouts.add(valueLayout);
            bindings.add(binding(valueLayout, CpuAccessPlan.AccessKind.READ));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            boundaryTypes.add(value.descriptor().dataType());
        }
        CpuAccessPlan.Binding outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        bindings.add(outputBinding);
        spans.add(output.descriptor().layout().orElseThrow().referencedElementSpan());
        boundaryTypes.add(output.descriptor().dataType());
        var boundaryIds = new ArrayList<>(uniqueIds); boundaryIds.add(outputId);
        List<DataType> semanticTypes = node.inputs().stream()
                .map(id -> require(values, id).descriptor().dataType()).toList();
        var ir = new CpuBatchNormInferenceIr(semanticTypes, output.descriptor().dataType(),
                epsilonBits(attrs.epsilon()), inputLayout.extents.length, axis, 1,
                CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE, map,
                bindings.subList(0, uniqueIds.size()).stream().map(CpuAccessPlan.Binding::plan)
                        .toList(), outputBinding.plan());
        var geometry = new Geometry(semanticTypes, output.descriptor().dataType(),
                epsilonBits(attrs.epsilon()), axis, prefix, channels, suffix, nonChannel,
                outputCount, CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE, map, layouts,
                outputLayout);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans,
                boundaryTypes, List.of(), new long[] {channels}, outputCount == 0 ? 0 : channels,
                "legal: one static batch-normalization inference", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(geometry), Optional.empty());
    }

    private static long epsilonBits(ScalarValue epsilon) {
        return switch (epsilon.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(epsilon.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(epsilon.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> epsilon.bfloat16Bits() & 0xffffL;
            default -> throw new IllegalArgumentException("batch-normalization epsilon is not floating");
        };
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0)) {
            throw new IllegalArgumentException(
                    "batch-normalization inference requires non-negative layouts");
        }
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
        long maximum = layout.offset;
        for (int axis = 0; axis < layout.extents.length; axis++) {
            maximum = Math.addExact(maximum,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
        return Math.addExact(maximum, 1);
    }

    private static long product(long[] extents, int begin, int end) {
        long result = 1;
        for (int axis = begin; axis < end; axis++) result = Math.multiplyExact(result, extents[axis]);
        return result;
    }

    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        return product(extents, 0, extents.length);
    }

    private static boolean supported(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }

    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents); if (count == 0) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>();
            long[] coordinates = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = layout.offset;
                for (int axis = 0; axis < coordinates.length; axis++) {
                    address = Math.addExact(address,
                            Math.multiplyExact(coordinates[axis], layout.strides[axis]));
                }
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "batch-normalization output layout must be injective");
                for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                    if (++coordinates[axis] < layout.extents[axis]) break;
                    coordinates[axis] = 0;
                }
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < layout.extents.length; axis++) {
            if (layout.extents[axis] > 1) axes.add(axis);
        }
        axes.sort(Comparator.comparingLong(axis -> layout.strides[axis]));
        long covered = 1;
        for (int axis : axes) {
            if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                    "batch-normalization output layout must be injective");
            covered = Math.addExact(covered,
                    Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        }
    }

    /**
     * Resolved static non-negative layout in element units.
     *
     * @param extents non-null resolved extents; snapshotted defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-null non-negative element strides matching {@code extents}; snapshotted
     *     defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Validates and snapshots one layout.
         *
         * @throws NullPointerException if {@code extents} or {@code strides} is {@code null}
         * @throws IllegalArgumentException if rank, extent, offset, or stride facts are invalid
         */
        public Layout {
            extents = extents.clone(); strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(value -> value < 0)
                    || Arrays.stream(strides).anyMatch(value -> value < 0)) {
                throw new IllegalArgumentException("batch-normalization layout is invalid");
            }
        }
        /**
         * Returns the resolved extents without exposing retained storage.
         *
         * @return a new defensive copy of the resolved extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns the element strides without exposing retained storage.
         *
         * @return a new defensive copy of the element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete cold arbitrary-axis geometry for one direct generated inference entry.
     *
     * @param inputTypes five semantic input types in occurrence order
     * @param resultType ordered-promotion result and exact computation type
     * @param epsilonBits raw result-type epsilon bits
     * @param channelAxis normalized channel axis
     * @param prefixCount product of dimensions before the channel axis
     * @param channelCount channel extent
     * @param suffixCount product of dimensions after the channel axis
     * @param nonChannelCount product of prefix and suffix counts
     * @param outputCount product of channel and non-channel counts
     * @param rangeForm preparation-selected range ownership form
     * @param positionToBoundary five-position map to unique input boundaries
     * @param inputs unique input layouts in first-occurrence order
     * @param output injective output layout
     */
    public record Geometry(List<DataType> inputTypes, DataType resultType, long epsilonBits,
            int channelAxis, long prefixCount, long channelCount, long suffixCount,
            long nonChannelCount, long outputCount, CpuBatchNormInferenceIr.RangeForm rangeForm,
            List<Integer> positionToBoundary, List<Layout> inputs, Layout output) {
        /**
         * Validates and snapshots one geometry.
         *
         * @throws NullPointerException if a required component or list element is {@code null}
         * @throws IllegalArgumentException if types, promotion, Shapes, layout, counts, axis, or
         *     first-occurrence mapping disagree
         * @throws ArithmeticException if exact geometry validation overflows
         */
        public Geometry {
            inputTypes = List.copyOf(inputTypes);
            Objects.requireNonNull(resultType, "resultType");
            Objects.requireNonNull(rangeForm, "rangeForm");
            positionToBoundary = List.copyOf(positionToBoundary);
            inputs = List.copyOf(inputs);
            Objects.requireNonNull(output, "output");
            DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
            for (int index = 1; index < inputTypes.size(); index++) {
                promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
            }
            int unique = positionToBoundary.stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
            if (inputTypes.size() != 5 || positionToBoundary.size() != 5 || inputs.size() != unique
                    || !supported(resultType) || promoted != resultType
                    || inputTypes.stream().anyMatch(type -> !supported(type))
                    || channelAxis < 0 || output.extents.length < 2
                    || channelAxis >= output.extents.length || prefixCount < 0 || channelCount < 0
                    || suffixCount < 0 || nonChannelCount < 0 || outputCount < 0
                    || channelCount != output.extents[channelAxis]
                    || prefixCount != product(output.extents, 0, channelAxis)
                    || suffixCount != product(output.extents, channelAxis + 1, output.extents.length)
                    || nonChannelCount != Math.multiplyExact(prefixCount, suffixCount)
                    || outputCount != Math.multiplyExact(channelCount, nonChannelCount)
                    || elementCount(output.extents) != outputCount) {
                throw new IllegalArgumentException("batch-normalization geometry is invalid");
            }
            int next = 0;
            for (int position = 0; position < 5; position++) {
                int boundary = positionToBoundary.get(position);
                if (boundary < 0 || boundary >= inputs.size() || boundary > next) {
                    throw new IllegalArgumentException("batch-normalization boundary map is invalid");
                }
                if (boundary == next) next++;
                int first = positionToBoundary.indexOf(boundary);
                if (inputTypes.get(position) != inputTypes.get(first)) {
                    throw new IllegalArgumentException(
                            "repeated batch-normalization boundary type disagrees");
                }
                Layout layout = inputs.get(boundary);
                if (position == 0 ? !Arrays.equals(layout.extents, output.extents)
                        : layout.extents.length != 1 || layout.extents[0] != channelCount) {
                    throw new IllegalArgumentException("batch-normalization input Shape disagrees");
                }
            }
            validateInjective(output);
        }

        /**
         * Returns the same geometry with the selected form.
         *
         * @param selected non-null range ownership form
         * @return this geometry when unchanged, otherwise a new validated immutable geometry
         * @throws NullPointerException if {@code selected} is {@code null}
        */
        public Geometry withRangeForm(CpuBatchNormInferenceIr.RangeForm selected) {
            return selected == rangeForm ? this : new Geometry(inputTypes, resultType, epsilonBits,
                    channelAxis, prefixCount, channelCount, suffixCount, nonChannelCount,
                    outputCount, selected, positionToBoundary, inputs, output);
        }

        /**
         * Returns the size of the selected execution domain.
         *
         * @return the selected execution-domain item count, or zero for empty output
         */
        public long rangeItemCount() {
            if (outputCount == 0) return 0;
            return rangeForm == CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE
                    ? channelCount : nonChannelCount;
        }

        /**
         * Returns the amount of coordinate work owned by one range item.
         *
         * @return output coordinates covered by one selected range item
         */
        public long coordinatesPerRangeItem() {
            return rangeForm == CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE
                    ? nonChannelCount : channelCount;
        }

        /**
         * Packs carrier-relative bases and complete resolved layouts for the generated entry.
         *
         * @param bases non-null element bases for unique inputs followed by output
         * @return a newly allocated packed geometry array owned by the caller
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if the base count disagrees with the boundaries
         * @throws ArithmeticException if adding a layout offset to a carrier base overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != inputs.size() + 1) throw new IllegalArgumentException(
                    "batch-normalization carrier base count disagrees");
            int size = 9 + bases.length;
            for (Layout layout : inputs) size += 1 + 2 * layout.extents.length;
            size += 1 + 2 * output.extents.length;
            long[] packed = new long[size]; int p = 0;
            packed[p++] = inputs.size(); packed[p++] = output.extents.length;
            packed[p++] = channelAxis; packed[p++] = prefixCount; packed[p++] = channelCount;
            packed[p++] = suffixCount; packed[p++] = nonChannelCount; packed[p++] = outputCount;
            packed[p++] = rangeForm.ordinal();
            for (int index = 0; index < inputs.size(); index++) {
                packed[p++] = Math.addExact(bases[index], inputs.get(index).offset);
            }
            packed[p++] = Math.addExact(bases[bases.length - 1], output.offset);
            for (Layout layout : inputs) {
                packed[p++] = layout.extents.length;
                for (long extent : layout.extents) packed[p++] = extent;
                for (long stride : layout.strides) packed[p++] = stride;
            }
            packed[p++] = output.extents.length;
            for (long extent : output.extents) packed[p++] = extent;
            for (long stride : output.strides) packed[p++] = stride;
            return packed;
        }
    }
}
