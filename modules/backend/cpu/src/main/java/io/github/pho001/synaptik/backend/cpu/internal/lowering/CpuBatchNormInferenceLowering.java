package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
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
 * <p>The single-B entry retains arbitrary channel, prefix, and suffix axes symbolically;
 * non-channel range ownership preserves exact empty-B behavior without sampling.</p>
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
        long epsilonBits = scalarBits(output.descriptor().dataType(), attrs.epsilon());
        var ir = new CpuBatchNormInferenceIr(semanticTypes, output.descriptor().dataType(),
                epsilonBits, inputLayout.extents.length, axis, 1,
                CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE, map,
                bindings.subList(0, uniqueIds.size()).stream().map(CpuAccessPlan.Binding::plan)
                        .toList(), outputBinding.plan());
        var geometry = new Geometry(semanticTypes, output.descriptor().dataType(),
                epsilonBits, axis, prefix, channels, suffix, nonChannel,
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

    /**
     * Qualifies the complete original BatchNorm occurrence without resolving its named axis.
     * All semantic input positions and saved output siblings remain explicit in the recipe.
     * @param context complete single-node ordinary analysis projection
     * @return unresolved family geometry with typed numeric boundary coefficients
     * @throws NullPointerException if the context or a required fact is null
     * @throws IllegalArgumentException if attributes, original descriptors or aliases disagree
     * @throws ArithmeticException if fixed coefficient or exact-state arithmetic overflows
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
        if (context.nodes().size() != 1) throw new IllegalArgumentException("BatchNorm needs one node");
        var node = context.nodes().getFirst();
        if (node.operation().kind() != BatchNormKind.BATCH_NORM_INFERENCE
                || !(node.operation().attrs() instanceof BatchNormInferenceAttrs attrs)
                || node.inputs().size() != 5 || node.outputs().size() != 1
                || node.inputs().contains(node.outputs().getFirst()))
            throw new IllegalArgumentException("unsupported symbolic BatchNorm inference");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var operands = node.inputs().stream().map(id -> require(values, id)).toList();
        var outputs = node.outputs().stream().map(id -> require(values, id)).toList();
        var input = operands.getFirst().descriptor(); int axis = attrs.channelAxis();
        if (input.shape().rank() < 2 || axis < 0 || axis >= input.shape().rank())
            throw new IllegalArgumentException("BatchNorm channel axis disagrees");
        DataType result = outputs.getFirst().descriptor().dataType();
        var ids = new ArrayList<ValueId>(); var map = new ArrayList<Integer>();
        for (var id : node.inputs()) { int b = ids.indexOf(id);
            if (b < 0) { b = ids.size(); ids.add(id); } map.add(b); }
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>(); var types = new ArrayList<DataType>();
        for (var id : ids) { var value = require(values, id);
            bindings.add(batchBinding(value, CpuAccessPlan.AccessKind.READ, layouts)); types.add(value.descriptor().dataType()); }
        int unique = ids.size();
        for (var output : outputs) { ids.add(output.id()); types.add(output.descriptor().dataType());
            bindings.add(batchBinding(output, CpuAccessPlan.AccessKind.WRITE, layouts)); }
        var read = bindings.getFirst();
        // Non-channel ranges make the range count affine even when B=0 empties a leading
        // or trailing non-channel axis. A B-bearing channel uses complete channel ranges.
        boolean channelBatch = read.extentBatchFactors().get(axis) != 0;
        var form = channelBatch ? CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE
                : CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE;
        long c = 1, f = 0;
        for (int a = 0; a < input.shape().rank(); a++) if (channelBatch ? a == axis : a != axis) {
            long ec = read.extentConstants().get(a), ef = read.extentBatchFactors().get(a);
            f = Math.addExact(Math.multiplyExact(f, ec), Math.multiplyExact(c, ef)); c = Math.multiplyExact(c, ec);
        }
        for (int a = 0; a < input.shape().rank(); a++) {
            if (read.extentConstants().get(a) == 0 && read.extentBatchFactors().get(a) == 0) {
                c = 0; f = 0; break;
            }
        }
        var ir = new CpuBatchNormInferenceIr(operands.stream().map(v -> v.descriptor().dataType()).toList(),
                result, scalarBits(result, attrs.epsilon()), input.shape().rank(), axis, 1, form, map,
                bindings.subList(0, unique).stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                bindings.getLast().plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir, ids, types,
                bindings, new BatchByteSize(c, f, 1), new BatchByteSize(0, 0, 1)));
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

    private static long scalarBits(DataType resultType, ScalarValue value) {
        if (resultType == DataType.FLOAT16 && value.dataType() == DataType.FLOAT32) {
            return Float.floatToFloat16(value.float32Value()) & 0xffffL;
        }
        return switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(value.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> value.bfloat16Bits() & 0xffffL;
            case FLOAT16 -> value.float16Bits() & 0xffffL;
            default -> throw new IllegalArgumentException(
                    "batch-normalization epsilon is not floating");
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
        return type == DataType.BFLOAT16 || type == DataType.FLOAT16
                || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }

    private static void validateInjective(Layout layout) {
            if (!CpuOutputInjectivity.isInjective(layout.extents, layout.strides)) {
                throw new IllegalArgumentException("output layout injectivity is unproved");
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
        public record BatchGeometry(CpuBatchNormInferenceIr ir, List<ValueId> boundaries,
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
                var layouts = resolved.stream().map(CpuBatchNormInferenceLowering::boundLayout).toList();
                int unique = layouts.size() - 1; var output = layouts.getLast();
                long prefix = product(output.extents, 0, ir.channelAxis());
                long channels = output.extents[ir.channelAxis()];
                long suffix = product(output.extents, ir.channelAxis() + 1, output.extents.length);
                long nonChannel = Math.multiplyExact(prefix, suffix);
                var geometry = new Geometry(ir.inputTypes(), ir.resultType(), ir.epsilonBits(),
                        ir.channelAxis(), prefix, channels, suffix, nonChannel,
                        Math.multiplyExact(channels, nonChannel), ir.rangeForm(), ir.positionToBoundary(),
                        layouts.subList(0, unique), output);
                long items = geometry.rangeItemCount();
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {items}, items,
                        "legal: symbolic batch-normalization inference", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.of(geometry), Optional.empty());
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
            var bases = new ArrayList<BaseCell>(inputs.size() + 1);
            for (int i = 0; i < inputs.size(); i++)
                bases.add(new BaseCell(9 + i, i, inputs.get(i).offset));
            bases.add(new BaseCell(9 + inputs.size(), inputs.size(), output.offset));
            return new TemporalRebind(bases, List.of());
        }

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
