package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Lowers one explicit, fully static batch-normalization training occurrence into CPU-private
 * identity, access, and complete-channel geometry.
 *
 * <p>Lowering deduplicates equal input values in first-occurrence order, preserves all five
 * semantic positions, requires five distinct non-overlapping logical outputs, and sizes one
 * exact-state slice for each simultaneously active channel range. It selects no runtime state or
 * public training behavior.</p>
 * <p>The single-B entry preserves every saved sibling and binds only numeric geometry.
 * Invocation-domain exact sums reserve a fixed long-contributor capacity and obtain the actual
 * denominator from the existing pack, without baking invocation B into generated identity.</p>
 */
public final class CpuBatchNormTrainingLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless training lowerer with the truthful CPU capability provider. */
    public CpuBatchNormTrainingLowering() { }

    /**
     * Lowers one supported five-input/five-output occurrence.
     *
     * @param context projected single-partition facts and resolved descriptors; must contain one
     *     explicit {@code BATCH_NORM_TRAINING} node and every referenced graph value
     * @return immutable lowered IR, unique boundary declarations, access plans, channel range,
     *     and exact-state geometry; never {@code null}
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence is unsupported, a referenced value is
     *     absent, outputs alias inputs or one another, a writable layout is not injective, or
     *     static geometry cannot be represented by current CPU contracts
     * @throws ArithmeticException if a checked count, span, address, or state-size calculation
     *     overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU batch-normalization training requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query) || node.operation().kind() != BatchNormKind.BATCH_NORM_TRAINING
                || !(node.operation().attrs() instanceof BatchNormTrainingAttrs attrs)
                || node.inputs().size() != 5 || node.outputs().size() != 5)
            throw new IllegalArgumentException("unsupported CPU batch-normalization training occurrence");
        var distinct = new HashSet<ValueId>(node.outputs());
        if (distinct.size() != 5 || node.outputs().stream().anyMatch(node.inputs()::contains))
            throw new IllegalArgumentException("training outputs must be mutually distinct from inputs");
        Layout input = layout(require(values, node.inputs().getFirst()));
        int axis = attrs.channelAxis();
        long prefix = product(input.extents, 0, axis), channels = input.extents[axis];
        long suffix = product(input.extents, axis + 1, input.extents.length);
        long reduction = Math.multiplyExact(prefix, suffix);
        long count = Math.multiplyExact(channels, reduction);
        if (channels > 0 && reduction < 2) throw new IllegalArgumentException(
                "training reduction count must be at least two");

        var uniqueIds = new ArrayList<ValueId>(); var map = new ArrayList<Integer>();
        for (ValueId id : node.inputs()) { int b = uniqueIds.indexOf(id);
            if (b < 0) { b = uniqueIds.size(); uniqueIds.add(id); } map.add(b); }
        var inputLayouts = new ArrayList<Layout>(); var outputLayouts = new ArrayList<Layout>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>(); var spans = new ArrayList<Long>();
        var types = new ArrayList<DataType>(); var boundaryIds = new ArrayList<ValueId>();
        for (ValueId id : uniqueIds) addBoundary(values, id, CpuAccessPlan.AccessKind.READ,
                inputLayouts, bindings, spans, types, boundaryIds);
        for (ValueId id : node.outputs()) {
            GraphValue value = require(values, id); Layout l = layout(value); validateInjective(l);
            outputLayouts.add(l); bindings.add(binding(l, CpuAccessPlan.AccessKind.WRITE));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            types.add(value.descriptor().dataType()); boundaryIds.add(id);
        }
        DataType result = require(values, node.outputs().getFirst()).descriptor().dataType();
        DataType exact = result.byteWidth() < Float.BYTES ? DataType.FLOAT32 : result;
        long slice = channels == 0 ? 0 : exactStateSliceBytes(exact, reduction);
        int limbs = slice == 0 ? 0 : Math.toIntExact(slice / Long.BYTES - 1);
        List<DataType> semanticTypes = node.inputs().stream()
                .map(id -> require(values, id).descriptor().dataType()).toList();
        List<CpuAccessPlan> inputPlans = bindings.subList(0, uniqueIds.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        List<CpuAccessPlan> outputPlans = bindings.subList(uniqueIds.size(), bindings.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        long momentumBits = bits(attrs.momentum());
        long epsilonBits = bits(attrs.epsilon());
        var ir = new CpuBatchNormTrainingIr(semanticTypes, result, momentumBits,
                epsilonBits, input.extents.length, axis, 1, 3, reduction,
                limbs, slice, map, inputPlans, outputPlans);
        var geometry = new Geometry(semanticTypes, result, momentumBits, epsilonBits,
                axis, prefix, channels, suffix, reduction, count, map, inputLayouts, outputLayouts,
                slice, limbs);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), new long[]{channels}, channels, "legal: one static batch-normalization training",
                new long[0], Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(geometry));
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
        if (node.operation().kind() != BatchNormKind.BATCH_NORM_TRAINING
                || !(node.operation().attrs() instanceof BatchNormTrainingAttrs attrs)
                || node.inputs().size() != 5 || node.outputs().size() != 5
                || new HashSet<>(node.outputs()).size() != 5
                || node.outputs().stream().anyMatch(node.inputs()::contains))
            throw new IllegalArgumentException("unsupported symbolic BatchNorm training");
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
        long rc = 1, rf = 0;
        for (int a = 0; a < input.shape().rank(); a++) if (a != axis) {
            long ec = read.extentConstants().get(a), ef = read.extentBatchFactors().get(a);
            rf = Math.addExact(Math.multiplyExact(rf, ec), Math.multiplyExact(rc, ef));
            rc = Math.multiplyExact(rc, ec);
        }
        boolean knownEmpty = read.extentConstants().get(axis) == 0 && read.extentBatchFactors().get(axis) == 0;
        boolean dynamicDomain = rf != 0 || rc < 2 && read.extentBatchFactors().get(axis) != 0;
        if (!dynamicDomain && !knownEmpty && rc < 2)
            throw new IllegalArgumentException("training reduction must contain at least two coordinates");
        DataType exact = result == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        // v2 explicitly names invocation-domain arithmetic. Long capacity is a proved upper
        // contributor bound, not an invented concrete shape or representative domain.
        long slice = knownEmpty ? 0 : exactStateSliceBytes(exact, dynamicDomain ? Long.MAX_VALUE : rc);
        int limbs = slice == 0 ? 0 : Math.toIntExact(slice / Long.BYTES - 1);
        var ir = new CpuBatchNormTrainingIr(operands.stream().map(v -> v.descriptor().dataType()).toList(),
                result, bits(attrs.momentum()), bits(attrs.epsilon()),
                input.shape().rank(), axis, dynamicDomain ? 2 : 1, 3, dynamicDomain ? 0 : rc,
                limbs, slice, map,
                bindings.subList(0, unique).stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                bindings.subList(unique, bindings.size()).stream().map(CpuAccessPlan.BatchBinding::plan).toList());
        var iterations = batchProduct(read, axis, axis + 1);
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir, ids, types,
                bindings, iterations, new BatchByteSize(slice, 0, 1)));
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

    private static BatchByteSize batchProduct(CpuAccessPlan.BatchBinding binding, int begin, int end) {
        long constant = 1, factor = 0;
        for (int axis = begin; axis < end; axis++) {
            long ec = binding.extentConstants().get(axis), ef = binding.extentBatchFactors().get(axis);
            if (factor != 0 && ef != 0) throw new IllegalArgumentException("non-affine normalization count");
            factor = Math.addExact(Math.multiplyExact(factor, ec), Math.multiplyExact(constant, ef));
            constant = Math.multiplyExact(constant, ec);
        }
        return new BatchByteSize(constant, factor, 1);
    }

    private static Layout boundLayout(CpuAccessPlan.Binding binding) {
        return new Layout(binding.extents().stream().mapToLong(Long::longValue).toArray(),
                binding.baseElementOffset(), binding.effectiveStrides().stream()
                        .mapToLong(Long::longValue).toArray());
    }

    private static void addBoundary(Map<ValueId, GraphValue> values, ValueId id,
            CpuAccessPlan.AccessKind kind, List<Layout> layouts, List<CpuAccessPlan.Binding> bindings,
            List<Long> spans, List<DataType> types, List<ValueId> ids) {
        GraphValue v = require(values, id); Layout l = layout(v); layouts.add(l); bindings.add(binding(l, kind));
        spans.add(v.descriptor().layout().orElseThrow().referencedElementSpan());
        types.add(v.descriptor().dataType()); ids.add(id);
    }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue v = values.get(id); if (v == null) throw new IllegalArgumentException("value is not projected: " + id); return v; }
    private static long bits(ScalarValue v) {
        return switch (v.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(v.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(v.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> v.bfloat16Bits() & 0xffffL;
            case FLOAT16 -> v.float16Bits() & 0xffffL;
            default -> throw new IllegalArgumentException("scalar");
        };
    }
    private static Layout layout(GraphValue v) { LayoutDescriptor l = v.descriptor().layout().orElseThrow();
        if (l.storageOffset() < 0 || Arrays.stream(l.strides()).anyMatch(x -> x < 0))
            throw new IllegalArgumentException("training requires non-negative layouts");
        return new Layout(v.descriptor().shape().toLongArray(), l.storageOffset(), l.strides()); }
    private static CpuAccessPlan.Binding binding(Layout l, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = l.extents.length - 1; axis >= 0; axis--) { if (l.strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, l.extents[axis])); }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < l.extents.length; axis++) roles.add(l.strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST : axis >= l.extents.length - suffix
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == l.extents.length ? CpuAccessPlan.Regime.DENSE_LINEAR
                : CpuAccessPlan.Regime.GENERAL_ODOMETER, l.extents.length, roles, suffix);
        long count = count(l.extents); return CpuAccessPlan.Binding.create(plan, l.extents, l.offset,
                l.strides, count, 0, count, span(l)); }
    private static long exactStateSliceBytes(DataType type, long count) { int emin = type == DataType.FLOAT64 ? -1074 : -149;
        int emax = type == DataType.FLOAT64 ? 1023 : 127; int cb = count <= 1 ? 0 : 64 - Long.numberOfLeadingZeros(count - 1);
        long bits = Math.addExact((long) emax + 1 - emin, Math.addExact(cb, 1));
        return Math.addExact(8, Math.multiplyExact(8, Math.addExact(bits, 63) / 64)); }
    private static long product(long[] e, int begin, int end) { long r=1; for(int i=begin;i<end;i++) r=Math.multiplyExact(r,e[i]); return r; }
    private static long count(long[] e) { for(long x:e) if(x==0)return 0; return product(e,0,e.length); }
    private static long span(Layout l) { if(count(l.extents)==0)return 0; long m=l.offset;
        for(int i=0;i<l.extents.length;i++)m=Math.addExact(m,Math.multiplyExact(l.extents[i]-1,l.strides[i])); return Math.addExact(m,1); }
    private static void validateInjective(Layout l) {
            if (!CpuOutputInjectivity.isInjective(l.extents, l.strides)) {
                throw new IllegalArgumentException("output layout injectivity is unproved");
            }
        }

    /**
     * Resolved non-negative layout in element units.
     *
     * @param extents static non-negative extents; defensively copied
     * @param offset non-negative carrier-relative element offset
     * @param strides non-negative element strides matching {@code extents}; defensively copied
     */
    public record Layout(long[] extents,long offset,long[] strides){
        /**
         * Validates and defensively copies the resolved layout arrays.
         *
         * @throws NullPointerException if {@code extents} or {@code strides} is {@code null}
         * @throws IllegalArgumentException if ranks differ or an extent, offset, or stride is
         *     negative
         */
        public Layout{extents=extents.clone();strides=strides.clone();
        if(extents.length!=strides.length||offset<0||Arrays.stream(extents).anyMatch(x->x<0)||Arrays.stream(strides).anyMatch(x->x<0))
            throw new IllegalArgumentException("invalid training layout");}
        /**
         * Returns the static extents without exposing the retained array.
         * @return a defensive copy of the static extents; never {@code null}
         */
        @Override public long[] extents(){return extents.clone();}
        /**
         * Returns the element strides without exposing the retained array.
         * @return a defensive copy of the element strides; never {@code null}
         */
        @Override public long[] strides(){return strides.clone();}}

    /**
     * Complete cold arbitrary-axis geometry for one generated training entry.
     *
     * @param inputTypes five semantic input types in occurrence order
     * @param resultType promoted result and computation type
     * @param momentumBits raw result-type momentum bits
     * @param epsilonBits raw result-type epsilon bits
     * @param channelAxis normalized input channel axis
     * @param prefixCount product of extents before the channel axis
     * @param channelCount channel-axis extent and generated range domain
     * @param suffixCount product of extents after the channel axis
     * @param reductionCount complete non-channel coordinate count per channel
     * @param inputCount total input coordinate count
     * @param positionToBoundary five-position map to unique inputs in first-occurrence order
     * @param inputs unique input layouts in boundary order
     * @param outputs five output layouts in semantic output order
     * @param scratchSliceBytes exact-state bytes required by one active range, or zero for empty
     *     channel work
     * @param stateLimbCount exact-sum limb count, or zero for empty channel work
     */
    public record Geometry(List<DataType> inputTypes,DataType resultType,long momentumBits,long epsilonBits,
            int channelAxis,long prefixCount,long channelCount,long suffixCount,long reductionCount,
            long inputCount,List<Integer> positionToBoundary,List<Layout> inputs,List<Layout> outputs,
            long scratchSliceBytes,int stateLimbCount){

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
        public record BatchGeometry(CpuBatchNormTrainingIr ir, List<ValueId> boundaries,
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
                var layouts = resolved.stream().map(CpuBatchNormTrainingLowering::boundLayout).toList();
                int unique = layouts.size() - 5; var input = layouts.getFirst();
                long prefix = product(input.extents, 0, ir.channelAxis());
                long channels = input.extents[ir.channelAxis()];
                long suffix = product(input.extents, ir.channelAxis() + 1, input.extents.length);
                long reduction = Math.multiplyExact(prefix, suffix);
                if (channels > 0 && reduction < 2) throw new IllegalArgumentException(
                        "training reduction count must be at least two");
                long slice = channels == 0 ? 0 : scratchBytes.byteSize(batch);
                int limbs = channels == 0 ? 0 : ir.stateLimbCount();
                var geometry = new Geometry(ir.inputTypes(), ir.resultType(), ir.momentumBits(),
                        ir.epsilonBits(), ir.channelAxis(), prefix, channels, suffix, reduction,
                        Math.multiplyExact(channels, reduction), ir.positionToBoundary(),
                        layouts.subList(0, unique), layouts.subList(unique, layouts.size()), slice, limbs);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {channels}, channels,
                        "legal: symbolic batch-normalization training", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.of(geometry));
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
            var bases = new ArrayList<BaseCell>(inputs.size() + outputs.size());
            for (int i = 0; i < inputs.size(); i++)
                bases.add(new BaseCell(11 + i, i, inputs.get(i).offset));
            for (int i = 0; i < outputs.size(); i++)
                bases.add(new BaseCell(11 + inputs.size() + i, inputs.size() + i,
                        outputs.get(i).offset));
            return new TemporalRebind(bases, List.of());
        }

        /**
         * Validates and snapshots collection-valued geometry facts.
         *
         * @throws NullPointerException if a required collection, type, layout, or mapping entry
         *     is {@code null}
         * @throws IllegalArgumentException if output cardinality, counts, or exact-state shape
         *     disagree
         * @throws ArithmeticException if the checked input-count relationship overflows
         */
        public Geometry{inputTypes=List.copyOf(inputTypes);
        positionToBoundary=List.copyOf(positionToBoundary);inputs=List.copyOf(inputs);outputs=List.copyOf(outputs);
        if(outputs.size()!=5||channelCount<0||reductionCount<0||inputCount!=Math.multiplyExact(channelCount,reductionCount)
                ||(channelCount>0&&(reductionCount<2||scratchSliceBytes<=0||stateLimbCount<=0))
                ||(channelCount==0&&(scratchSliceBytes!=0||stateLimbCount!=0)))throw new IllegalArgumentException("invalid training geometry");}
        /**
         * Returns the exact aggregate workspace size for simultaneous complete-channel ranges.
         *
         * @param ranges non-negative selected range count
         * @return exact workspace bytes; zero for empty work
         * @throws ArithmeticException if the byte count overflows
         */
        public long workspaceBytes(int ranges){return Math.multiplyExact(scratchSliceBytes,ranges);}
        /**
         * Packs carrier-relative bases, layouts, counts, and scratch placement for invocation.
         *
         * @param bases unique-input bases followed by five output bases; not retained
         * @param scratchOffset byte offset of this range's exact-state slice
         * @return a newly allocated packed primitive geometry array; never {@code null}
         * @throws IllegalArgumentException if {@code bases} has the wrong cardinality
         * @throws ArithmeticException if a base plus layout offset overflows
         */
        public long[] pack(long[] bases,long scratchOffset){if(bases.length!=inputs.size()+5)throw new IllegalArgumentException("base count");
            int size=11+bases.length;for(Layout l:inputs)size+=1+2*l.extents.length;for(Layout l:outputs)size+=1+2*l.extents.length;
            long[] p=new long[size];int i=0;p[i++]=inputs.size();p[i++]=outputs.size();p[i++]=channelAxis;p[i++]=prefixCount;
            p[i++]=channelCount;p[i++]=suffixCount;p[i++]=inputCount;p[i++]=reductionCount;p[i++]=stateLimbCount;
            p[i++]=scratchSliceBytes;p[i++]=scratchOffset;for(int b=0;b<bases.length;b++){Layout l=b<inputs.size()?inputs.get(b):outputs.get(b-inputs.size());p[i++]=Math.addExact(bases[b],l.offset);}
            for(Layout l:inputs)i=packLayout(p,i,l);for(Layout l:outputs)i=packLayout(p,i,l);return p;}
        private static int packLayout(long[] p,int i,Layout l){p[i++]=l.extents.length;for(long x:l.extents)p[i++]=x;for(long x:l.strides)p[i++]=x;return i;}}
}
