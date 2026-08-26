package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr;
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
        DataType exact = result == DataType.BFLOAT16 ? DataType.FLOAT32 : result;
        long slice = channels == 0 ? 0 : exactStateSliceBytes(exact, reduction);
        int limbs = slice == 0 ? 0 : Math.toIntExact(slice / Long.BYTES - 1);
        List<DataType> semanticTypes = node.inputs().stream()
                .map(id -> require(values, id).descriptor().dataType()).toList();
        List<CpuAccessPlan> inputPlans = bindings.subList(0, uniqueIds.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        List<CpuAccessPlan> outputPlans = bindings.subList(uniqueIds.size(), bindings.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        var ir = new CpuBatchNormTrainingIr(semanticTypes, result, bits(attrs.momentum()),
                bits(attrs.epsilon()), input.extents.length, axis, 1, 3, reduction,
                limbs, slice, map, inputPlans, outputPlans);
        var geometry = new Geometry(semanticTypes, result, bits(attrs.momentum()), bits(attrs.epsilon()),
                axis, prefix, channels, suffix, reduction, count, map, inputLayouts, outputLayouts,
                slice, limbs);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), new long[]{channels}, channels, "legal: one static batch-normalization training",
                new long[0], Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(geometry));
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
    private static long bits(ScalarValue v) { return switch (v.dataType()) {
        case FLOAT64 -> Double.doubleToRawLongBits(v.float64Value());
        case FLOAT32 -> Float.floatToRawIntBits(v.float32Value()) & 0xffff_ffffL;
        case BFLOAT16 -> v.bfloat16Bits() & 0xffffL; default -> throw new IllegalArgumentException("scalar"); }; }
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
    private static void validateInjective(Layout l) { long n=count(l.extents); if(n==0)return;
        if(n<=1_000_000){var seen=new HashSet<Long>();long[] c=new long[l.extents.length];for(long k=0;k<n;k++){long a=l.offset;
            for(int i=0;i<c.length;i++)a=Math.addExact(a,Math.multiplyExact(c[i],l.strides[i]));
            if(!seen.add(a))throw new IllegalArgumentException("training output layout is not injective");
            for(int i=c.length-1;i>=0;i--){if(++c[i]<l.extents[i])break;c[i]=0;}}return;}
        var axes=new ArrayList<Integer>();for(int i=0;i<l.extents.length;i++)if(l.extents[i]>1)axes.add(i);
        axes.sort(Comparator.comparingLong(i->l.strides[i]));long covered=1;for(int i:axes){if(l.strides[i]<covered)
            throw new IllegalArgumentException("training output layout is not injective");covered=Math.addExact(covered,Math.multiplyExact(l.extents[i]-1,l.strides[i]));}}

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
