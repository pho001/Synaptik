package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.StaticDimension;
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
import java.util.OptionalLong;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;

/**
 * Lowers one fully static supported ordinary aggregate or binding-aware SUM-to-Shape occurrence.
 *
 * <p>The lowerer canonicalizes selected-axis membership, derives complete output-cell and
 * selected-domain geometry, and declares exactly one input and one distinct injective output.
 * SUM-to-Shape derives leading and unequal target-one axes from exact right alignment while
 * preserving equal aligned extents before considering target-one reduction.
 * Floating SUM, MEAN, and PROD derive exact fixed-width state; other rows select no workspace.
 * No row creates a partial reduction or combine state.</p>
 * <p>The optional ordinary fusion proof composes this unchanged exact geometry with existing
 * pointwise lowering in separate contribution and result domains. It removes only complete-use
 * proved private stores; a tail never expands the aggregate output Shape.</p>
 * <p>The fixed softmax derivative proof is separate: its exact trailing SUM precedes a
 * second full-gradient-domain pass, retaining every captured rounded primitive and removing
 * only three private stores. Unproved patterns, access mappings and bounds retain decomposition.</p>
 * <p>Package-local axis, attribute-form and target checks also serve Scan without constructing
 * an ordinary exact-state aggregate IR. Dimensions compare by exact value/name; actual batch
 * singleton filtering, counts and workspace remain the existing numeric binding's responsibility.</p>
 * <p>Ordinary fully static explicit typed-one SUMs have a separate cold count proof. It reuses
 * this lowerer's geometry guards and directly rounds the exact integer to the represented CPU
 * SUM type, including low floating types, before declaring an output-only constant write.
 * Publication, outside use or fan-out at an affine intermediate rejects absorption.</p>
 * <p>The separate {@link #lowerBatch} path authenticates the original canonical named-B
 * occurrence and retains only checked numeric coefficients. Cold binding creates this owner's
 * existing concrete geometry and pack once; nonleading batch axes and exact boundary types are
 * preserved without a representative graph or a batch-specialized generated class.</p>
 */
public final class CpuAggregateLowering {
    /**
     * Complete prepared numeric FLOAT32/FLOAT64 SUM/MEAN domain and conditional exact-state declaration.
     * Axis membership is selected during preparation; SUM-to-Shape may become a copy at B=1.
     * @param kind SUM or MEAN (SUM only for SUM-to-Shape)
     * @param dataType exact homogeneous FLOAT32 or FLOAT64 type
     * @param form non-null reduction attribute form
     * @param axes non-null increasing prepared axis membership, copied
     * @param keepDimensions whether selected ordinary axes remain extent one
     * @param input non-null prepared read geometry
     * @param output non-null prepared canonical write geometry
     */
    public record BatchGeometry(CpuAggregateIr.Kind kind, DataType dataType,
            CpuAggregateIr.Form form, List<Integer> axes, boolean keepDimensions,
            CpuAccessPlan.BatchBinding input, CpuAccessPlan.BatchBinding output)
            implements PreparedMemoryPlan.ByteSize {
        /**
         * Validates supported kind, type and complete prepared axis declarations.
         * @param kind SUM or MEAN
         * @param dataType FLOAT32 or FLOAT64; never a narrowing working type
         * @param form non-null attribute form
         * @param axes non-null increasing normalized axes without nulls, copied
         * @param keepDimensions whether reduced ordinary axes remain extent one
         * @param input non-null read recipe
         * @param output non-null canonical write recipe
         * @throws NullPointerException if a reference or axis is null
         * @throws IllegalArgumentException if type, form, direction or axes disagree
         */
        public BatchGeometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(form, "form"); axes = List.copyOf(axes);
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64
                    || kind != CpuAggregateIr.Kind.SUM && kind != CpuAggregateIr.Kind.MEAN
                    || form == CpuAggregateIr.Form.SUM_TO_SHAPE && (kind != CpuAggregateIr.Kind.SUM
                        || keepDimensions)
                    || input.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                    || output.plan().accessKind() != CpuAccessPlan.AccessKind.WRITE) {
                throw new IllegalArgumentException("unsupported ordinary batch aggregate");
            }
            int previous = -1;
            for (int axis : axes) {
                if (axis <= previous || axis >= input.plan().iterationRank()) {
                    throw new IllegalArgumentException("batch aggregate axes disagree");
                }
                previous = axis;
            }
        }

        /**
         * Resolves actual domains and layouts with fixed five-limb F32 or 34-limb F64 exact
         * state. Int-bounded contributors need at most 309/2130 signed bits respectively;
         * static domain-sized state is deliberately unchanged.
         * @param batchSize non-negative actual int-representable B
         * @return immutable numeric actual geometry; all arrays are run-owned snapshots
         * @throws IllegalArgumentException if B or resolved semantic dimensions disagree
         * @throws ArithmeticException if numeric geometry is unrepresentable
         */
        public Geometry bind(long batchSize) {
            var in = input.bind(batchSize); var out = output.bind(batchSize);
            int leading = in.extents().size() - out.extents().size();
            int[] selected = axes.stream().filter(axis -> form != CpuAggregateIr.Form.SUM_TO_SHAPE
                    || axis < leading || !in.extents().get(axis).equals(out.extents().get(axis - leading)))
                    .mapToInt(Integer::intValue).toArray();
            long domain = 1;
            for (int index = selected.length - 1; index >= 0; index--) {
                domain = Math.toIntExact(Math.multiplyExact(domain, in.extents().get(selected[index])));
            }
            return new Geometry(kind, dataType, form, selected, keepDimensions,
                    new Layout(in.extents().stream().mapToLong(Long::longValue).toArray(),
                            in.baseElementOffset(), in.effectiveStrides().stream().mapToLong(Long::longValue).toArray()),
                    new Layout(out.extents().stream().mapToLong(Long::longValue).toArray(),
                            out.baseElementOffset(), out.effectiveStrides().stream().mapToLong(Long::longValue).toArray()),
                    out.elementCount(), domain, dataType == DataType.FLOAT64 ? 34 : 5,
                    dataType == DataType.FLOAT64 ? 280 : 48, true);
        }
        /** Keeps scratch declarations run-bound even when a particular B needs no bytes.
         * @return empty; conditional scratch is structurally run-bound */
        @Override public OptionalLong fixedByteSize() { return OptionalLong.empty(); }
        /**
         * Resolves exact single-range scratch, including zero for an empty output.
         * @param batchSize non-negative actual int-representable B
         * @return 48 F32 or 280 F64 bytes when actual output cells exist, otherwise zero
         * @throws IllegalArgumentException if actual geometry is invalid
         * @throws ArithmeticException if checked geometry exceeds CPU limits
         */
        @Override public long byteSize(long batchSize) { return bind(batchSize).workspaceBytes(1); }
    }
    /**
     * Authenticates original ordinary symbolic dimensions and boundary cardinality without
     * substituting a batch value. Independent named coordinates and output aliases are rejected.
     * @param context complete one-node projection, borrowed only during lowering
     * @return original semantic query, with no sampled descriptors
     * @throws IllegalArgumentException if the projection is not one canonical batch domain
     */
    static OperationCapabilityQuery batchQuery(PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException("batch occurrence must be atomic");
        var node = context.nodes().getFirst();
        if (node.outputs().size() != 1 || node.inputs().contains(node.outputs().getFirst()))
            throw new IllegalArgumentException("batch output must be unique and distinct");
        var values = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(v -> values.put(v.id(), v));
        Dimension named = null;
        for (var id : java.util.stream.Stream.concat(node.inputs().stream(), node.outputs().stream()).toList()) {
            var descriptor = require(values, id).descriptor(); int dynamic = 0;
            for (var dimension : descriptor.shape().dimensions()) if (!(dimension instanceof StaticDimension)) {
                if (!(dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension)
                        || ++dynamic > 1 || named != null && !named.equals(dimension))
                    throw new IllegalArgumentException("ordinary geometry requires one canonical named batch");
                named = dimension;
            }
            CpuPartitionLowering.batchCanonical(descriptor, CpuAccessPlan.AccessKind.READ);
        }
        return new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values,id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values,id).descriptor()).toList());
    }

    /**
     * Retains exact projected read layouts and the original output's proved injective mapping.
     * @param context original one-node projection
     * @param boundaries unique ordered inputs followed by the sole output
     * @param layouts complete enclosing numeric physical layout map
     * @return immutable numeric coefficient bindings with an explicit write permission
     * @throws IllegalArgumentException if a layout or output injectivity is unproved
     */
    static List<CpuAccessPlan.BatchBinding> batchBindings(
            PrepareContext<? extends BackendAnalysisInputs> context, List<ValueId> boundaries,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var result = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (var id : boundaries) result.add(Objects.requireNonNull(layouts.get(id)));
        var out = result.getLast();
        var write = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.WRITE,
                out.extentConstants(),out.extentBatchFactors(),out.strideConstants(),out.strideBatchFactors());
        result.set(result.size()-1,new CpuAccessPlan.BatchBinding(write.plan(),out.extentConstants(),
                out.extentBatchFactors(),out.baseElementOffset(),out.strideConstants(),out.strideBatchFactors()));
        var output = context.values().stream().filter(v -> v.id().equals(boundaries.getLast())).findFirst().orElseThrow();
        if (output.descriptor().shape().isFullyStatic() && output.descriptor().layout().isPresent())
            validateInjective(layout(output));
        return List.copyOf(result);
    }

    /**
     * Multiplies selected extent coefficients with checked affine arithmetic, never binding B.
     * @param binding original numeric access recipe
     * @param axes selected axes, in any unique order
     * @return element-count recipe; its byte unit is deliberately one
     * @throws ArithmeticException if a coefficient overflows
     * @throws IllegalArgumentException if the count is non-affine
     */
    static BatchByteSize batchCount(CpuAccessPlan.BatchBinding binding, int[] axes) {
        long constant = 1, factor = 0;
        for (int axis : axes) {
            long c = binding.extentConstants().get(axis), f = binding.extentBatchFactors().get(axis);
            if (factor != 0 && f != 0) throw new IllegalArgumentException("non-affine selected count");
            factor = Math.addExact(Math.multiplyExact(factor,c),Math.multiplyExact(constant,f));
            constant = Math.multiplyExact(constant,c);
        }
        return new BatchByteSize(constant,factor,1);
    }

    /**
     * Derives the complete logical element count directly from coefficients.
     * @param binding non-null original numeric recipe
     * @return checked affine element-count recipe
     */
    static BatchByteSize batchCount(CpuAccessPlan.BatchBinding binding) {
        return batchCount(binding,java.util.stream.IntStream.range(0,binding.plan().iterationRank()).toArray());
    }

    /**
     * Returns an exact fixed domain or the structural int-limited contributor budget.
     * @param count affine selected-domain recipe
     * @return upper bound independent of invocation B
     */
    static long batchDomainBudget(BatchByteSize count) {
        return count.bytesPerBatch() == 0 ? count.constantBytes() : Integer.MAX_VALUE;
    }

    /**
     * Lowers the original one-canonical-B occurrence to an unresolved numeric recipe.
     * No descriptor, graph or generated identity is specialized to a representative batch.
     * @param context complete original one-node analysis projection
     * @return immutable ordinary family recipe, resolved only during cold invocation binding
     * @throws IllegalArgumentException if original attributes, types, shapes or layouts disagree
     * @throws ArithmeticException if structural coefficient or scratch arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context,CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Authenticates an ordinary symbolic occurrence using complete enclosing physical layouts.
     * @param context original one-node projection with exact semantic descriptors
     * @param layouts immutable full-local numeric layout map, including views outside this unit
     * @return unresolved family recipe retaining those exact physical coefficients
     * @throws IllegalArgumentException if original semantics or physical geometry are unproved
     * @throws ArithmeticException if structural arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var query = CpuAggregateLowering.batchQuery(context);
        var node = context.nodes().getFirst();
        CpuPartitionLowering.requireBatchContext(context);
        if (node.inputs().size() != 1) throw new IllegalArgumentException("unary reduction required");
        var input = query.inputs().getFirst(); var output = query.outputs().getFirst();
        var type = input.dataType();
        var boundaries = List.of(node.inputs().getFirst(),node.outputs().getFirst());
        var types = List.of(type,output.dataType());
        var bindings = CpuAggregateLowering.batchBindings(context,boundaries,layouts);
        var in = bindings.getFirst(); var out = bindings.getLast();
        if (!(node.operation().kind() instanceof AggregateReductionKind modelKind) || type != output.dataType())
            throw new IllegalArgumentException("aggregate batch types disagree");
        var kind = CpuAggregateIr.Kind.valueOf(modelKind.name());
        var attrs = node.operation().attrs(); var form = reductionForm(attrs); boolean keep = keepDimensions(attrs);
        int[] axes = normalizedAxes(input.shape(),output.shape(),attrs).stream().mapToInt(Integer::intValue).toArray();
        var domain = batchCount(in,axes); long budget = batchDomainBudget(domain);
        boolean floating = type == DataType.FLOAT32 || type == DataType.FLOAT64 || type == DataType.FLOAT16 || type == DataType.BFLOAT16;
        boolean exact = floating && (kind == CpuAggregateIr.Kind.SUM || kind == CpuAggregateIr.Kind.MEAN || kind == CpuAggregateIr.Kind.PROD)
                && (form != CpuAggregateIr.Form.SUM_TO_SHAPE || axes.length != 0);
        int limbs = exact ? stateLimbCount(kind,type,budget) : 0;
        long slice = exact ? Math.addExact(kind == CpuAggregateIr.Kind.PROD ? 24 : 8,Math.multiplyExact(8L,limbs)) : 0;
        BatchByteSize scratch;
        if (exact && kind == CpuAggregateIr.Kind.PROD && domain.bytesPerBatch() != 0) {
            long perBatchLimbs = stateLimbCount(kind,type,domain.bytesPerBatch());
            scratch = new BatchByteSize(32,Math.multiplyExact(8,perBatchLimbs),8);
        } else scratch = new BatchByteSize(slice,0,8);
        var ir = new CpuAggregateIr(kind,type,form,axes,keep,in.plan(),out.plan(),new long[0],new long[0],
                budget,limbs,slice,CpuAggregateIr.FIRST_LOGICAL_NAN_AND_SIGNED_ZERO,
                CpuAggregateIr.COMPLETE_OUTPUT_CELLS,exact ? CpuAggregateIr.EXACT_FLOATING_STATE : CpuAggregateIr.ZERO_WORKSPACE,true);
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                boundaries, types, bindings, batchCount(out), scratch));
    }

    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless ordinary-aggregate lowerer. */
    public CpuAggregateLowering() { }

    /**
     * Recognizes only the two captured four-node softmax derivative DAGs. Original operand
     * order, full logical contribution domain and complete private-use obligations are exact;
     * failed access/type/topology proofs retain ordinary decomposition without resource removal.
     * @param <I> immutable analysis-input type
     * @param context complete four-member projection protecting all outside consumers
     * @param pointwise existing typed pointwise lowerer and access normalizer
     * @return complete two-pass lowering, or empty when any required fact is unproved
     * @throws NullPointerException if an argument is null
     */
    <I extends BackendAnalysisInputs> Optional<CpuPartitionLowering.LoweredPartition> fuseSoftmaxBackward(
            PrepareContext<I> context, CpuPartitionLowering pointwise) {
        if (context.requiresBatchSize() || context.nodes().size() != 4) return Optional.empty();
        try {
            var nodes = context.nodes(); var root = nodes.getLast();
            var binary = io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.MUL;
            var subtract = io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.SUB;
            boolean log = root.operation().kind() == subtract;
            if ((!log && root.operation().kind() != binary) || root.inputs().size() != 2) return Optional.empty();
            var producers = new LinkedHashMap<ValueId,io.github.pho001.synaptik.model.graph.CompiledNode>();
            for (var n : nodes) {
                if (n.outputs().size() != 1) return Optional.empty();
                producers.put(n.outputs().getFirst(),n);
            }
            var second = producers.get(root.inputs().get(1));
            if (second == null || second.inputs().size() != 2
                    || second.operation().kind() != (log ? binary : subtract)) return Optional.empty();
            var sum = producers.get(second.inputs().get(1));
            var first = producers.get(log ? second.inputs().getFirst()
                    : sum == null || sum.inputs().size() != 1 ? null : sum.inputs().getFirst());
            if (sum == null || first == null || sum == first || first == second || sum == second
                    || sum.operation().kind() != AggregateReductionKind.SUM || sum.inputs().size() != 1
                    || !(sum.operation().attrs() instanceof AxisReductionAttrs axis) || !axis.keepDimensions()
                    || first.operation().kind() != (log
                        ? io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.EXP : binary)
                    || first.inputs().size() != (log ? 1 : 2)) return Optional.empty();
            ValueId saved = log ? first.inputs().getFirst() : root.inputs().getFirst();
            ValueId gradient = log ? root.inputs().getFirst() : second.inputs().getFirst();
            if (producers.containsKey(saved) || producers.containsKey(gradient)
                    || log && !sum.inputs().getFirst().equals(gradient)
                    || !log && !first.inputs().equals(List.of(gradient,saved))) return Optional.empty();
            var memory = new LinkedHashMap<ValueId,io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement>();
            context.memoryRequirements().forEach(r -> memory.put(r.valueId(),r));
            for (var n : List.of(first,sum,second)) {
                var id = n.outputs().getFirst(); var r = memory.get(id);
                var uses = context.partitionDag().consumers(id);
                var expected = n == first ? (log ? second : sum) : n == sum ? second : root;
                if (r == null || r.graphOutput() || uses.size() != 1 || uses.getFirst().node() != expected
                        || r.consumerPartitions().stream().anyMatch(p -> !p.equals(context.partition()))) return Optional.empty();
            }
            var values = new LinkedHashMap<ValueId,GraphValue>(); context.values().forEach(v -> values.put(v.id(),v));
            var shape = require(values,root.outputs().getFirst()).descriptor().shape();
            if (!shape.isFullyStatic() || shape.rank() < 1 || shape.rank() > 2
                    || axis.axis() != shape.rank()-1 || !require(values,saved).descriptor().shape().equals(shape)) return Optional.empty();
            // Existing lowerers authenticate every removed site as well as every physical read.
            var base = lower(CpuPartitionDagDecomposer.project(context,List.of(sum),context.backendInputs()));
            var aggregate = (CpuAggregateIr)base.portableKernelIr();
            if (aggregate.dataType() != DataType.FLOAT32 && aggregate.dataType() != DataType.FLOAT64
                    || !Arrays.equals(base.aggregateGeometry().orElseThrow().input().extents(),shape.toLongArray())) return Optional.empty();
            var parts = new LinkedHashMap<io.github.pho001.synaptik.model.graph.CompiledNode,CpuPartitionLowering.LoweredPartition>();
            for (var n : List.of(first,second,root)) {
                var p = pointwise.lower(CpuPartitionDagDecomposer.project(context,List.of(n),context.backendInputs()));
                if (!(p.portableKernelIr() instanceof CpuKernelIr k) || k.instructions().size() != 1
                        || p.boundaryDataTypes().stream().anyMatch(t -> t != aggregate.dataType())
                        || !require(values,n.outputs().getFirst()).descriptor().shape().equals(shape)) return Optional.empty();
                parts.put(n,p);
            }
            var ids = new ArrayList<ValueId>(); var bindings = new ArrayList<CpuAccessPlan.Binding>();
            for (var id : List.of(saved,gradient)) if (!ids.contains(id)) {
                CpuAccessPlan.Binding binding = null;
                for (var p : parts.values()) {
                    int position = p.boundaryValues().indexOf(id);
                    if (position >= 0) {
                        var candidate = p.accessBindings().get(position);
                        if (binding != null && !binding.equals(candidate)) return Optional.empty();
                        binding = candidate;
                    }
                }
                if (binding == null) return Optional.empty();
                ids.add(id); bindings.add(binding);
            }
            ids.add(root.outputs().getFirst()); bindings.add(parts.get(root).accessBindings().getLast());
            var ir = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr(log,aggregate,bindings,
                    ids.indexOf(saved),ids.indexOf(gradient));
            return Optional.of(new CpuPartitionLowering.LoweredPartition(ir,ids,bindings,
                    bindings.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                    java.util.Collections.nCopies(ids.size(),aggregate.dataType()),
                    nodes.stream().filter(n -> n != root).map(n -> n.outputs().getFirst()).toList(),
                    base.extents(),base.elementCount(),"legal: private exact softmax backward two-pass composition",new long[0],
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                    Optional.empty(),base.aggregateGeometry()));
        } catch (IllegalArgumentException | ArithmeticException notProved) { return Optional.empty(); }
    }

    /**
     * Composes existing pointwise lowering around one ordinary static exact SUM/MEAN.
     * Complete projected uses protect publication, saved values and outside consumers before
     * any resource is removed. A failed proof leaves the established split route untouched.
     * @param <I> retained immutable analysis-input type
     * @param context complete candidate projection with outside uses protected
     * @param pointwise existing pointwise lowerer; reused for normalization and typed sites
     * @return complete fused unit, or empty if any boundary is not proved private/admitted
     * @throws NullPointerException if either argument is null
     * @throws ArithmeticException if checked existing geometry overflows
     */
    <I extends BackendAnalysisInputs> Optional<CpuPartitionLowering.LoweredPartition> fuse(
            PrepareContext<I> context, CpuPartitionLowering pointwise) {
        var nodes = context.nodes();
        if (context.requiresBatchSize() || nodes.size() < 2 || nodes.size() > 8) return Optional.empty();
        int anchor = -1;
        for (int i = 0; i < nodes.size(); i++) if (nodes.get(i).operation().kind() instanceof AggregateReductionKind) {
            if (anchor >= 0) return Optional.empty(); anchor = i;
        }
        if (anchor < 0) return Optional.empty();
        var reduction = nodes.get(anchor);
        if (reduction.operation().kind() != AggregateReductionKind.SUM
                && reduction.operation().kind() != AggregateReductionKind.MEAN
                || reduction.operation().attrs() instanceof SumToShapeAttrs) return Optional.empty();
        try {
            var base = lower(CpuPartitionDagDecomposer.project(context, List.of(reduction), context.backendInputs()));
            var exact = (CpuAggregateIr) base.portableKernelIr();
            if (exact.dataType() != DataType.FLOAT32 && exact.dataType() != DataType.FLOAT64) return Optional.empty();
            var memory = new LinkedHashMap<ValueId, io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement>();
            context.memoryRequirements().forEach(r -> memory.put(r.valueId(), r));
            // Each removed producing occurrence has exactly one next use, including both
            // operand occurrences of x+x. No replay, fan-out or protected store is admitted.
            for (int i = 0; i < nodes.size()-1; i++) {
                var node = nodes.get(i);
                if (node.outputs().size() != 1) return Optional.empty();
                var id = node.outputs().getFirst(); var requirement = memory.get(id);
                var uses = context.partitionDag().consumers(id);
                if (requirement == null || requirement.graphOutput()
                        || requirement.consumerPartitions().stream().anyMatch(p -> !p.equals(context.partition()))
                        || uses.size() != 1 || uses.getFirst().node() != nodes.get(i+1)) return Optional.empty();
            }
            var values = new LinkedHashMap<ValueId, GraphValue>(); context.values().forEach(v -> values.put(v.id(),v));
            Shape inputShape = require(values,reduction.inputs().getFirst()).descriptor().shape();
            for (int i = 0; i < anchor; i++) if (!require(values,nodes.get(i).outputs().getFirst())
                    .descriptor().shape().equals(inputShape)) return Optional.empty();
            var tail = nodes.subList(anchor+1,nodes.size());
            if (tail.size() > 2) return Optional.empty();
            Shape resultShape = require(values,reduction.outputs().getFirst()).descriptor().shape();
            for (int i = 0; i < tail.size(); i++) {
                if (tail.get(i).outputs().size() != 1
                        || !require(values,tail.get(i).outputs().getFirst()).descriptor().shape().equals(resultShape))
                    return Optional.empty();
                var kind = tail.get(i).operation().kind();
                if (!(kind == io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD
                        && i == 0 || kind == io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.RELU
                        && i == tail.size()-1)) return Optional.empty();
            }
            var producer = anchor == 0 ? null : pointwise.lower(CpuPartitionDagDecomposer.project(
                    context, nodes.subList(0,anchor), context.backendInputs()));
            var epilogue = tail.isEmpty() ? null : pointwise.lower(CpuPartitionDagDecomposer.project(
                    context, tail, context.backendInputs()));
            for (var part : new CpuPartitionLowering.LoweredPartition[]{producer,epilogue}) if (part != null) {
                if (!(part.portableKernelIr() instanceof CpuKernelIr ir) || ir.stores().size() != 1
                        || ir.values().stream().anyMatch(v -> v.dataType() != exact.dataType())
                        || ir.instructions().stream().anyMatch(i -> i.opcode()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST)
                        || !Arrays.equals(require(values,part.boundaryValues().getLast()).descriptor().layout().orElseThrow().strides(),
                            LayoutDescriptor.contiguous(require(values,part.boundaryValues().getLast()).descriptor().shape()).strides()))
                    return Optional.empty();
            }
            var ids = new ArrayList<ValueId>(); var bindings = new ArrayList<CpuAccessPlan.Binding>();
            var spans = new ArrayList<Long>(); var pi = new ArrayList<Integer>(); var ti = new ArrayList<Integer>();
            var reads = producer == null ? base : producer;
            for (int i = 0; i < reads.boundaryValues().size()-1; i++) {
                pi.add(ids.size()); ids.add(reads.boundaryValues().get(i));
                bindings.add(reads.accessBindings().get(i)); spans.add(reads.referencedElementSpans().get(i));
            }
            if (epilogue != null) for (int i = 0; i < epilogue.boundaryValues().size()-1; i++) {
                var id = epilogue.boundaryValues().get(i);
                if (id.equals(reduction.outputs().getFirst())) { ti.add(-1); continue; }
                int position = ids.indexOf(id);
                if (position < 0) {
                    position = ids.size(); ids.add(id); bindings.add(epilogue.accessBindings().get(i));
                    spans.add(epilogue.referencedElementSpans().get(i));
                }
                ti.add(position);
            }
            var result = epilogue == null ? base : epilogue;
            ids.add(result.boundaryValues().getLast()); bindings.add(result.accessBindings().getLast());
            spans.add(result.referencedElementSpans().getLast());
            if (ids.subList(0,ids.size()-1).stream().anyMatch(id -> !canonicalBroadcastRead(require(values,id)))
                    || !Arrays.equals(require(values,ids.getLast()).descriptor().layout().orElseThrow().strides(),
                        LayoutDescriptor.contiguous(require(values,ids.getLast()).descriptor().shape()).strides())) return Optional.empty();
            var ir = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr(exact,
                    producer == null ? Optional.empty() : Optional.of(producer.kernelIr()),
                    producer == null ? List.of() : pi, producer == null ? List.of() : producer.accessBindings(),
                    producer == null ? 0 : -1,
                    epilogue == null ? Optional.empty() : Optional.of(epilogue.kernelIr()), ti,
                    epilogue == null ? List.of() : epilogue.accessBindings(), bindings,
                    Arrays.stream(inputShape.toLongArray()).boxed().toList());
            var virtual = nodes.subList(0,nodes.size()-1).stream().map(n -> n.outputs().getFirst()).toList();
            return Optional.of(new CpuPartitionLowering.LoweredPartition(ir,ids,bindings,spans,
                    java.util.Collections.nCopies(ids.size(),exact.dataType()),virtual,base.extents(),base.elementCount(),
                    "legal: private rounded pointwise sites around exact SUM/MEAN",new long[0],
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                    Optional.empty(),base.aggregateGeometry()));
        } catch (IllegalArgumentException notProved) { return Optional.empty(); }
    }

    /**
     * Authenticates a resolved canonical read, including an existing right-aligned broadcast.
     * Normalizer STRIDED roles alone cannot distinguish a canonical [R,1] read broadcast to
     * [R,F] from arbitrary striding. Zero-stride axes instead reconstruct the canonical source
     * with extent one; nonzero axes must retain its exact dense strides.
     * @param value nonnull static graph read with a resolved layout
     * @return true only for canonical source strides or their zero-stride broadcast extension
     * @throws ArithmeticException if checked canonical stride arithmetic overflows
     */
    private static boolean canonicalBroadcastRead(GraphValue value) {
        var shape = value.descriptor().shape(); var layout = value.descriptor().layout().orElseThrow();
        long[] strides = layout.strides(), extents = shape.toLongArray();
        if (layout.storageOffset() < 0) return false;
        if (Arrays.equals(strides,LayoutDescriptor.contiguous(shape).strides())) return true;
        long expected = 1;
        for (int axis = extents.length-1; axis >= 0; axis--) {
            if (strides[axis] == 0) continue;
            if (strides[axis] != expected) return false;
            expected = Math.multiplyExact(expected,extents[axis]);
        }
        return true;
    }

    /**
     * Evaluates one static SUM from an immutable exact typed positive-one splat fact.
     *
     * <p>The source may feed SUM directly or through EXPAND followed optionally by CONTIGUOUS.
     * Every absorbed affine output must be private and have exactly its next use; final SUM
     * publication is allowed. Selected axes and output geometry come from ordinary aggregate
     * lowering, including empty and no-axis domains. Checked affine-prefix products and byte
     * spans remain admission guards even when the corresponding storage is removed. Original
     * source initialization is not virtualized, and no caller storage is inspected.</p>
     * @param <I> retained immutable backend analysis-input type
     * @param context non-null complete selected unit projection with outside uses protected
     * @return output-only lowering when proved, otherwise empty without changing the fallback
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if existing ordinary SUM geometry cannot be lowered
     * @throws ArithmeticException if existing checked geometry or byte-size arithmetic overflows
     */
    <I extends BackendAnalysisInputs> Optional<CpuPartitionLowering.LoweredPartition> foldStaticTypedOne(
            PrepareContext<I> context) {
        var nodes = context.nodes();
        if (context.requiresBatchSize() || nodes.isEmpty() || nodes.size() > 3
                || nodes.getLast().operation().kind() != AggregateReductionKind.SUM) return Optional.empty();
        var values = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(v -> values.put(v.id(), v));
        ValueId source = nodes.getFirst().inputs().getFirst();
        ScalarValue fact = context.constants().get(source);
        if (fact == null || !fact.equals(positiveOne(fact.dataType()))) return Optional.empty();
        var memory = new LinkedHashMap<ValueId, io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement>();
        context.memoryRequirements().forEach(r -> memory.put(r.valueId(), r));
        ValueId previous = source;
        for (int index = 0; index < nodes.size(); index++) {
            var node = nodes.get(index);
            if (node.inputs().size() != 1 || node.outputs().size() != 1
                    || !node.inputs().getFirst().equals(previous)) return Optional.empty();
            var input = require(values, previous);
            var output = require(values, node.outputs().getFirst());
            if (!input.descriptor().shape().isFullyStatic() || !output.descriptor().shape().isFullyStatic()
                    || input.descriptor().dataType() != fact.dataType()
                    || output.descriptor().dataType() != fact.dataType()) return Optional.empty();
            if (!capabilities.supports(new OperationCapabilityQuery(node.operation(),
                    List.of(input.descriptor()), List.of(output.descriptor())))) return Optional.empty();
            // Preserve checked referenced-span/byte and access admission even for removed values.
            for (var value : List.of(input, output)) {
                var access = binding(layout(value), CpuAccessPlan.AccessKind.READ);
                Math.multiplyExact(access.referencedElementSpan(), fact.dataType().byteWidth());
            }
            if (index + 1 < nodes.size()) {
                // Match the affine lowerer's left-to-right checked logical product, even
                // when a later zero makes aggregate geometry empty. Never enumerate it.
                for (var value : List.of(input, output)) {
                    long logicalCount = 1;
                    for (long extent : value.descriptor().shape().toLongArray())
                        logicalCount = Math.multiplyExact(logicalCount, extent);
                }
                Object kind = node.operation().kind();
                if (!(index == 0 && kind == ShapeTransformKind.EXPAND
                        || index == 1 && nodes.size() == 3 && kind == ContiguousKind.CONTIGUOUS)) return Optional.empty();
                var id = output.id(); var obligation = memory.get(id);
                var uses = context.partitionDag().consumers(id);
                if (obligation == null || obligation.graphOutput()
                        || obligation.consumerPartitions().stream().anyMatch(p -> !p.equals(context.partition()))
                        || uses.size() != 1 || uses.getFirst().node() != nodes.get(index + 1)) return Optional.empty();
            }
            previous = output.id();
        }
        var sum = nodes.getLast();
        var aggregate = lower(CpuPartitionDagDecomposer.project(context, List.of(sum), context.backendInputs()));
        var geometry = aggregate.aggregateGeometry().orElseThrow();
        var output = aggregate.accessBindings().getLast();
        var ir = new CpuConstantWriteIr(countImmediate(fact.dataType(), geometry.domainCount()), output.plan());
        return Optional.of(new CpuPartitionLowering.LoweredPartition(ir, List.of(previous),
                List.of(output), List.of(aggregate.referencedElementSpans().getLast()),
                List.of(fact.dataType()), nodes.subList(0, nodes.size() - 1).stream()
                    .map(n -> n.outputs().getFirst()).toList(), output.extents().stream().mapToLong(Long::longValue).toArray(),
                geometry.outputCount(), "legal: explicit static typed-one count; output stores only", new long[0]));
    }

    /**
     * Rounds a nonnegative exact integer directly to the represented CPU SUM type. This mirrors
     * the exact accumulator's final ties-to-even rounding, not long-to-float-to-low conversion.
     * Integer results use modular bits; count/shape overflow is checked before this seam.
     * @param type non-null supported homogeneous numeric SUM type
     * @param count nonnegative checked contributor count, at most Long.MAX_VALUE
     * @return exact typed result bits, with positive zero for the empty domain
     * @throws IllegalArgumentException if the count is negative or the type is BOOL
     * @throws NullPointerException if {@code type} is {@code null}
     */
    static CpuKernelIr.ScalarImmediate countImmediate(DataType type, long count) {
        if (count < 0 || type == DataType.BOOL) throw new IllegalArgumentException("unsupported count");
        if (type == DataType.INT64) return new CpuKernelIr.ScalarImmediate(type, count);
        if (type == DataType.INT32) return new CpuKernelIr.ScalarImmediate(type, count & 0xffff_ffffL);
        int fraction = type == DataType.FLOAT64 ? 52 : type == DataType.FLOAT32 ? 23
                : type == DataType.FLOAT16 ? 10 : 7;
        int bias = type == DataType.FLOAT64 ? 1023 : type == DataType.FLOAT16 ? 15 : 127;
        if (count == 0) return new CpuKernelIr.ScalarImmediate(type, 0);
        int exponent = 63 - Long.numberOfLeadingZeros(count);
        int shift = Math.max(0, exponent - fraction);
        long significant = count >>> shift;
        if (shift > 0) {
            long remainder = count & ((1L << shift) - 1), halfway = 1L << (shift - 1);
            if (remainder > halfway || remainder == halfway && (significant & 1) != 0) significant++;
            if (significant == 1L << (fraction + 1)) { significant >>>= 1; exponent++; }
        } else significant <<= fraction - exponent;
        long bits = exponent > bias ? ((1L << (type == DataType.FLOAT64 ? 11 : type == DataType.FLOAT16 ? 5 : 8)) - 1) << fraction
                : ((long) exponent + bias) << fraction | significant & ((1L << fraction) - 1);
        return new CpuKernelIr.ScalarImmediate(type, bits);
    }

    /**
     * Names exact typed positive-one facts without any cross-type equality or payload sampling.
     * @param type non-null source representation type
     * @return exact immutable numeric +1 fact, or null for unsupported BOOL
     * @throws NullPointerException if {@code type} is {@code null}
     */
    private static ScalarValue positiveOne(DataType type) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(1); case FLOAT64 -> ScalarValue.float64(1);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3f80);
            case FLOAT16 -> ScalarValue.float16Bits((short) 0x3c00);
            case INT32 -> ScalarValue.int32(1); case INT64 -> ScalarValue.int64(1);
            case BOOL -> null;
        };
    }

    /**
     * Identifies the existing attribute form without selecting arithmetic or resource policy.
     * @param attrs non-null supported Model aggregate attributes
     * @return exact FULL, SINGLE_AXIS, MULTI_AXIS or SUM_TO_SHAPE form
     * @throws IllegalArgumentException if attributes are outside those forms
     */
    static CpuAggregateIr.Form reductionForm(Object attrs) {
        if (attrs == NoOperationAttrs.INSTANCE) return CpuAggregateIr.Form.FULL;
        if (attrs instanceof AxisReductionAttrs) return CpuAggregateIr.Form.SINGLE_AXIS;
        if (attrs instanceof MultiAxisReductionAttrs) return CpuAggregateIr.Form.MULTI_AXIS;
        if (attrs instanceof SumToShapeAttrs) return CpuAggregateIr.Form.SUM_TO_SHAPE;
        throw new IllegalArgumentException("unsupported aggregate attributes");
    }

    /**
     * Reads the Model keep-dimensions fact; target forms never retain selected axes.
     * @param attrs non-null supported Model aggregate attributes
     * @return whether an ordinary axis reduction retains selected extent-one axes
     * @throws IllegalArgumentException if attributes have no supported reduction form
     */
    static boolean keepDimensions(Object attrs) {
        reductionForm(attrs);
        return attrs instanceof AxisReductionAttrs axis ? axis.keepDimensions()
                : attrs instanceof MultiAxisReductionAttrs axes && axes.keepDimensions();
    }

    /**
     * Normalizes existing aggregate forms using exact Dimensions and validates the result Shape.
     * This shares facts only: callers independently select exact ordinary or sequential SUM math.
     * @param input source shape
     * @param output declared result shape
     * @param attrs exact Model attributes
     * @return immutable increasing selected axes; empty means no selected axis, not an arithmetic policy
     * @throws IllegalArgumentException if attributes, axes or target mapping disagree
     */
    static List<Integer> normalizedAxes(Shape input, Shape output, Object attrs) {
        var axes = new ArrayList<Integer>(); boolean keep = keepDimensions(attrs);
        if (attrs instanceof SumToShapeAttrs target) {
            if (!target.targetShape().equals(output) || output.rank() > input.rank()) throw new IllegalArgumentException("aggregate axes, target or output Shape disagree");
            int leading = input.rank() - output.rank();
            for (int axis = 0; axis < input.rank(); axis++) {
                if (axis < leading) axes.add(axis);
                else if (!input.dimension(axis).equals(output.dimension(axis-leading))) {
                    if (!output.dimension(axis-leading).equals(new StaticDimension(1))) throw new IllegalArgumentException("aggregate axes, target or output Shape disagree");
                    axes.add(axis);
                }
            }
            return List.copyOf(axes);
        }
        if (attrs == NoOperationAttrs.INSTANCE) for (int axis = 0; axis < input.rank(); axis++) axes.add(axis);
        else if (attrs instanceof AxisReductionAttrs axis) { axes.add(axis.axis()); keep = axis.keepDimensions(); }
        else if (attrs instanceof MultiAxisReductionAttrs multi) { axes.addAll(multi.axes()); keep = multi.keepDimensions(); }
        else throw new IllegalArgumentException("aggregate axes, target or output Shape disagree");
        axes.sort(Integer::compare);
        if (new HashSet<>(axes).size() != axes.size() || axes.stream().anyMatch(n -> n < 0 || n >= input.rank())) throw new IllegalArgumentException("aggregate axes, target or output Shape disagree");
        var dimensions = new ArrayList<Dimension>();
        for (int axis = 0; axis < input.rank(); axis++) {
            if (!axes.contains(axis)) dimensions.add(input.dimension(axis)); else if (keep) dimensions.add(new StaticDimension(1));
        }
        if (!Shape.ofDimensions(dimensions.toArray(Dimension[]::new)).equals(output)) throw new IllegalArgumentException("aggregate axes, target or output Shape disagree");
        return List.copyOf(axes);
    }

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
        CpuAggregateIr.Form form = reductionForm(attrs);
        boolean keep = keepDimensions(attrs);
        int[] selected = normalizedAxes(input.descriptor().shape(),output.descriptor().shape(),attrs)
                .stream().mapToInt(Integer::intValue).toArray();
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
                    || type == DataType.BFLOAT16 || type == DataType.FLOAT16);
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
        int precision = type == DataType.FLOAT64 ? 53 : type == DataType.FLOAT32 ? 24
                : type == DataType.FLOAT16 ? 11 : 8;
        if (kind == CpuAggregateIr.Kind.PROD) {
            int maximumExponentMagnitude = type == DataType.FLOAT64 ? 1074
                    : type == DataType.FLOAT32 ? 149 : type == DataType.FLOAT16 ? 24 : 133;
            Math.multiplyExact((long) maximumExponentMagnitude, Math.max(1L, domainCount));
            long bits = Math.multiplyExact((long) precision, Math.max(1L, domainCount));
            return Math.toIntExact(Math.max(1L, Math.addExact(bits, 63L) / 64L));
        }
        int emin = type == DataType.FLOAT64 ? -1074
                : type == DataType.FLOAT32 ? -149 : type == DataType.FLOAT16 ? -24 : -133;
        int emax = type == DataType.FLOAT64 ? 1023
                : type == DataType.FLOAT16 ? 15 : 127;
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
     * @param ordinaryBatch whether the qualified typed path retains five F32 or 34 F64 limbs even at copy entry
     */
    public record Geometry(CpuAggregateIr.Kind kind, DataType dataType, CpuAggregateIr.Form form,
            int[] selectedAxes, boolean keepDimensions, Layout input, Layout output,
            long outputCount, long domainCount, int stateLimbCount, long scratchSliceBytes,
            boolean ordinaryBatch) {
        /**
         * Immutable coefficient-only ordinary recipe; no graph, descriptor or invocation storage
         * survives lowering. The structural IR is reused for every actual batch.
         * @param ir authenticated structural ordinary IR
         * @param boundaries unique input/output IDs in carrier order
         * @param boundaryTypes exact represented type of every boundary
         * @param bindings numeric extent/stride coefficients in boundary order
         * @param iterationCount affine independent-range count
         * @param scratchBytes checked affine reserved scratch bytes
         */
        public record BatchGeometry(CpuAggregateIr ir, List<ValueId> boundaries,
                List<io.github.pho001.synaptik.model.datatype.DataType> boundaryTypes,
                List<CpuAccessPlan.BatchBinding> bindings, BatchByteSize iterationCount,
                BatchByteSize scratchBytes) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots the authenticated structural/numeric recipe without retaining a graph.
             * @param ir non-null ordinary structural IR
             * @param boundaries unique ordered boundary IDs, copied
             * @param boundaryTypes exact represented boundary types, copied
             * @param bindings complete physical coefficient recipes, copied
             * @param iterationCount checked affine capacity in complete helper ranges
             * @param scratchBytes checked affine scratch reservation in bytes
             * @throws NullPointerException if a required reference is null
             * @throws IllegalArgumentException if boundary cardinalities disagree
             */
            public BatchGeometry {
                Objects.requireNonNull(ir,"ir"); boundaries = List.copyOf(boundaries);
                boundaryTypes = List.copyOf(boundaryTypes); bindings = List.copyOf(bindings);
                Objects.requireNonNull(iterationCount,"iterationCount"); Objects.requireNonNull(scratchBytes,"scratchBytes");
                if (boundaries.size() != bindings.size() || boundaries.size() != boundaryTypes.size())
                    throw new IllegalArgumentException("batch boundary cardinality disagrees");
            }
            /** @return original zero-based coordinate cells restored before every helper call */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset> resets() {
                int rank = ir.inputAccess().iterationRank(), count = rank+ir.outputAccess().iterationRank();
                return java.util.stream.IntStream.range(0,count).mapToObj(i ->
                        new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset(11+rank+i,0)).toList();
            }
            /** @return deterministic coefficient identity, excluding actual B and storage */
            @Override public String identity() { return ir.structuralKey()+":bindings="+bindings+":ranges="+iterationCount+":scratch="+scratchBytes; }
            /**
             * Resolves existing concrete family geometry exactly once during cold binding.
             * @param batch non-negative actual int-representable batch
             * @return ordinary typed lowering with numeric geometry and unchanged structural key
             * @throws IllegalArgumentException if actual numerical domain policy is violated
             * @throws ArithmeticException if checked invocation geometry overflows
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var actual = bindings.stream().map(b -> b.bind(batch)).toList();
                scratchBytes.byteSize(batch);
                var inputBinding = actual.get(0);
                var input = new Layout(inputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        inputBinding.baseElementOffset(),inputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                var outputBinding = actual.get(1);
                var output = new Layout(outputBinding.extents().stream().mapToLong(Long::longValue).toArray(),
                        outputBinding.baseElementOffset(),outputBinding.effectiveStrides().stream().mapToLong(Long::longValue).toArray());
                int leading = input.extents.length-output.extents.length;
                int[] axes = Arrays.stream(ir.selectedAxes()).filter(a -> ir.form() != CpuAggregateIr.Form.SUM_TO_SHAPE
                        || a < leading || input.extents[a] != output.extents[a-leading]).toArray();
                long domain = 1; for (int axis : axes) domain = Math.multiplyExact(domain,input.extents[axis]);
                boolean exact = ir.workspacePolicy() == CpuAggregateIr.EXACT_FLOATING_STATE;
                boolean boundedSum = exact && (ir.kind() == CpuAggregateIr.Kind.SUM || ir.kind() == CpuAggregateIr.Kind.MEAN)
                        && (ir.dataType() == DataType.FLOAT32 || ir.dataType() == DataType.FLOAT64)
                        && ir.stateLimbCount() == (ir.dataType() == DataType.FLOAT64 ? 34 : 5);
                int limbs = boundedSum ? ir.stateLimbCount() : exact && (ir.form() != CpuAggregateIr.Form.SUM_TO_SHAPE || axes.length != 0)
                        ? CpuAggregateLowering.stateLimbCount(ir.kind(),ir.dataType(),domain) : 0;
                long slice = limbs == 0 ? 0 : Math.addExact(ir.kind() == CpuAggregateIr.Kind.PROD ? 24 : 8,Math.multiplyExact(8L,limbs));
                long work = iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(),ir.dataType(),ir.form(),axes,ir.keepDimensions(),input,output,
                        work,domain,limbs,slice,boundedSum);
                return new CpuPartitionLowering.LoweredPartition(ir,boundaries,actual,
                        actual.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes,List.of(),new long[] {work},work,"legal: bound ordinary batch geometry",new long[0],
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.of(geometry));
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
            int inputRank = input.extents.length, outputRank = output.extents.length;
            int coordinates = 11 + inputRank;
            int layout = coordinates + inputRank + outputRank;
            var resets = new ArrayList<ResetCell>(inputRank + outputRank);
            for (int i = 0; i < inputRank + outputRank; i++)
                resets.add(new ResetCell(coordinates + i, 0));
            return new TemporalRebind(List.of(new BaseCell(layout + 1, 0, input.offset),
                    new BaseCell(layout + 3 + 2 * inputRank, 1, output.offset)), resets);
        }

        /**
         * Describes the appended raw physical bases of the ordinary fused aggregate pack.
         * Header layouts are virtual in this form and must not be rebound to physical carriers.
         * @param start non-negative inclusive helper range start
         * @param boundaryCount number of unique physical reads followed by the final write, at least two
         * @return immutable appended base facts and the same original cursor reset facts
         * @throws IllegalArgumentException if start is negative or boundaryCount is less than two
         * @throws ArithmeticException if checked pack indexing overflows
         */
        public TemporalRebind temporalRebindFusion(long start, int boundaryCount) {
            if (boundaryCount < 2) throw new IllegalArgumentException("fusion requires physical reads and output");
            var ordinary = temporalRebind(start);
            int header = 15 + 4 * input.extents.length + 3 * output.extents.length;
            var bases = new ArrayList<BaseCell>(boundaryCount);
            for (int i = 0; i < boundaryCount; i++)
                bases.add(new BaseCell(Math.addExact(header, i), i, 0));
            return new TemporalRebind(bases, ordinary.resets());
        }

        /**
         * Retains the static geometry constructor and its existing exact-state policy.
         * @param kind non-null fold kind
         * @param dataType non-null represented type
         * @param form non-null attribute form
         * @param selectedAxes increasing normalized membership, copied
         * @param keepDimensions whether selected axes remain extent one
         * @param input non-null resolved read layout
         * @param output non-null resolved injective write layout
         * @param outputCount actual output cells
         * @param domainCount actual contributors per cell
         * @param stateLimbCount exact state limbs, otherwise zero
         * @param scratchSliceBytes exact state bytes per range, otherwise zero
         */
        public Geometry(CpuAggregateIr.Kind kind, DataType dataType, CpuAggregateIr.Form form,
                int[] selectedAxes, boolean keepDimensions, Layout input, Layout output,
                long outputCount, long domainCount, int stateLimbCount, long scratchSliceBytes) {
            this(kind, dataType, form, selectedAxes, keepDimensions, input, output, outputCount,
                    domainCount, stateLimbCount, scratchSliceBytes, false);
        }
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
         * @param stateLimbCount exact floating state limbs, otherwise zero; ordinary batch
         *     SUM/MEAN retains five for FLOAT32 or 34 for FLOAT64, including empty/singleton domains
         * @param scratchSliceBytes exact bytes per floating state slice, otherwise zero;
         *     ordinary batch retains 48/280 typed bytes; empty output needs zero total workspace
         * @param ordinaryBatch whether to enforce bounded typed SUM/MEAN constant-limb state
         *     even at singleton/copy entry; static domain-sized policy remains unchanged
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
                        || dataType == DataType.BFLOAT16 || dataType == DataType.FLOAT16);
            if (ordinaryBatch && (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64
                    || kind != CpuAggregateIr.Kind.SUM && kind != CpuAggregateIr.Kind.MEAN)) {
                throw new IllegalArgumentException("unsupported batch aggregate state");
            }
            int expectedLimbs = ordinaryBatch ? dataType == DataType.FLOAT64 ? 34 : 5 : exactFloating ? CpuAggregateLowering.stateLimbCount(
                    kind, dataType, domainCount) : 0;
            long expectedSlice = ordinaryBatch ? dataType == DataType.FLOAT64 ? 280 : 48 : exactFloating ? Math.addExact(
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
        /**
         * Packs the unchanged exact-state header plus explicit physical bases for a fused unit.
         * Removed producer/result layouts have no Runtime carrier. Appended bases are in unique
         * read/final-write order, and every generated access adds its own normalized offset.
         * @param bases all physical element bases, read but not retained; at least two
         * @return fresh run-owned geometry with scratch slice zero and appended boundary bases
         * @throws NullPointerException if bases is null
         * @throws IllegalArgumentException if fewer than two physical bases are supplied
         * @throws ArithmeticException if checked geometry or layout arithmetic overflows
         */
        public long[] packFusion(long[] bases) {
            if (bases.length < 2) throw new IllegalArgumentException("fusion requires physical reads and output");
            var header = pack(new long[]{0,0},0);
            var result = Arrays.copyOf(header,Math.addExact(header.length,bases.length));
            System.arraycopy(bases,0,result,header.length,bases.length);
            return result;
        }
        private static int packLayout(long[] target, int x, Layout layout, long base) {
            target[x++] = layout.extents.length; target[x++] = Math.addExact(base, layout.offset);
            for (long extent : layout.extents) target[x++] = extent;
            for (long stride : layout.strides) target[x++] = stride;
            return x;
        }
    }
}
