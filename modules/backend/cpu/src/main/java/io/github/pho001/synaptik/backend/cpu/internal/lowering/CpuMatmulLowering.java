package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Arrays;
import java.util.Objects;

/**
 * Cold, checked normalization of static and single-named-B MATMUL invocation geometry.
 *
 * <p>Rank-one operands receive internal unit M or N axes. Leading batch axes are aligned from
 * the right and absent or expanded singleton coordinates receive effective stride zero. The
 * record retains no graph, operation, carrier, worker, policy, or allocation state.</p>
 *
 * <p>One or two adjacent private rank-two transposes may be consumed as read mappings of their
 * original backing values. This removes only their copies, not arithmetic or observable values;
 * existing MATMUL and optional epilogue emitters receive the same checked geometry protocol.</p>
 */
public final class CpuMatmulLowering {
    /**
     * Prepared rank-two FLOAT32 matrix geometry; M, K and N bind from numeric access recipes.
     * Forward B/F/H and parameter-adjoint F/B/H use the same unchanged packed invocation ABI.
     * @param dataType exact homogeneous FLOAT32 represented type
     * @param left non-null prepared read mapping
     * @param right non-null prepared read mapping
     * @param result non-null prepared canonical write mapping
     */
    public record BatchGeometry(DataType dataType, CpuAccessPlan.BatchBinding left,
            CpuAccessPlan.BatchBinding right, CpuAccessPlan.BatchBinding result) {
        /**
         * Proves matrix dimensions structurally, without substituting a representative B.
         * @param dataType homogeneous FLOAT32 type
         * @param left non-null rank-two read recipe
         * @param right non-null rank-two read recipe
         * @param result non-null rank-two canonical write recipe
         * @throws NullPointerException if a reference is null
         * @throws IllegalArgumentException if type, direction, rank or matrix dimensions disagree
         */
        public BatchGeometry {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(left, "left"); Objects.requireNonNull(right, "right");
            Objects.requireNonNull(result, "result");
            if (dataType != DataType.FLOAT32 || left.plan().iterationRank() != 2
                    || right.plan().iterationRank() != 2 || result.plan().iterationRank() != 2
                    || left.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                    || right.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                    || result.plan().accessKind() != CpuAccessPlan.AccessKind.WRITE
                    || !sameExtent(left, 1, right, 0) || !sameExtent(left, 0, result, 0)
                    || !sameExtent(right, 1, result, 1)) {
                throw new IllegalArgumentException("ordinary batch MATMUL geometry disagrees");
            }
        }

        private static boolean sameExtent(CpuAccessPlan.BatchBinding a, int x,
                CpuAccessPlan.BatchBinding b, int y) {
            return a.extentConstants().get(x).equals(b.extentConstants().get(y))
                    && a.extentBatchFactors().get(x).equals(b.extentBatchFactors().get(y));
        }

        /**
         * Resolves numeric matrix geometry once; no Model lowerer or injectivity scan is run.
         * @param batchSize non-negative actual B within CPU int geometry limits
         * @return fresh resolved matrix geometry with actual transposed strides and contraction K
         * @throws IllegalArgumentException if B is negative
         * @throws ArithmeticException if checked geometry exceeds representability limits
         */
        public Geometry bind(long batchSize) {
            var a = left.bind(batchSize); var b = right.bind(batchSize); var c = result.bind(batchSize);
            return new Geometry(dataType, dataType, dataType, new long[0], new long[0],
                    new long[0], new long[0], 1, a.extents().get(0), a.extents().get(1),
                    b.extents().get(1), false, false, a.baseElementOffset(), b.baseElementOffset(),
                    c.baseElementOffset(), a.effectiveStrides().get(0), a.effectiveStrides().get(1),
                    b.effectiveStrides().get(0), b.effectiveStrides().get(1),
                    c.effectiveStrides().get(0), c.effectiveStrides().get(1), c.elementCount(),
                    a.referencedElementSpan(), b.referencedElementSpan(), c.referencedElementSpan());
        }
    }
    /** Creates a stateless geometry lowerer. */
    public CpuMatmulLowering() { }

    /**
     * Checks the common rank-two family equation using exact Dimension identity/value, never
     * a representative batch size. Dtype, layout and route admission remain caller-owned;
     * in particular this check does not relax ordinary batch's FLOAT32-only coefficient guards.
     * @param left non-null rank-two left shape, with axes M/K
     * @param right non-null rank-two right shape, with axes K/N
     * @param result non-null rank-two result shape, with axes M/N
     * @throws NullPointerException if any shape is null
     * @throws IllegalArgumentException if ranks or exact dimension equations disagree
     */
    static void validateRankTwo(Shape left, Shape right, Shape result) {
        Objects.requireNonNull(left,"left"); Objects.requireNonNull(right,"right");
        Objects.requireNonNull(result,"result");
        if (left.rank() != 2 || right.rank() != 2 || result.rank() != 2
                || !left.dimension(1).equals(right.dimension(0))
                || !left.dimension(0).equals(result.dimension(0))
                || !right.dimension(1).equals(result.dimension(1)))
            throw new IllegalArgumentException("rank-two MATMUL dimensions disagree");
    }

    /**
     * Lowers one MATMUL with optional proved private transpose inputs, but no arithmetic suffix,
     * into the existing scalar or bounded tiled/vector form selected by cold compute preference.
     * The recognition-fact overload separately supplies any bias/terminal formula.
     *
     * @param context non-null MATMUL projection with at most two private rank-two transpose inputs
     * @return immutable lowering with exact boundary bindings and MATMUL geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the unit is not one exact bounded MATMUL chain or
     *     lacks CPU-specific cold execution inputs
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerRecognized(context,CpuMatmulIr.Epilogue.none(),false, null);
    }

    /**
     * Selects the qualified direct-scalar realization for a controlled ordinary helper before
     * its kernel, fingerprint, geometry and work range are retained. This reuses the ordinary
     * descriptor and epilogue proof without changing standalone scalar/vector candidate policy.
     * @param context non-null exact ordinary MATMUL member projection with CPU cold inputs
     * @param epilogue non-null already-proved literal bias and terminal formula
     * @param effectiveBoundaryValues non-null empty destination populated with logical operand,
     *     bias and result IDs in boundary order; absorbed transposes retain their output IDs here,
     *     while the returned physical boundaries retain their backing IDs
     * @return consistent direct-scalar lowering with the original physical read/write mappings
     * @throws NullPointerException if any argument is null
     * @throws IllegalArgumentException if the destination is not empty or membership, descriptors
     *     or epilogue disagree
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerControlled(
            PrepareContext<? extends BackendAnalysisInputs> context, CpuMatmulIr.Epilogue epilogue,
            List<ValueId> effectiveBoundaryValues) {
        Objects.requireNonNull(effectiveBoundaryValues, "effectiveBoundaryValues");
        if (!effectiveBoundaryValues.isEmpty())
            throw new IllegalArgumentException("effective boundary destination must be empty");
        return lowerRecognized(context, Objects.requireNonNull(epilogue, "epilogue"), true,
                effectiveBoundaryValues);
    }

    /**
     * Lowers an ordinary single-named-B MATMUL without choosing representative dimensions.
     * Original Model descriptor equations and purity are checked before numeric recipes exist.
     * Vector promotion, arbitrary right-aligned batch ranks and promoted numeric types retain
     * the ordinary scalar full-K implementation; no row axis or provenance is invented here.
     *
     * @param context non-null exact one-node MATMUL projection with fixed features and one named B
     * @return unresolved numeric geometry with an actual-B-independent direct scalar IR
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if semantics, numerical types or layouts are unsupported
     * @throws ArithmeticException if structural geometry exceeds checked CPU limits
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers one MATMUL using the full local composition's already-proved physical layouts.
     * This preserves affine views and nonleading row coordinates across ordinary unit cuts.
     * @param context non-null exact one-node ordinary MATMUL projection
     * @param layouts non-null full-local numeric layouts keyed by the original local value IDs
     * @return unresolved direct scalar recipe retaining the supplied offsets and strides
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if semantics, numerical types or physical layouts disagree
     * @throws ArithmeticException if checked coefficient geometry overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        Objects.requireNonNull(context, "context"); Objects.requireNonNull(layouts, "layouts");
        CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1)
            throw new IllegalArgumentException("symbolic MATMUL requires one ordinary occurrence");
        var node = context.nodes().getFirst();
        if (node.operation().kind() != MatmulKind.MATMUL
                || node.inputs().size() != 2 || node.outputs().size() != 1)
            throw new IllegalArgumentException("symbolic MATMUL boundary cardinality disagrees");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var ids = List.of(node.inputs().get(0), node.inputs().get(1), node.outputs().getFirst());
        var types = ids.stream().map(id -> require(values, id).descriptor().dataType()).toList();
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int position = 0; position < ids.size(); position++) bindings.add(batchBinding(
                require(values, ids.get(position)).descriptor(), layouts.get(ids.get(position)),
                position == 2 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
        var matmul = new CpuMatmulIr(types.get(0), types.get(1), types.get(2),
                CpuMatmulIr.Realization.DIRECT_SCALAR, CpuMatmulIr.Epilogue.none(), 0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,
                List.of(bindings.get(0).plan(), bindings.get(1).plan()), bindings.get(2).plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.SymbolicGeometry(
                matmul.encodedKernelIr(), matmul, ids, types, bindings,
                CpuPartitionLowering.batchElements(bindings.get(2)), new BatchByteSize(0, 0, 1)));
    }

    /**
     * Authenticates a supplied physical mapping against original symbolic logical extents.
     * @param descriptor non-null original logical boundary descriptor
     * @param layout exact full-local mapping; null is rejected as a missing physical proof
     * @param kind ordinary read or write permission for this occurrence
     * @return immutable checked mapping with unchanged offset/strides and requested permission
     * @throws IllegalArgumentException if the mapping is missing or its extents disagree
     * @throws ArithmeticException if structural coefficients exceed CPU geometry limits
     */
    private static CpuAccessPlan.BatchBinding batchBinding(TensorDescriptor descriptor,
            CpuAccessPlan.BatchBinding layout, CpuAccessPlan.AccessKind kind) {
        if (layout == null) throw new IllegalArgumentException("missing MATMUL physical layout");
        var canonical = CpuPartitionLowering.batchCanonical(descriptor, CpuAccessPlan.AccessKind.READ);
        if (!canonical.extentConstants().equals(layout.extentConstants())
                || !canonical.extentBatchFactors().equals(layout.extentBatchFactors()))
            throw new IllegalArgumentException("MATMUL physical layout extents disagree");
        var source = layout.plan();
        var plan = new CpuAccessPlan(kind, source.regime(), source.iterationRank(),
                source.axisRoles(), source.contiguousSuffix());
        return new CpuAccessPlan.BatchBinding(plan, layout.extentConstants(),
                layout.extentBatchFactors(), layout.baseElementOffset(), layout.strideConstants(),
                layout.strideBatchFactors());
    }

    /**
     * Lowers exactly one 0008C-associated executable alternative without rediscovering suffix
     * semantics from the graph.
     *
     * @param context non-null exact projected member occurrences in stable order, with
     *     {@link CpuPartitionAnalysisInputs} carrying the cold CPU compute preference
     * @param fact non-null recognized MATMUL fact whose literal epilogue is authoritative
     * @return immutable fused MATMUL lowering
     * @throws NullPointerException if {@code context} or {@code fact} is {@code null}
     * @throws IllegalArgumentException if CPU-specific preparation inputs are absent, or
     *     membership, chain boundaries, or descriptors disagree
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph.MatmulEpilogue fact) {
        Objects.requireNonNull(fact,"fact");
        if(context.nodes().size()!=fact.memberNodeOrdinals().size())throw new IllegalArgumentException(
                "recognized MATMUL membership disagrees");
        var source=fact.epilogue();
        var order=switch(source.addInputOrder()) {
            case NONE->CpuMatmulIr.Epilogue.AddInputOrder.NONE;
            case PRECEDING_LEFT->CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT;
            case PRECEDING_RIGHT->CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_RIGHT;};
        var terminal=CpuMatmulIr.Epilogue.Terminal.valueOf(source.terminal().name());
        return lowerRecognized(context,new CpuMatmulIr.Epilogue(order,terminal,
                source.clampRange().orElse(null)),false, null);
    }

    /**
     * Composes proved backing reads with the exact requested MATMUL epilogue, without mutating
     * graph descriptors or adding a runtime view step.
     *
     * @param context non-null exact projected member occurrences and CPU execution inputs
     * @param epilogue non-null already selected literal bias/terminal formula
     * @param controlled whether to select the existing qualified direct-scalar controlled form
     * @param effectiveBoundaryValues nullable controlled-only destination for effective logical
     *     boundary IDs before backing-source substitution; ordinary callers pass null
     * @return checked existing MATMUL IR, backing boundaries and invocation geometry
     * @throws IllegalArgumentException if prefix privacy, membership, descriptors or suffix disagree
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    private CpuPartitionLowering.LoweredPartition lowerRecognized(
            PrepareContext<? extends BackendAnalysisInputs> context, CpuMatmulIr.Epilogue epilogue,
            boolean controlled, List<ValueId> effectiveBoundaryValues) {
        Objects.requireNonNull(context, "context");
        if (!(context.backendInputs() instanceof CpuPartitionAnalysisInputs cpuInputs)) {
            throw new IllegalArgumentException("MATMUL requires CPU preparation inputs");
        }
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        int anchor = 0;
        while (anchor < context.nodes().size()
                && context.nodes().get(anchor).operation().kind() == AxisTransformKind.PERMUTE) anchor++;
        if (anchor > 2 || anchor == context.nodes().size()
                || context.nodes().size() - anchor > 3
                || context.nodes().get(anchor).operation().kind() != MatmulKind.MATMUL
                || privateTransposePrefixStart(context, values, anchor) != 0) {
            throw new IllegalArgumentException("MATMUL unit requires one bounded chain");
        }
        var node = context.nodes().get(anchor);
        if (node.inputs().size() != 2 || node.outputs().size() != 1)
            throw new IllegalArgumentException("MATMUL boundary cardinality disagrees");
        var inputIds = new ArrayList<ValueId>(node.inputs());
        var readDescriptors = new ArrayList<TensorDescriptor>();
        for (int role = 0; role < 2; role++) {
            ValueId id = inputIds.get(role);
            TensorDescriptor descriptor = require(values, id).descriptor();
            for (int prefix = 0; prefix < anchor; prefix++) {
                var transpose = context.nodes().get(prefix);
                if (!transpose.outputs().getFirst().equals(id)) continue;
                ValueId source = transpose.inputs().getFirst();
                var sourceLayout = require(values, source).descriptor().layout().orElseThrow();
                descriptor = new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                        Optional.of(LayoutDescriptor.of(descriptor.shape(),
                                new long[] {sourceLayout.stride(1), sourceLayout.stride(0)},
                                sourceLayout.storageOffset(), true)), descriptor.requiresGrad());
                inputIds.set(role, source);
            }
            readDescriptors.add(descriptor);
        }
        ValueId matmulOutput=node.outputs().getFirst();ValueId current=matmulOutput;
        ValueId biasId=null;
        int suffix=anchor+1;
        if(epilogue.hasBias()) {
            var add=context.nodes().get(suffix++);
            if(add.operation().kind()!=io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD
                    ||add.inputs().size()!=2||add.outputs().size()!=1)
                throw new IllegalArgumentException("MATMUL bias ADD cardinality disagrees");
            int preceding=epilogue.addInputOrder()==CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT?0:1;
            if(!add.inputs().get(preceding).equals(current))throw new IllegalArgumentException(
                    "recognized MATMUL ADD order disagrees");
            biasId=add.inputs().get(1-preceding);
            current=add.outputs().getFirst();
        }
        if(epilogue.hasTerminal()) {
            var terminalNode=context.nodes().get(suffix++);
            if(terminalNode.inputs().size()!=1||terminalNode.outputs().size()!=1
                    ||!terminalNode.inputs().getFirst().equals(current))throw new IllegalArgumentException(
                            "MATMUL terminal does not consume the suffix");
            current=terminalNode.outputs().getFirst();
        }
        if(suffix!=context.nodes().size())throw new IllegalArgumentException("MATMUL suffix is not linear");
        GraphValue result = require(values,current);
        Geometry geometry = lower(readDescriptors.get(0), readDescriptors.get(1), result.descriptor());
        if(biasId!=null){GraphValue bias=require(values,biasId);long[] be=bias.descriptor().shape().toLongArray();
            if(be.length!=1||be[0]!=geometry.n()||bias.descriptor().dataType()!=geometry.resultType())
                throw new IllegalArgumentException("MATMUL bias descriptor disagrees");}
        var mutableIds=new ArrayList<ValueId>(inputIds);
        if(biasId!=null)mutableIds.add(biasId);mutableIds.add(current);
        List<ValueId> ids=List.copyOf(mutableIds);
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int i = 0; i < ids.size(); i++) bindings.add(binding(
                i < 2 ? readDescriptors.get(i) : require(values, ids.get(i)).descriptor(),
                i == ids.size() - 1 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
        boolean vectorRequested = cpuInputs.portableExecution().computePreference()
                == ComputePreference.VECTOR_IF_ELIGIBLE;
        int bits = vectorRequested || ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()
                ? preferredBits(geometry.resultType()) : 0;
        int lanes = bits / geometry.resultType().bitWidth();
        var realization = controlled ? CpuMatmulIr.Realization.DIRECT_SCALAR
                : new CpuMatmulCandidateSelector().select(
                new CpuMatmulCandidateSelector.Facts(geometry.leftType(), geometry.rightType(),
                        geometry.resultType(), geometry.batchCount(), geometry.m(), geometry.k(),
                        geometry.n(), geometry.rightNStride(), geometry.resultNStride(), lanes,
                        epilogue.terminal(), biasId == null ? 1 : require(values, biasId)
                            .descriptor().layout().orElseThrow().stride(0)),
                cpuInputs.portableExecution().computePreference()).selected();
        boolean vector = realization == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                || realization == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        var ir = new CpuMatmulIr(geometry.leftType(), geometry.rightType(), geometry.resultType(),
                realization, epilogue, vector ? bits : 0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,
                bindings.subList(0,bindings.size()-1).stream().map(CpuAccessPlan.Binding::plan).toList(),
                bindings.getLast().plan());
        if (effectiveBoundaryValues != null) {
            effectiveBoundaryValues.addAll(node.inputs());
            if (biasId != null) effectiveBoundaryValues.add(biasId);
            effectiveBoundaryValues.add(current);
        }
        return new CpuPartitionLowering.LoweredPartition(ir.encodedKernelIr(), ids, bindings,
                ids.stream().map(id -> require(values, id).descriptor().layout().orElseThrow()
                        .referencedElementSpan()).toList(),
                ids.stream().map(id->require(values,id).descriptor().dataType()).toList(),
                context.nodes().subList(0, anchor).stream().map(n -> n.outputs().getFirst()).toList(),
                new long[] {geometry.outputCount()}, geometry.outputCount(),
                "legal: bounded portable MATMUL " + realization, new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(
                        new CpuPartitionLowering.LoweredPartition.MatmulFacts(ir,geometry)));
    }

    /**
     * Reuses an already-qualified selected unit, including its exact fused epilogue and types.
     * @param unit complete resolved ordinary MATMUL unit with immutable invocation geometry
     * @return every implemented legal fixed realization for that exact selected operation
     * @throws IllegalArgumentException if the unit has no resolved MATMUL facts
     */
    public static List<CpuPartitionLowering.LoweredPartition> tuningAlternatives(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        return tuningAlternatives(unit, unit.accessBindings());
    }

    /**
     * Enumerates fixed forms using the exact physical bindings selected by representation planning.
     * @param unit freshly qualified logical unit, retaining original source declarations
     * @param effectiveBindings actual generated-consumer bindings after declared copies
     * @return immutable scalar, vector, scalar-2x2 and vector-2x2 legal forms; no species search
     */
    public static List<CpuPartitionLowering.LoweredPartition> tuningAlternatives(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            List<CpuAccessPlan.Binding> effectiveBindings) {
        var original = unit.portablePlan().specialization().matmulIr().orElseThrow();
        var geometry = effectiveBindings.equals(unit.accessBindings()) ? unit.matmulGeometry().orElseThrow()
                : representedGeometry(unit.matmulGeometry().orElseThrow(), effectiveBindings);
        int bits = ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()
                ? preferredBits(geometry.resultType()) : 0;
        int lanes = bits / geometry.resultType().bitWidth();
        var forms = new CpuMatmulCandidateSelector().select(new CpuMatmulCandidateSelector.Facts(
                geometry.leftType(), geometry.rightType(), geometry.resultType(), geometry.batchCount(),
                geometry.m(), geometry.k(), geometry.n(), geometry.rightNStride(), geometry.resultNStride(),
                lanes, original.epilogue().terminal(), original.epilogue().hasBias()
                    ? effectiveBindings.get(2).effectiveStrides().getLast() : 1), ComputePreference.SCALAR).candidates();
        var result = new ArrayList<CpuPartitionLowering.LoweredPartition>(forms.size());
        for (var form : forms) result.add(loweringForUnit(unit, selectedIr(original, form, bits)));
        return List.copyOf(result);
    }

    /**
     * Reconstructs one exact fixed realization without enumerating implementation alternatives.
     * @param unit freshly qualified logical unit
     * @param realization exact selected fixed form
     * @param effectiveBindings generated-consumer bindings after declared representation copies
     * @return resolved lowering with current target species and unchanged typed arithmetic
     * @throws IllegalArgumentException if current vector qualification rejects the selected form
     */
    public static CpuPartitionLowering.LoweredPartition tuningRealization(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            CpuMatmulIr.Realization realization, List<CpuAccessPlan.Binding> effectiveBindings) {
        var original = unit.portablePlan().specialization().matmulIr().orElseThrow();
        var geometry = effectiveBindings.equals(unit.accessBindings()) ? unit.matmulGeometry().orElseThrow()
                : representedGeometry(unit.matmulGeometry().orElseThrow(), effectiveBindings);
        boolean vector = realization == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                || realization == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        int bits = vector && ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()
                ? preferredBits(geometry.resultType()) : 0;
        if (vector && (bits == 0 || geometry.rightNStride() != 1 || geometry.resultNStride() != 1
                || original.epilogue().terminal() == CpuMatmulIr.Epilogue.Terminal.TANH
                    && original.epilogue().hasBias()
                    && effectiveBindings.get(2).effectiveStrides().getLast() != 1))
            throw new IllegalArgumentException("frozen vector MATMUL is no longer qualified");
        return loweringForUnit(unit, selectedIr(original, realization, bits));
    }

    private static CpuMatmulIr selectedIr(CpuMatmulIr original, CpuMatmulIr.Realization form, int bits) {
        boolean vector = form == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                || form == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        return new CpuMatmulIr(original.leftType(), original.rightType(), original.resultType(), form,
                original.epilogue(), vector ? bits : 0, original.numericalForm(),
                original.inputAccesses(), original.outputAccess());
    }

    private static CpuPartitionLowering.LoweredPartition loweringForUnit(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            CpuMatmulIr ir) {
        var geometry = unit.matmulGeometry().orElseThrow();
        return new CpuPartitionLowering.LoweredPartition(ir.encodedKernelIr(),
                unit.boundaryValues(), unit.accessBindings(), unit.accessBindings().stream()
                    .map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                unit.portablePlan().specialization().boundaryDataTypes(), List.of(),
                new long[] {geometry.outputCount()}, geometry.outputCount(), unit.fusionReason(), new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(
                        new CpuPartitionLowering.LoweredPartition.MatmulFacts(ir, geometry)));
    }
    /**
     * Finds the adjacent, single-use private rank-two transpose prefix of one MATMUL anchor.
     * Public or outside-unit/partition obligations stop the prefix; unsupported views remain
     * separate copy units. Repeated MATMUL input occurrences do not acquire a fan-out exception.
     * A substitution that would merge the existing distinct read identities also keeps a copy.
     *
     * @param context non-null complete or privacy-preserving projected CPU context
     * @param values non-null exact projected descriptors by identity
     * @param anchor valid node ordinal of the candidate MATMUL
     * @return first included transpose ordinal, or {@code anchor} when none qualifies
     * @throws IllegalArgumentException if required descriptor facts are missing
     */
    static int privateTransposePrefixStart(PrepareContext<?> context,
            Map<ValueId, GraphValue> values, int anchor) {
        CompiledNode matmul = context.nodes().get(anchor);
        if (matmul.operation().kind() != MatmulKind.MATMUL || matmul.inputs().size() != 2
                || matmul.inputs().stream().anyMatch(id -> require(values, id).descriptor()
                        .shape().rank() != 2)) return anchor;
        int first = anchor;
        var backingIds = new ArrayList<>(matmul.inputs());
        for (int index = anchor - 1; index >= Math.max(0, anchor - 2); index--) {
            CompiledNode view = context.nodes().get(index);
            if (view.operation().kind() != AxisTransformKind.PERMUTE
                    || !(view.operation().attrs() instanceof PermutationAttrs attrs)
                    || !attrs.axes().equals(List.of(1, 0))
                    || view.inputs().size() != 1 || view.outputs().size() != 1) break;
            ValueId output = view.outputs().getFirst();
            var uses = context.partitionDag().consumers(output);
            var memory = context.memoryRequirements().stream()
                    .filter(requirement -> requirement.valueId().equals(output)).findFirst().orElse(null);
            if (uses.size() != 1 || uses.getFirst().node() != matmul
                    || memory == null || memory.graphOutput()
                    || !memory.producerPartition().equals(Optional.of(context.partition()))
                    || !memory.consumerPartitions().equals(List.of(context.partition()))) break;
            int role = uses.getFirst().inputPosition();
            ValueId source = view.inputs().getFirst();
            // The existing MATMUL boundary ABI requires distinct logical read identities.
            // Retain a copy if replacing this view would merge the two read boundaries.
            if (backingIds.get(1 - role).equals(source)) break;
            var input = require(values, view.inputs().getFirst()).descriptor();
            var result = require(values, output).descriptor();
            if (input.shape().rank() != 2 || !new io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider()
                    .supports(new OperationCapabilityQuery(view.operation(), List.of(input),
                            List.of(result)))) break;
            backingIds.set(role, source);
            first = index;
        }
        return first;
    }

    private static int preferredBits(DataType type) {
        return switch (type) {
            case FLOAT64 -> jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT32 -> jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> jdk.incubator.vector.IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> jdk.incubator.vector.LongVector.SPECIES_PREFERRED.vectorBitSize();
            default -> 0;
        };
    }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("MATMUL value is not projected: " + id);
        return value;
    }
    /**
     * Builds a semantic read/write binding from an immutable descriptor. For an absorbed input,
     * this is a cold read descriptor over the original backing, not a replacement graph value.
     *
     * @param descriptor non-null static descriptor with resolved layout
     * @param kind non-null read or write direction
     * @return checked numeric binding retaining exact strides, offset and referenced span
     * @throws IllegalArgumentException if the binding's addressing contract is invalid
     * @throws ArithmeticException if checked address/count arithmetic overflows
     */
    private static CpuAccessPlan.Binding binding(TensorDescriptor descriptor, CpuAccessPlan.AccessKind kind) {
        long[] extents = descriptor.shape().toLongArray();
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        long expected = 1; int suffix = 0;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST : axis >= extents.length - suffix
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                extents.length, roles, suffix);
        long elements = count(extents);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), strides,
                elements, 0, elements, layout.referencedElementSpan());
    }

    /**
     * Revalidates and normalizes one exact static descriptor triple.
     *
     * @param left non-null left input descriptor with a resolved non-negative layout
     * @param right non-null right input descriptor with a resolved non-negative layout
     * @param result non-null exact promoted output descriptor with an injective layout
     * @return immutable checked geometry
     * @throws NullPointerException if a descriptor is {@code null}
     * @throws IllegalArgumentException if semantic, promotion or layout facts disagree
     * @throws ArithmeticException if a count or address calculation overflows
     */
    public Geometry lower(TensorDescriptor left, TensorDescriptor right, TensorDescriptor result) {
        return normalizeGeometry(left, right, result);
    }

    /**
     * Reconstructs exact invocation strides after an existing declared representation copy.
     * @param original unchanged typed logical MATMUL equation
     * @param bindings exact effective read/bias/write bindings used by the generated consumer
     * @return geometry for the same operation over the selected physical representations
     * @throws IllegalArgumentException if effective bindings violate the original equation
     */
    public static Geometry representedGeometry(Geometry original, List<CpuAccessPlan.Binding> bindings) {
        return normalizeGeometry(representedDescriptor(original.leftType(), bindings.get(0)),
                representedDescriptor(original.rightType(), bindings.get(1)),
                representedDescriptor(original.resultType(), bindings.getLast()));
    }

    private static TensorDescriptor representedDescriptor(DataType type, CpuAccessPlan.Binding binding) {
        Shape shape = Shape.of(binding.extents().stream().mapToLong(Long::longValue).toArray());
        var layout = io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(shape,
                binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray(),
                binding.baseElementOffset(), true);
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    private static Geometry normalizeGeometry(TensorDescriptor left, TensorDescriptor right, TensorDescriptor result) {
        Objects.requireNonNull(left, "left"); Objects.requireNonNull(right, "right");
        Objects.requireNonNull(result, "result");
        long[] a = extents(left), b = extents(right), c = extents(result);
        if (a.length == 2 && b.length == 2)
            validateRankTwo(left.shape(),right.shape(),result.shape());
        if (a.length < 1 || b.length < 1) throw new IllegalArgumentException(
                "MATMUL operand rank must be at least one");
        DataType promoted = DataTypePromotion.promoteNumeric(left.dataType(), right.dataType());
        if (promoted != result.dataType()) throw new IllegalArgumentException(
                "MATMUL result type disagrees with numeric promotion");
        long k = a[a.length - 1], rightK = b[b.length == 1 ? 0 : b.length - 2];
        if (k != rightK) throw new IllegalArgumentException("MATMUL K extents disagree");
        boolean removeM = a.length == 1, removeN = b.length == 1;
        long m = removeM ? 1 : a[a.length - 2], n = removeN ? 1 : b[b.length - 1];
        int aBatchRank = Math.max(0, a.length - 2), bBatchRank = Math.max(0, b.length - 2);
        int batchRank = Math.max(aBatchRank, bBatchRank);
        int expectedRank = batchRank + (removeM ? 0 : 1) + (removeN ? 0 : 1);
        if (c.length != expectedRank) throw new IllegalArgumentException("MATMUL result rank disagrees");
        long[] batchExtents = new long[batchRank];
        long[] leftBatchStrides = new long[batchRank], rightBatchStrides = new long[batchRank];
        long[] resultBatchStrides = new long[batchRank];
        long[] as = strides(left), bs = strides(right), cs = strides(result);
        int aShift = batchRank - aBatchRank, bShift = batchRank - bBatchRank;
        for (int axis = 0; axis < batchRank; axis++) {
            long ae = axis < aShift ? 1 : a[axis - aShift];
            long be = axis < bShift ? 1 : b[axis - bShift];
            if (ae != be && ae != 1 && be != 1) throw new IllegalArgumentException(
                    "MATMUL batch extents do not right-broadcast");
            long extent = ae == be ? ae : ae == 1 ? be : ae;
            if (c[axis] != extent) throw new IllegalArgumentException("MATMUL result batch disagrees");
            batchExtents[axis] = extent;
            leftBatchStrides[axis] = axis < aShift || ae == 1 && extent != 1
                    ? 0 : as[axis - aShift];
            rightBatchStrides[axis] = axis < bShift || be == 1 && extent != 1
                    ? 0 : bs[axis - bShift];
            resultBatchStrides[axis] = cs[axis];
        }
        int resultAxis = batchRank;
        long resultMStride = 0, resultNStride = 0;
        if (!removeM) {
            if (c[resultAxis] != m) throw new IllegalArgumentException("MATMUL result M disagrees");
            resultMStride = cs[resultAxis++];
        }
        if (!removeN) {
            if (c[resultAxis] != n) throw new IllegalArgumentException("MATMUL result N disagrees");
            resultNStride = cs[resultAxis];
        }
        if (!injective(c, cs)) throw new IllegalArgumentException("MATMUL output is not injective");
        long batchCount = count(batchExtents), outputCount = Math.multiplyExact(
                Math.multiplyExact(batchCount, m), n);
        long leftSpan = span(left, a, as), rightSpan = span(right, b, bs);
        long resultSpan = span(result, c, cs);
        return new Geometry(left.dataType(), right.dataType(), result.dataType(), batchExtents,
                leftBatchStrides, rightBatchStrides, resultBatchStrides, batchCount, m, k, n,
                removeM, removeN, left.layout().orElseThrow().storageOffset(),
                right.layout().orElseThrow().storageOffset(), result.layout().orElseThrow().storageOffset(),
                removeM ? 0 : as[a.length - 2], as[a.length - 1],
                bs[b.length == 1 ? 0 : b.length - 2], removeN ? 0 : bs[b.length - 1],
                resultMStride, resultNStride, outputCount, leftSpan, rightSpan, resultSpan);
    }

    private static long[] extents(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty())
            throw new IllegalArgumentException("MATMUL requires static resolved descriptors");
        return descriptor.shape().toLongArray();
    }
    private static long[] strides(TensorDescriptor descriptor) {
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        if (layout.storageOffset() < 0 || Arrays.stream(strides).anyMatch(value -> value < 0))
            throw new IllegalArgumentException("MATMUL requires non-negative layouts");
        return strides;
    }
    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long count = 1;
        for (long extent : extents) count = Math.multiplyExact(count, extent);
        return count;
    }
    private static long span(TensorDescriptor descriptor, long[] extents, long[] strides) {
        if (count(extents) == 0) return 0;
        long maximum = descriptor.layout().orElseThrow().storageOffset();
        for (int axis = 0; axis < extents.length; axis++) maximum = Math.addExact(maximum,
                Math.multiplyExact(extents[axis] - 1, strides[axis]));
        return Math.addExact(maximum, 1);
    }
    private static boolean injective(long[] extents, long[] strides) {
            return CpuOutputInjectivity.isInjective(extents, strides);
        }

    /**
     * Complete normalized logical geometry. Array components are defensively copied.
     *
     * @param leftType exact left representation type
     * @param rightType exact right representation type
     * @param resultType exact promoted result type
     * @param batchExtents normalized right-aligned batch extents
     * @param leftBatchStrides effective left batch strides, including broadcast zeroes
     * @param rightBatchStrides effective right batch strides, including broadcast zeroes
     * @param resultBatchStrides injective result batch strides
     * @param batchCount checked batch product
     * @param m normalized M extent
     * @param k normalized K extent
     * @param n normalized N extent
     * @param removedM whether rank-one left promotion removes M from the result
     * @param removedN whether rank-one right promotion removes N from the result
     * @param leftOffset left layout origin in elements
     * @param rightOffset right layout origin in elements
     * @param resultOffset result layout origin in elements
     * @param leftMStride effective left M stride
     * @param leftKStride effective left K stride
     * @param rightKStride effective right K stride
     * @param rightNStride effective right N stride
     * @param resultMStride effective result M stride
     * @param resultNStride effective result N stride
     * @param outputCount checked logical result element count
     * @param leftSpan checked referenced left span in elements
     * @param rightSpan checked referenced right span in elements
     * @param resultSpan checked referenced result span in elements
     */
    public record Geometry(DataType leftType, DataType rightType, DataType resultType,
            long[] batchExtents, long[] leftBatchStrides, long[] rightBatchStrides,
            long[] resultBatchStrides, long batchCount, long m, long k, long n,
            boolean removedM, boolean removedN, long leftOffset, long rightOffset, long resultOffset,
            long leftMStride, long leftKStride, long rightKStride, long rightNStride,
            long resultMStride, long resultNStride, long outputCount,
            long leftSpan, long rightSpan, long resultSpan) {

        /**
         * Complete coefficient-only ordinary MATMUL recipe for composition inside a Scan.
         * No descriptor, graph, temporal role or invocation storage is retained. Dimension
         * equations are proved on coefficients, including empty and B-contracting domains.
         *
         * @param ir immutable encoded portable identity shared by every numeric binding
         * @param matmulIr direct scalar family identity, with no historical rank-two batch payload
         * @param boundaries left, right and result identities in ordinary carrier order
         * @param boundaryTypes exact represented left, right and promoted result types
         * @param bindings immutable original-rank element access recipes
         * @param iterationCount complete scalar output-count capacity, in elements
         * @param scratchBytes zero scratch capacity for direct scalar MATMUL
         */
        public record SymbolicGeometry(CpuKernelIr ir, CpuMatmulIr matmulIr,
                List<ValueId> boundaries, List<DataType> boundaryTypes,
                List<CpuAccessPlan.BatchBinding> bindings, BatchByteSize iterationCount,
                BatchByteSize scratchBytes) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots numeric facts and proves the complete rank, type and dimension equations.
             * @param ir non-null encoded direct scalar identity
             * @param matmulIr non-null exact family identity without a fused suffix
             * @param boundaries non-null ordered left/right/result identities
             * @param boundaryTypes non-null exact represented types in boundary order
             * @param bindings non-null proved read/read/write recipes, each rank at most sixteen
             * @param iterationCount non-null exact result element-count recipe
             * @param scratchBytes non-null zero scratch recipe
             * @throws NullPointerException if a reference or list entry is null
             * @throws IllegalArgumentException if numeric equations, layouts or identities disagree
             * @throws ArithmeticException if coefficient products overflow
             */
            public SymbolicGeometry {
                Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(matmulIr, "matmulIr");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
                Objects.requireNonNull(iterationCount, "iterationCount");
                Objects.requireNonNull(scratchBytes, "scratchBytes");
                if (boundaries.size() != 3 || boundaryTypes.size() != 3 || bindings.size() != 3
                        || !ir.structuralKey().equals(matmulIr.encodedKernelIr().structuralKey())
                        || matmulIr.realization() != CpuMatmulIr.Realization.DIRECT_SCALAR
                        || matmulIr.epilogue().hasBias() || matmulIr.epilogue().hasTerminal()
                        || matmulIr.numericalForm() != CpuMatmulIr.NumericalForm.SEQUENTIAL
                        || matmulIr.batchGeometry() != null
                        || !boundaryTypes.equals(List.of(matmulIr.leftType(),
                                matmulIr.rightType(), matmulIr.resultType()))
                        || !matmulIr.inputAccesses().equals(List.of(
                                bindings.get(0).plan(), bindings.get(1).plan()))
                        || !matmulIr.outputAccess().equals(bindings.get(2).plan())
                        || !iterationCount.equals(CpuPartitionLowering.batchElements(bindings.get(2)))
                        || !scratchBytes.equals(new BatchByteSize(0, 0, 1)))
                    throw new IllegalArgumentException("symbolic MATMUL identity disagrees");
                for (int position = 0; position < 3; position++) {
                    var plan = bindings.get(position).plan();
                    if (plan.iterationRank() > 16 || plan.accessKind() != (position == 2
                            ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ))
                        throw new IllegalArgumentException("symbolic MATMUL access disagrees");
                }
                validateDimensions(bindings.get(0), bindings.get(1), bindings.get(2));
            }

            private static void validateDimensions(CpuAccessPlan.BatchBinding a,
                    CpuAccessPlan.BatchBinding b, CpuAccessPlan.BatchBinding c) {
                int ar = a.plan().iterationRank(), br = b.plan().iterationRank();
                if (ar < 1 || br < 1)
                    throw new IllegalArgumentException("MATMUL operands require positive rank");
                requireExtent(a, ar - 1, b, br == 1 ? 0 : br - 2);
                int abr = Math.max(0, ar - 2), bbr = Math.max(0, br - 2);
                int rank = Math.max(abr, bbr), out = rank;
                if (c.plan().iterationRank() != rank + (ar == 1 ? 0 : 1) + (br == 1 ? 0 : 1))
                    throw new IllegalArgumentException("MATMUL result rank disagrees");
                for (int axis = 0; axis < rank; axis++) {
                    int aa = axis - (rank - abr), ba = axis - (rank - bbr);
                    if (singleton(a, aa)) requireExtent(b, ba, c, axis);
                    else if (singleton(b, ba)) requireExtent(a, aa, c, axis);
                    else {
                        requireExtent(a, aa, b, ba);
                        requireExtent(a, aa, c, axis);
                    }
                }
                if (ar != 1) requireExtent(a, ar - 2, c, out++);
                if (br != 1) requireExtent(b, br - 1, c, out);
            }

            private static boolean singleton(CpuAccessPlan.BatchBinding binding, int axis) {
                return axis < 0 || binding.extentConstants().get(axis) == 1
                        && binding.extentBatchFactors().get(axis) == 0;
            }

            private static void requireExtent(CpuAccessPlan.BatchBinding a, int aa,
                    CpuAccessPlan.BatchBinding b, int ba) {
                long ac = aa < 0 ? 1 : a.extentConstants().get(aa);
                long af = aa < 0 ? 0 : a.extentBatchFactors().get(aa);
                long bc = ba < 0 ? 1 : b.extentConstants().get(ba);
                long bf = ba < 0 ? 0 : b.extentBatchFactors().get(ba);
                if (ac != bc || af != bf)
                    throw new IllegalArgumentException("symbolic MATMUL dimension equation disagrees");
            }

            /**
             * Resolves an original boundary shape for numeric composition/control binding.
             * @param boundary ordinary carrier position, from zero through two
             * @param batch non-negative actual canonical B within CPU int geometry limits
             * @return fresh original-rank extents, without constructing a Model Shape
             * @throws IndexOutOfBoundsException if the boundary position is invalid
             * @throws IllegalArgumentException if B is negative
             * @throws ArithmeticException if B or an extent exceeds CPU int geometry limits
             */
            public long[] shape(int boundary, long batch) {
                if (batch < 0) throw new IllegalArgumentException("negative MATMUL batch");
                Math.toIntExact(batch);
                var binding = bindings.get(boundary);
                long[] result = new long[binding.plan().iterationRank()];
                for (int axis = 0; axis < result.length; axis++)
                    result[axis] = Math.toIntExact(Math.addExact(binding.extentConstants().get(axis),
                            Math.multiplyExact(binding.extentBatchFactors().get(axis), batch)));
                return result;
            }

            /** @return empty inventory because ordinary MATMUL packs have no mutable cursor cells */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset>
                    resets() { return List.of(); }

            /** @return the ordinary matrix family facts without resolving any invocation B */
            @Override public Optional<CpuMatmulIr> matrixIr() { return Optional.of(matmulIr); }

            /** @return deterministic typed coefficient identity, excluding actual B and storage */
            @Override public String identity() {
                return ir.structuralKey() + ":scan-matmul-batch-v1:" + boundaryTypes + ':' + bindings
                        + ":items=" + iterationCount + ":scratch=" + scratchBytes;
            }

            /**
             * Resolves original-rank numeric layouts and normalizes vector/batch axes directly.
             * No descriptor inference, semantic lowerer or generated-class selection runs here.
             * @param batch non-negative actual canonical B within checked CPU geometry limits
             * @return ordinary MATMUL facts retaining exactly this recipe's compiled IR
             * @throws IllegalArgumentException if B is negative
             * @throws ArithmeticException if numeric count, stride, span or byte limits overflow
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var resolved = bindings.stream().map(binding -> binding.bind(batch)).toList();
                for (int position = 0; position < resolved.size(); position++)
                    Math.multiplyExact(resolved.get(position).referencedElementSpan(),
                            boundaryTypes.get(position).byteWidth());
                var a = resolved.get(0); var b = resolved.get(1); var c = resolved.get(2);
                int ar = a.extents().size(), br = b.extents().size();
                boolean removeM = ar == 1, removeN = br == 1;
                int abr = Math.max(0, ar - 2), bbr = Math.max(0, br - 2);
                int rank = Math.max(abr, bbr);
                long[] be = new long[rank], as = new long[rank],
                        bs = new long[rank], cs = new long[rank];
                for (int axis = 0; axis < rank; axis++) {
                    int aa = axis - (rank - abr), ba = axis - (rank - bbr);
                    be[axis] = c.extents().get(axis);
                    as[axis] = singleton(bindings.get(0), aa) ? 0 : a.effectiveStrides().get(aa);
                    bs[axis] = singleton(bindings.get(1), ba) ? 0 : b.effectiveStrides().get(ba);
                    cs[axis] = c.effectiveStrides().get(axis);
                }
                long m = removeM ? 1 : a.extents().get(ar - 2);
                long k = a.extents().get(ar - 1), n = removeN ? 1 : b.extents().get(br - 1);
                long outputCount = iterationCount.byteSize(batch);
                var geometry = new Geometry(boundaryTypes.get(0), boundaryTypes.get(1),
                        boundaryTypes.get(2), be, as, bs, cs, count(be), m, k, n, removeM, removeN,
                        a.baseElementOffset(), b.baseElementOffset(), c.baseElementOffset(),
                        removeM ? 0 : a.effectiveStrides().get(ar - 2),
                        a.effectiveStrides().get(ar - 1),
                        b.effectiveStrides().get(removeN ? 0 : br - 2),
                        removeN ? 0 : b.effectiveStrides().get(br - 1),
                        removeM ? 0 : c.effectiveStrides().get(rank),
                        removeN ? 0 : c.effectiveStrides().get(rank + (removeM ? 0 : 1)),
                        outputCount, a.referencedElementSpan(), b.referencedElementSpan(),
                        c.referencedElementSpan());
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {outputCount}, outputCount,
                        "legal: general symbolic direct scalar MATMUL", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.of(
                                new CpuPartitionLowering.LoweredPartition.MatmulFacts(matmulIr, geometry)));
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
            return new TemporalRebind(List.of(new BaseCell(0, 0, leftOffset),
                    new BaseCell(1, 1, rightOffset), new BaseCell(2, 2, resultOffset)), List.of());
        }

        /**
         * Describes an ordinary biased MATMUL's left, right, bias, then result boundaries.
         * The bias layout is owned by ordinary fusion, so its checked offset is supplied explicitly.
         * @param start non-negative inclusive helper range start
         * @param biasOffset checked carrier-relative bias layout offset in bias elements
         * @return immutable base facts for the existing bias pack; no cursor cells are mutable
         * @throws IllegalArgumentException if start is negative
         */
        public TemporalRebind temporalRebind(long start, long biasOffset) {
            if (start < 0) throw new IllegalArgumentException("negative temporal range start");
            return new TemporalRebind(List.of(new BaseCell(0, 0, leftOffset),
                    new BaseCell(1, 1, rightOffset), new BaseCell(14, 2, biasOffset),
                    new BaseCell(2, 3, resultOffset)), List.of());
        }

        /** Snapshots and validates complete normalized geometry. */
        public Geometry {
            Objects.requireNonNull(leftType, "leftType"); Objects.requireNonNull(rightType, "rightType");
            Objects.requireNonNull(resultType, "resultType");
            batchExtents = batchExtents.clone(); leftBatchStrides = leftBatchStrides.clone();
            rightBatchStrides = rightBatchStrides.clone(); resultBatchStrides = resultBatchStrides.clone();
            int rank = batchExtents.length;
            if (leftBatchStrides.length != rank || rightBatchStrides.length != rank
                    || resultBatchStrides.length != rank || batchCount < 0 || m < 0 || k < 0
                    || n < 0 || outputCount < 0 || leftOffset < 0 || rightOffset < 0
                    || resultOffset < 0 || leftMStride < 0 || leftKStride < 0
                    || rightKStride < 0 || rightNStride < 0 || resultMStride < 0
                    || resultNStride < 0 || leftSpan < 0 || rightSpan < 0 || resultSpan < 0)
                throw new IllegalArgumentException("MATMUL geometry is invalid");
        }
        /**
         * Returns the normalized batch extents without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] batchExtents() { return batchExtents.clone(); }
        /**
         * Returns the effective left batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] leftBatchStrides() { return leftBatchStrides.clone(); }
        /**
         * Returns the effective right batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] rightBatchStrides() { return rightBatchStrides.clone(); }
        /**
         * Returns the injective result batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] resultBatchStrides() { return resultBatchStrides.clone(); }

        /**
         * Packs bases, batch arrays, M/K/N strides, and counts for a generated invocation.
         * @param leftBase carrier-relative left element base
         * @param rightBase carrier-relative right element base
         * @param resultBase carrier-relative output element base
         * @return a new packed primitive geometry array
         */
        public long[] pack(long leftBase, long rightBase, long resultBase) {
            return pack(leftBase,rightBase,resultBase,0,0);
        }

        /**
         * Packs one fused rank-one bias origin and N stride with the ordinary MATMUL geometry.
         * @param leftBase carrier-relative left element base
         * @param rightBase carrier-relative right element base
         * @param resultBase carrier-relative output element base
         * @param biasBase checked carrier-relative bias origin including its layout offset
         * @param biasNStride checked non-negative logical bias stride
         * @return a new packed primitive geometry array
         */
        public long[] pack(long leftBase,long rightBase,long resultBase,long biasBase,
                long biasNStride) {
            int rank = batchExtents.length;
            long[] packed = new long[18 + 4 * rank];
            packed[0] = Math.addExact(leftBase, leftOffset); packed[1] = Math.addExact(rightBase, rightOffset);
            packed[2] = Math.addExact(resultBase, resultOffset); packed[3] = batchCount;
            packed[4] = m; packed[5] = k; packed[6] = n; packed[7] = leftMStride;
            packed[8] = leftKStride; packed[9] = rightKStride; packed[10] = rightNStride;
            packed[11] = resultMStride; packed[12] = resultNStride; packed[13] = rank;
            packed[14] = biasBase; packed[15] = biasNStride;
            packed[16] = outputCount; packed[17] = 1;
            int cursor = 18;
            System.arraycopy(batchExtents, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(leftBatchStrides, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(rightBatchStrides, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(resultBatchStrides, 0, packed, cursor, rank);
            return packed;
        }
    }
}

/**
 * Enumerates implemented legal fixed realizations and applies unchanged production thresholds.
 * Only homogeneous FLOAT32/FLOAT64
 * TANH extends the no-terminal vector subset, with contiguous bias when present.
 */
final class CpuMatmulCandidateSelector {
    /** Creates a stateless bounded selector. */
    CpuMatmulCandidateSelector() { }

    /**
     * Constructs all legal fixed forms, independent of profitability thresholds, then selects
     * its unchanged deterministic production member.
     *
     * @param facts checked non-null occurrence facts
     * @param preference non-null cold compute preference; scalar keeps eligible vector forms
     *     available as candidates when their species is resolved, but selects the scalar form
     * @return immutable candidates in scalar, vector, scalar-tiled, vector-tiled order and selection
     * @throws NullPointerException if {@code facts} or {@code preference} is {@code null}
     * @throws ArithmeticException if the multiply-add count overflows
     */
    Selection select(Facts facts, ComputePreference preference) {
        Objects.requireNonNull(facts, "facts");
        Objects.requireNonNull(preference, "preference");
        long operations = Math.multiplyExact(Math.multiplyExact(
                Math.multiplyExact(facts.batchCount(), facts.m()), facts.k()), facts.n());
        boolean vectorEligible = facts.preferredLanes() > 0 && facts.rightNStride() == 1
                && facts.resultNStride() == 1
                && (facts.terminal() == CpuMatmulIr.Epilogue.Terminal.NONE
                    || facts.terminal() == CpuMatmulIr.Epilogue.Terminal.TANH
                        && facts.biasNStride() == 1
                        && (facts.resultType() == DataType.FLOAT32 || facts.resultType() == DataType.FLOAT64))
                && facts.leftType() == facts.rightType() && facts.rightType() == facts.resultType()
                && (facts.resultType() == DataType.FLOAT32 || facts.resultType() == DataType.FLOAT64
                    || facts.resultType() == DataType.INT32 || facts.resultType() == DataType.INT64);
        var candidates = new ArrayList<CpuMatmulIr.Realization>();
        candidates.add(CpuMatmulIr.Realization.DIRECT_SCALAR);
        if (vectorEligible) candidates.add(CpuMatmulIr.Realization.DIRECT_N_VECTOR);
        candidates.add(CpuMatmulIr.Realization.TILED_SCALAR_2X2);
        if (vectorEligible) candidates.add(CpuMatmulIr.Realization.TILED_N_VECTOR_2X2);
        boolean selectVector = vectorEligible && facts.k() > 0 && facts.n() >= facts.preferredLanes()
                && preference == ComputePreference.VECTOR_IF_ELIGIBLE;
        CpuMatmulIr.Realization selected;
        if (operations == 0 || facts.k() == 0 || facts.m() == 1 || facts.n() == 1
                || operations < 4_096) selected = CpuMatmulIr.Realization.DIRECT_SCALAR;
        else if (selectVector && facts.m() >= 2 && facts.n() >= 2L * facts.preferredLanes()
                && operations >= 65_536 && candidates.contains(
                CpuMatmulIr.Realization.TILED_N_VECTOR_2X2))
            selected = CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        else if (operations >= 16_384 && !selectVector)
            selected = CpuMatmulIr.Realization.TILED_SCALAR_2X2;
        else if (selectVector) selected = CpuMatmulIr.Realization.DIRECT_N_VECTOR;
        else selected = CpuMatmulIr.Realization.DIRECT_SCALAR;
        return new Selection(candidates, selected, operations);
    }

    /** Cold facts used only for eligibility and selection.
     * @param leftType exact left type
     * @param rightType exact right type
     * @param resultType promoted result and accumulator type
     * @param batchCount checked batch product
     * @param m normalized row count
     * @param k normalized contraction count
     * @param n normalized column count
     * @param rightNStride right logical-N element stride
     * @param resultNStride result logical-N element stride
     * @param preferredLanes exact preferred species lane count, or zero when no species is available
     * @param terminal non-null exact fused terminal; only NONE or floating TANH permits vectors
     * @param biasNStride bias element stride, or one when no bias is present; TANH vectors require one
     */
    record Facts(DataType leftType, DataType rightType, DataType resultType,
            long batchCount, long m, long k, long n, long rightNStride, long resultNStride,
            int preferredLanes, CpuMatmulIr.Epilogue.Terminal terminal, long biasNStride) {
        /** Validates non-negative geometry and an available lane count or zero. */
        Facts {
            Objects.requireNonNull(leftType, "leftType"); Objects.requireNonNull(rightType, "rightType");
            Objects.requireNonNull(resultType, "resultType");
            Objects.requireNonNull(terminal, "terminal");
            if (batchCount < 0 || m < 0 || k < 0 || n < 0 || rightNStride < 0
                    || resultNStride < 0 || preferredLanes < 0 || biasNStride < 0)
                throw new IllegalArgumentException("MATMUL candidate facts must be non-negative");
        }
    }

    /** Immutable eligible-set and selected-form result.
     * @param candidates ordered bounded eligible realization set
     * @param selected deterministic production member of {@code candidates}
     * @param multiplyAdds checked occurrence operation count
     */
    record Selection(List<CpuMatmulIr.Realization> candidates,
            CpuMatmulIr.Realization selected, long multiplyAdds) {
        /** Snapshots and validates one selection. */
        Selection {
            candidates = List.copyOf(candidates); Objects.requireNonNull(selected, "selected");
            if (candidates.isEmpty() || candidates.getFirst() != CpuMatmulIr.Realization.DIRECT_SCALAR
                    || candidates.size() > 4 || !candidates.contains(selected) || multiplyAdds < 0)
                throw new IllegalArgumentException("MATMUL selection disagrees");
        }
    }
}
