package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSpecializedSubgraphRecognizer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFusionProfitabilitySelector;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRepresentationPlanner;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartialReductionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasRouteSelector;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.Optional;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ByteVector;

/**
 * Whole-partition CPU analysis entry for static portable families. Complete-plan alternative
 * enumeration remains bounded to eight compiled nodes; longer partitions take a direct,
 * deterministic multi-unit path with no complete-plan candidate enumeration or representation-
 * copy alternatives. For partitions within that bounded domain, analysis retains complete direct,
 * eligible single-copy, and eligible disjoint-consumer pair
 * representation candidates for FLOAT64, FLOAT32, INT32, INT64, and canonical BOOL pointwise
 * work. It rejects a pair when one represented instruction consumes both copied sources, and one
 * compatible copy identity may serve repeated or cross-unit consumers. Ordinary preparation
 * selects the exact CPU 0008D direct topology with no representation-copy resources; retained
 * materialized forms remain complete candidate data for later explicit pre-Runtime promotion.
 * Analysis then selects scalar or preferred-species vector compute and single-thread or bounded
 * parallel orchestration before shared resource assignment. Exact vector eligibility is
 * typed across floating, signed-integral, canonical-BOOL, and bounded dense floating-mask
 * boundaries while preserving unit-private masks; every BFLOAT16 pointwise topology, direct
 * power, and unsafe mask storage remain scalar. Any pointwise topology containing cross-type CAST is also scalar-only, while same-type
 * CAST retains its previous vector eligibility. Cross-type CAST adds no workspace,
 * materialization, or route and uses schema 60 only for its generated class identity. Analysis
 * admits loss vector compute only for MSE {@code NONE} with same-typed FLOAT32 or FLOAT64
 * prediction, target, and result, dense-linear occurrence accesses and resolved bindings, an
 * available multi-lane preferred species with at least one full vector, and an explicit vector
 * preference. MSE {@code SUM}, MSE {@code MEAN}, and every ineligible loss retain scalar or
 * parallel-scalar compute under the existing range policy. Only the admitted vector loss uses
 * class-identity schema 62; scalar loss artifacts retain schema 58.
 * Direct Conv2d/Conv3d selects preferred-species output-width vector compute only for the proved
 * same-typed FLOAT32/FLOAT64 dense-layout subset with width stride and width dilation equal to one
 * and at least one statically in-bounds full-width block. Non-width stride and dilation remain
 * eligible. Array, native-order segment, and ordered mixed-carrier patterns may qualify. Conv2d
 * external epilogues, BFLOAT16, non-dense access, short or interior-free widths, and every other
 * failed proof retain scalar or parallel-scalar compute. Conv1d composition remains scalar. Only
 * the admitted convolution vector realization uses class-identity schema 63; scalar Conv2d/Conv3d
 * artifacts retain their historical schema-52 class projection.
 * Analysis measures nothing and performs no artifact or persistence access. Static affine
 * chains instead retain scalar compute,
 * compose one exact distinct-write address domain, declare only source and final result buffers,
 * and use deterministic scalar fallback when vector compute was preferred.
 * One-node scatter, fold, and ordering plans remain scalar compute and use disjoint output or
 * logical-slice ranges. Fold selects neither materialization nor workspace; scatter declares
 * per-range scratch only for nonempty floating multiplication. Ordering always declares the
 * checked run-owned workspace required by its two primitive INT64 merge-index regions per
 * selected range and may expose one or two output stores.
 * One-node cumulative scans remain scalar compute, declare exactly input and output with no
 * workspace or materialization, and may parallelize only across complete independent slices.
 * One-node ordinary aggregates likewise remain scalar compute, declare exactly input and output
 * with no workspace or materialization, and may parallelize only across complete output cells.
 * Fixed softmax derivative units instead force scalar single-range execution, reuse one
 * exact-state slice across rows and declare the full-gradient final span, not the SUM span.
 * One-node masked SUM/MEAN plans declare ordered data, mask, and output buffers plus one exact
 * floating-state workspace sliced by the simultaneously used complete-output-cell ranges. They
 * never materialize the mask or declare selected-count, partial, or combine state.
 * One-node trailing Layer/RMS plans likewise keep scalar compute and partition complete leading
 * slices. Layer declares one exact-state slice per simultaneous range; RMS declares no workspace.
 * Neither family selects materialization, partial values, or combine state.
 * After all common lowering and ordinary representation decisions, an explicitly qualified and
 * strictly cheaper bare rank-two FLOAT32/FLOAT64 MATMUL may select the narrow OpenBLAS route.
 * That post-lowering decision uses only immutable storage, thread-configuration, and cost facts;
 * it retains the complete portable realization and performs no provider query or allocation.
 * Portable MATMUL realization and generated compute strategy follow the same cold preference;
 * a scalar preference retains direct or threshold-eligible tiled scalar work even when the
 * descriptor would also qualify for a vector candidate.
 * Scalar analysis does not resolve optional Vector API species; only an explicit vector
 * preference requests those target facts.
 * The profile-free analysis retains the qualified CPU routes and binds only non-profile
 * lowering, generated-artifact, OpenBLAS, and tuning identity facts. Route choice remains cold;
 * it does not alter the generated hot-loop algorithm. Proved row-map/multiple-terminal units
 * are selected before declarations, use scalar single-range full-body work count one even at
 * B=0, and retain every exact-SUM/saved/public boundary. Only selected ordinary regions use
 * projection 72; represented Scan regions remain inside their existing projection-71 unit.
 */
public final class CpuPartitionPreparer implements BackendPartitionPreparer<
        CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> {
    private final CpuPartitionLowering lowering;
    private final CpuPartitionDagDecomposer decomposer = new CpuPartitionDagDecomposer();
    private final CpuSpecializedSubgraphRecognizer recognizer =
            new CpuSpecializedSubgraphRecognizer();
    private final CpuFusionProfitabilitySelector selector =
            new CpuFusionProfitabilitySelector();
    private final CpuRepresentationPlanner representationPlanner =
            new CpuRepresentationPlanner();
    private final CpuOpenBlasRouteSelector openBlasSelector = new CpuOpenBlasRouteSelector();
    private final CpuPartialReductionLowering partialReductionLowering =
            new CpuPartialReductionLowering();

    /** Creates a preparer with the permanent common lowering owner. */
    public CpuPartitionPreparer() { this(new CpuPartitionLowering()); }

    /**
     * Creates a preparer with an explicit lowering collaborator.
     * @param lowering non-null whole-partition lowering retained by the preparer
     * @throws NullPointerException if {@code lowering} is {@code null}
     */
    public CpuPartitionPreparer(CpuPartitionLowering lowering) {
        this.lowering = Objects.requireNonNull(lowering, "lowering");
    }

    /**
     * Lowers and fuses one complete CPU-owned partition, selects a bounded complete plan when
     * available or a direct decomposed plan, and declares exact post-fusion resources. The direct
     * path has no whole-partition node/boundary ceiling and makes no complete-candidate or copy-tuning claim.
     * Qualified symbolic partitions retain fixed scalar, single-range ordinary units and
     * independently proved Scan units. Their exact run-size declarations precede slot assignment;
     * invocation B never selects topology, a route or new shared requirements.
     * Proved static typed-one counts retain every original source's initialized representation
     * as a partition declaration while their generated units expose only output boundaries.
     * Absorbed private expansions and contiguous copies have no declaration or executable.
     * A nonempty carrier pattern describes all final deduplicated partition declarations,
     * including retained sources, in declaration order; each unit receives only its own carriers.
     * @param context non-null complete CPU analysis context
     * @return one immutable analysis with non-empty topologically ordered units, one exact
     *     deduplicated declaration per materialized value, and each unit's already-selected
     *     portable strategy and optional exact workspace; never {@code null}. Multi-unit workspace
     *     declarations are rebased to the final unit index before they enter the partition
     *     requirement list. A one-unit plan may retain the established optional external-read
     *     materialization, while a multi-unit plan disables it and retains only family-intrinsic
     *     unit workspaces. In the bounded candidate domain, the returned plan also carries ordered
     *     CPU-private recognition facts only after their exact baseline-unit IR and resource
     *     snapshot has been validated. The unfused baseline's graph-value provenance is retained
     *     only for cold plan validation, including selected virtual epilogue outputs;
     *     those facts do not alter declarations, artifact identity, finalization, or execution.
     *     The plan also retains the authoritative logical-memory graph-publication boundary
     *     positions solely so its selected boundary roles can be independently recomputed.
     *     An eligible exact MATMUL additionally retains one complete portable-first tuning batch;
     *     a compatible decision or the safe heuristic may select one OpenBLAS plan with up to
     *     three exact native copy-workspace declarations after the portable plan is complete.
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        requirePermittedSource(context);
        if (context.requiresBatchSize()) return analyzeOrdinaryBatch(context);
        var units = decomposer.decompose(context, lowering);
        if (!context.constants().isEmpty()) {
            if (units.stream().anyMatch(u -> u.lowering().portableKernelIr()
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr))
                return withMetadata(context, analyzeTopology(context, units, null), List.of(), List.of(), null);
        }
        if (directPartition(context, units)) return analyzeDirectPartition(context, units);
        return finishOpenBlas(context, selectPlan(context, Optional.empty(), units).analysis());
    }

    /**
     * Composes qualified ordinary and Scan units using the complete partition's actual read
     * mappings. Scoped bodies remain inside their Scan unit; only unique outer reads and demanded
     * writes enter shared declarations. Structural zero Scan workspaces retain their membership.
     *
     * @param context non-null batch-required CPU projection
     * @return non-null immutable fixed-topology scalar analysis with exact numeric size recipes
     * @throws IllegalArgumentException if a unit, carrier or access mapping is unsupported
     * @throws ArithmeticException if structural extent, stride or byte coefficients overflow
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeOrdinaryBatch(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        var rowRegions = ordinaryRowRegions(context);
        var regionMembers = new java.util.HashSet<io.github.pho001.synaptik.model.graph.CompiledNode>();
        for (var lowered : rowRegions.values()) {
            var ir = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)lowered.portableKernelIr();
            int inputs = (int)ir.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count();
            var selected = new java.util.HashSet<>(lowered.virtualValues());
            selected.addAll(lowered.boundaryValues().subList(inputs,lowered.boundaryValues().size()));
            context.nodes().stream().filter(n -> n.outputs().stream().anyMatch(selected::contains))
                    .forEach(regionMembers::add);
        }
        var layouts = CpuPartitionLowering.batchLayouts(context);
        var sizes = CpuPartitionLowering.batchByteSizes(context);
        var values = new LinkedHashMap<ValueId, io.github.pho001.synaptik.model.graph.GraphValue>();
        context.values().forEach(value -> values.put(value.id(), value));
        var ordered = new java.util.LinkedHashSet<ValueId>();
        for (var node : context.nodes()) {
            if (rowRegions.containsKey(node)) { ordered.addAll(rowRegions.get(node).boundaryValues()); continue; }
            if (regionMembers.contains(node)) continue;
            ordered.addAll(node.inputs());
            if (node.scanBody().isPresent()) node.scanBody().orElseThrow().demandedOutputIndices()
                    .forEach(p -> ordered.add(node.outputs().get(p)));
            else ordered.addAll(node.outputs());
        }
        var boundaryValues = List.copyOf(ordered);
        var requested = context.backendInputs().carrierPattern().isEmpty()
                ? java.util.Collections.nCopies(boundaryValues.size(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : context.backendInputs().carrierPattern();
        if (requested.size() != boundaryValues.size()) throw new IllegalArgumentException("batch carrier count disagrees");
        var carriers = new LinkedHashMap<ValueId, CpuKernelSpecialization.CarrierAccess>();
        for (int i = 0; i < boundaryValues.size(); i++) {
            var carrier = requested.get(i);
            if (carrier != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                    && carrier != CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY) throw new IllegalArgumentException("unsupported batch carrier");
            carriers.put(boundaryValues.get(i), carrier);
        }
        var declarations = boundaryValues.stream().map(id -> new PreparationResourceRequirement.Buffer(
                id, sizes.get(id), values.get(id).descriptor().dataType().byteWidth())).toList();
        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        var units = new ArrayList<CpuPartitionPreparationPlan.ExecutionUnitPlan>();
        var producers = new LinkedHashMap<ValueId, Integer>();
        for (int position = 0; position < context.nodes().size(); position++) {
            var node = context.nodes().get(position);
            if (rowRegions.containsKey(node)) {
                var lowered = rowRegions.get(node);
                var ir = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)lowered.portableKernelIr();
                var pattern = lowered.boundaryValues().stream().map(carriers::get).toList();
                var rowExec = rowEmission(context,ir,null);
                int ranges = rowExec.map(CpuKernelSpecialization.RowFusionEmission::maximumPartialCount).orElse(1);
                boolean vector = rowExec.isPresent() && context.backendInputs().portableExecution().computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE && vectorRowEligible(ir);
                var strategy = vector ? ranges > 1 ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR : ranges > 1
                        ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
                int bits = vector ? speciesBits(ir.arithmetic().values().getFirst().dataType()) : 0;
                var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        strategy,lowered.boundaryDataTypes(),pattern,
                        bits,-1,List.of(),rowExec.isPresent(),72,Optional.empty(),Optional.empty(),true,false,Optional.of(ir),rowExec);
                var ordinal = new LinkedHashMap<ValueId,Integer>();
                int inputs = (int)ir.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count();
                int read = 0, write = inputs;
                for (var v : ir.arithmetic().values()) if (v.kind() == CpuKernelIr.Value.Kind.INPUT)
                    ordinal.put(lowered.boundaryValues().get(read++),v.ordinal());
                else if (v.kind() == CpuKernelIr.Value.Kind.OUTPUT) ordinal.put(lowered.boundaryValues().get(write++),v.ordinal());
                var recipes = lowered.boundaryValues().stream().map(id -> ir.accesses().get(ordinal.get(id))).toList();
                var route = new CpuPortableRoutePlan(ir,spec,recipes,Optional.empty(),Optional.empty());
                var outputIds = lowered.boundaryValues().subList(inputs,lowered.boundaryValues().size());
                var dependencies = lowered.boundaryValues().subList(0,inputs).stream().map(producers::get)
                        .filter(Objects::nonNull).distinct().toList();
                var members = java.util.stream.IntStream.range(0,context.nodes().size()).filter(i -> {
                    var out = context.nodes().get(i).outputs().getFirst();
                    return lowered.virtualValues().contains(out) || outputIds.contains(out);
                }).boxed().toList();
                int unitIndex = units.size();
                var workspace = rowExec.map(e -> new PreparationResourceRequirement.Workspace(unitIndex,rowWorkspaceBytes(ir,e),8));
                workspace.ifPresent(requirements::add);
                var runtime = new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0],Optional.empty(),workspace,
                        rowExec.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.GENERAL_SCAN_BANKS : CpuPartitionPreparationPlan.WorkspaceUse.NONE,Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty());
                units.add(new CpuPartitionPreparationPlan.ExecutionUnitPlan(route,lowered.boundaryValues(),List.of(),pattern,pattern,
                        new long[]{1},1,strategy,ranges,rowExec.isPresent() ? context.backendInputs().portableExecution().minimumElementsPerWorker() : 1,bits,
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        outputIds.size(),lowered.fusionReason(),dependencies,members,runtime));
                outputIds.forEach(id -> producers.put(id,unitIndex));
                continue;
            }
            if (regionMembers.contains(node)) continue;
            if (node.scanBody().isPresent()) {
                var local = decomposer.unitContext(context,List.of(node),context.backendInputs());
                var lowered = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering().lower(local,layouts);
                var analysis = analyzeGeneralScan(local,lowered,0);
                var unit = analysis.plan().units().getFirst();
                int requirementId = units.size();
                var workspace = unit.runtimeFacts().workspaceDeclaration().map(w ->
                        new PreparationResourceRequirement.Workspace(requirementId,w.size(),w.byteAlignment()));
                workspace.ifPresent(requirements::add);
                var runtime = new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0],Optional.empty(),workspace,
                        workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.GENERAL_SCAN_BANKS : CpuPartitionPreparationPlan.WorkspaceUse.NONE,
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),unit.runtimeFacts().generalScan());
                var dependencies = node.inputs().stream().map(producers::get).filter(Objects::nonNull).distinct().toList();
                units.add(new CpuPartitionPreparationPlan.ExecutionUnitPlan(unit.portablePlan(),unit.boundaryValues(),
                        unit.accessBindings(),unit.carrierPattern(),unit.generatedCarrierPattern(),unit.extents(),unit.elementCount(),
                        unit.executionStrategy(),unit.selectedRangeCount(),unit.minimumElementsPerWorker(),unit.vectorSpeciesBitSize(),
                        Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                        unit.outputCount(),unit.fusionReason(),dependencies,List.of(position),runtime));
                int unitIndex = units.size()-1; node.outputs().forEach(id -> producers.put(id,unitIndex));
                continue;
            }
            var ids = new ArrayList<ValueId>(); for (var id : node.inputs()) if (!ids.contains(id)) ids.add(id);
            ids.addAll(node.outputs());
            for (var id : node.outputs()) if (carriers.get(id) != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                throw new IllegalArgumentException("batch created outputs require segment carriers");
            var pattern = ids.stream().map(carriers::get).toList();
            var route = CpuPartitionLowering.lowerOrdinaryBatch(node, values, layouts, sizes, pattern);
            int unitIndex = units.size();
            Optional<PreparationResourceRequirement.Workspace> workspace = route.batchAggregate()
                    .map(recipe -> new PreparationResourceRequirement.Workspace(unitIndex, recipe, 8));
            workspace.ifPresent(requirements::add);
            var runtime = new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0], Optional.empty(), workspace,
                    workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE : CpuPartitionPreparationPlan.WorkspaceUse.NONE,
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty());
            var dependencies = node.inputs().stream().map(producers::get).filter(Objects::nonNull).distinct().toList();
            units.add(new CpuPartitionPreparationPlan.ExecutionUnitPlan(route, ids, List.of(), pattern, pattern,
                    new long[0], 0, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, 1, 1, 0,
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    1, "ordinary-batch-v1: fixed scalar unit; numeric invocation geometry", dependencies, List.of(position), runtime));
            node.outputs().forEach(id -> producers.put(id, unitIndex));
        }
        var combined = new CpuPartitionPreparationPlan(units, CpuPartitionPreparationPlan.Route.PORTABLE,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, declarations, boundaryValues, List.of(), requested, requested,
                new long[0], 0, new long[0], 1, 1, 0, "ordinary-batch-v1", Optional.empty(), Optional.empty(),
                CpuPartitionPreparationPlan.WorkspaceUse.NONE, new CpuSpecializationBudget(4, 1, 0, 0),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), Optional.empty());
        return new BackendPartitionAnalysis<>(context.partition(), combined, requirements);
    }

    /**
     * Selects bounded ordinary symbolic regions against full partition uses before declarations.
     * F64 symbolic matrices and explicit legacy carrier lists remain on their existing paths.
     * @param context complete symbolic CPU partition
     * @return immutable anchor-to-complete-region map, empty when no proof succeeds
     */
    private Map<io.github.pho001.synaptik.model.graph.CompiledNode,CpuPartitionLowering.LoweredPartition>
            ordinaryRowRegions(PrepareContext<CpuPartitionAnalysisInputs> context) {
        if (!context.backendInputs().carrierPattern().isEmpty()) return Map.of();
        var result = new LinkedHashMap<io.github.pho001.synaptik.model.graph.CompiledNode,CpuPartitionLowering.LoweredPartition>();
        for (int start = 0; start < context.nodes().size(); start++) {
            for (int end = Math.min(context.nodes().size(),start+64); end > start+1; end--) {
                var nodes = context.nodes().subList(start,end);
                try {
                    var lowered = lowering.lower(decomposer.unitContext(context,nodes,context.backendInputs()));
                    if (!(lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir)
                            || lowered.boundaryDataTypes().stream().anyMatch(t -> t != DataType.FLOAT32)) continue;
                    var ids = new LinkedHashMap<ValueId,Integer>();
                    int inputs = (int)ir.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count();
                    int read = 0, write = inputs, virtual = 0;
                    for (var v : ir.arithmetic().values()) ids.put(switch (v.kind()) {
                        case INPUT -> lowered.boundaryValues().get(read++);
                        case OUTPUT -> lowered.boundaryValues().get(write++);
                        case VIRTUAL -> lowered.virtualValues().get(virtual++);
                    },v.ordinal());
                    for (var v : context.values()) if (!ids.containsKey(v.id())) ids.put(v.id(),ids.size());
                    var uses = new ArrayList<List<Integer>>();
                    for (int v = 0; v < ids.size(); v++) uses.add(new ArrayList<>());
                    for (var n : context.nodes()) for (var in : n.inputs()) for (var out : n.outputs()) uses.get(ids.get(in)).add(ids.get(out));
                    for (int v = 0; v < ir.viewOrigins().size(); v++) if (ir.viewOrigins().get(v) != v
                            && !uses.get(ir.viewOrigins().get(v)).contains(v)) uses.get(ir.viewOrigins().get(v)).add(v);
                    if (io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLowering.prove(ir,uses,
                            java.util.stream.IntStream.range(0,ids.size()).mapToObj(v -> List.<Integer>of()).toList()).isEmpty()) continue;
                    result.put(nodes.getFirst(),lowered); start = end-1; break;
                } catch (IllegalArgumentException rejected) { /* Existing per-node path is unchanged. */ }
            }
        }
        return Map.copyOf(result);
    }

    /**
     * Analyzes a partition outside bounded complete-plan enumeration or representation copies,
     * including variadic ordinary units exceeding its per-unit or aggregate boundary ceilings.
     * All distinct buffer and unit-indexed intrinsic-workspace declarations are assembled before
     * shared slot assignment.
     *
     * @param context non-null complete CPU partition context outside bounded candidate discovery
     * @param units exact already-lowered complete direct decomposition
     * @return immutable portable analysis with exact distinct buffers and unit-indexed workspaces;
     *     never {@code null}
     * @throws IllegalArgumentException if any node, projection, or declaration is unsupported
     * @throws ArithmeticException if exact resource geometry overflows
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeDirectPartition(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> units) {
        var analysis = analyzeTopology(context, units, null);
        return withMetadata(context, analysis, List.of(), List.of(), null);
    }

    /**
     * Returns every complete retained topology/representation identity in stable rank order.
     *
     * <p>The result describes retained analysis alternatives only. It performs no route
     * measurement, ranking, execution, or resource allocation.</p>
     *
     * @param context non-null complete CPU analysis context; inspected but not mutated
     * @return a new immutable candidate-identity snapshot and its exact completeness outcome;
     *     for partitions outside the bounded topology/resource protocol, the valid direct plan
     *     is analyzed but the identity list is empty and completeness is false
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if analysis facts are unsupported or inconsistent
     * @throws ArithmeticException if exact candidate or resource arithmetic overflows
     */
    public CompletePlanCandidates completePlanCandidates(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        requirePermittedSource(context);
        var units = decomposer.decompose(context, lowering);
        if (directPartition(context, units)) {
            analyzeDirectPartition(context, units);
            return new CompletePlanCandidates(List.of(), false);
        }
        SelectionAnalysis selected = selectPlan(context, Optional.empty(), units);
        List<CpuRepresentationDecision.Variant>
                variants = selected.representation().decisions().stream()
                .filter(CpuRepresentationDecision.Variant.class::isInstance)
                .map(CpuRepresentationDecision.Variant.class::cast)
                .sorted(java.util.Comparator.comparingInt(
                        CpuRepresentationDecision.Variant::stableTopologyRank))
                .toList();
        List<CpuRepresentationDecision.VariantIdentity>
                identities = variants.stream().map(
                        CpuRepresentationDecision.Variant::identity)
                .distinct().toList();
        return new CompletePlanCandidates(identities, selected.representation().complete());
    }

    /**
     * Freshly selects one exact retained complete plan and performs current route evaluation.
     *
     * <p>The selected identity bypasses ordinary topology/representation ranking but does not
     * grant legality. Current route evaluation must preserve the already authenticated Phase-1
     * state carried by {@code context}.</p>
     *
     * @param context non-null complete CPU analysis context; inspected but not mutated
     * @param selectedPlan non-null exact authenticated selected-plan input; retained only in the
     *     scope of this analysis call
     * @return a new exact immutable analysis with complete declarations for the selected retained
     *     plan
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the selected identity is no longer retained or the
     *     complete partition is outside bounded topology/resource discovery
     * @throws ArithmeticException if exact plan or resource arithmetic overflows
     */
    public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeSelected(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            SelectedCompletePlan selectedPlan) {
        Objects.requireNonNull(selectedPlan, "selectedPlan");
        Objects.requireNonNull(context, "context");
        requirePermittedSource(context);
        var units = decomposer.decompose(context, lowering);
        if (directPartition(context, units)) throw new IllegalArgumentException(
                "complete CPU plan selection is unavailable outside bounded topology/resource discovery");
        return finishOpenBlas(context, selectPlan(context, Optional.of(selectedPlan), units).analysis());
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> finishOpenBlas(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable) {
        return openBlasSelector.evaluate(context, portable.plan(), representationPlanner)
                .map(result -> withOpenBlasEvaluation(portable, result)).orElse(portable);
    }

    private SelectionAnalysis selectPlan(PrepareContext<CpuPartitionAnalysisInputs> context,
            Optional<SelectedCompletePlan> explicitSelection,
            List<CpuPartitionDagDecomposer.Unit> baseline) {
        Objects.requireNonNull(context, "context");
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> baselineAnalysis =
                analyzeTopology(context, baseline, null);
        List<CpuSpecializedSubgraph> recognition = recognizer.recognize(context,
                baselineAnalysis.plan().units());
        CpuPartitionDagDecomposer.Enumeration enumeration = decomposer.enumerate(context,
                lowering, recognition);
        var analyses = new ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>>();
        var candidates = new ArrayList<CpuFusionProfitabilitySelector.Candidate>();
        for (List<CpuPartitionDagDecomposer.Unit> topology : enumeration.candidates()) {
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                    sameTopology(topology, baseline) ? baselineAnalysis
                            : analyzeTopology(context, topology,
                                    baselineAnalysis.plan().boundaryValues());
            analyses.add(analysis);
            candidates.add(new CpuFusionProfitabilitySelector.Candidate(topology, analysis.plan()));
        }
        CpuFusionProfitabilitySelector.Result selected = explicitSelection.isPresent()
                ? selector.selectExact(context, enumeration, candidates,
                        explicitSelection.orElseThrow().identity().topology())
                : selector.select(context, enumeration, candidates);
        CpuRepresentationPlanner.Result representation = explicitSelection.isPresent()
                ? representationPlanner.selectExact(context, candidates, selected.decisions(),
                        explicitSelection.orElseThrow().identity())
                : representationPlanner.select(context, candidates, selected.decisions());
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> represented = withRepresentation(
                analyses.get(representation.candidateIndex()), representation);
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable = withMetadata(
                context, represented, recognition, selected.decisions(), baselineAnalysis.plan());
        return new SelectionAnalysis(portable, representation);
    }

    /**
     * Bounded retained complete-plan candidates. An incomplete result cannot form a public
     * tuning batch; a direct partition reports an empty identity list and {@code false}.
     * @param identities non-null ordered retained topology/representation identities;
     *     defensively snapshotted
     * @param complete whether every bounded alternative was proved complete; false for a
     *     partition outside the complete-candidate topology/resource domain
     */
    public record CompletePlanCandidates(
            List<CpuRepresentationDecision.VariantIdentity> identities,
            boolean complete) {
        /**
         * Validates and snapshots the ordered identities.
         *
         * @throws NullPointerException if {@code identities} or an element is {@code null}
         */
        public CompletePlanCandidates { identities = List.copyOf(identities); }
    }

    /**
     * Immutable authenticated request to select one exact already-retained complete CPU plan.
     * The value owns no resource and retains only immutable identity/fingerprint data.
     *
     * @param schemaVersion positive complete-plan schema version
     * @param identity non-null exact retained topology/representation identity
     * @param phaseOneFingerprint non-null non-empty unsigned canonical fingerprint octets;
     *     defensively snapshotted
     */
    public record SelectedCompletePlan(int schemaVersion,
            CpuRepresentationDecision.VariantIdentity identity,
            List<Integer> phaseOneFingerprint) {
        /**
         * Validates and snapshots the exact selection input.
         *
         * @throws NullPointerException if {@code identity}, {@code phaseOneFingerprint}, or an
         *     octet is {@code null}
         * @throws IllegalArgumentException if the schema is not positive, the fingerprint is
         *     empty, or an octet is outside {@code [0, 255]}
         */
        public SelectedCompletePlan {
            Objects.requireNonNull(identity, "identity");
            phaseOneFingerprint = List.copyOf(phaseOneFingerprint);
            if (schemaVersion <= 0 || phaseOneFingerprint.isEmpty()
                    || phaseOneFingerprint.stream().anyMatch(value -> value == null
                            || value < 0 || value > 255)) {
                throw new IllegalArgumentException("CPU complete-plan selection input is invalid");
            }
        }
    }

    private record SelectionAnalysis(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuRepresentationPlanner.Result representation) { }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withOpenBlasEvaluation(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuOpenBlasRouteSelector.Result result) {
        var plan = analysis.plan();
        Optional<CpuOpenBlasRoutePlan> nativePlan = result.selectedOpenBlasPlan();
        var selected = new CpuPartitionPreparationPlan(plan.units(),
                nativePlan.isPresent() ? CpuPartitionPreparationPlan.Route.OPENBLAS
                        : CpuPartitionPreparationPlan.Route.PORTABLE,
                plan.executionStrategy(),
                plan.bufferDeclarations(), plan.boundaryValues(), plan.accessBindings(),
                plan.carrierPattern(), plan.generatedCarrierPattern(), plan.extents(),
                plan.elementCount(), plan.affineAddressPairs(), plan.selectedRangeCount(),
                plan.minimumElementsPerWorker(), plan.vectorSpeciesBitSize(),
                plan.loweringManifest(), plan.materialization(), plan.workspaceDeclaration(),
                plan.workspaceUse(), plan.specializationBudget(), plan.movementGeometry(),
                plan.indexingGeometry(), plan.scatterGeometry(), plan.foldGeometry(),
                plan.orderingGeometry(), plan.randomGeometry(), plan.scanGeometry(),
                plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), plan.specializedSubgraphs(), plan.fusionDecisions(),
                plan.publicationBoundaryPositions(), plan.materializations(),
                plan.representationUnits(), plan.representationDecisions(),
                plan.partialReductionRecipe(), nativePlan, Optional.of(result.batch()),
                Optional.of(result.selected().identity()), plan.retainedBaselineBinding());
        var requirements = new ArrayList<PreparationResourceRequirement>(analysis.requirements());
        nativePlan.ifPresent(value -> requirements.addAll(value.workspaceRequirements()));
        return new BackendPartitionAnalysis<>(analysis.partition(), selected, requirements);
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withRepresentation(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuRepresentationPlanner.Result representation) {
        var plan = analysis.plan();
        var representedUnits = new ArrayList<CpuPartitionPreparationPlan.RepresentationUnitPlan>();
        for (int unitIndex = 0; !representation.materializations().isEmpty()
                && unitIndex < plan.units().size(); unitIndex++) {
            int representedUnitIndex = unitIndex;
            var unit = plan.units().get(unitIndex);
            var bindings = new ArrayList<>(unit.accessBindings());
            var carriers = new ArrayList<>(unit.generatedCarrierPattern());
            CpuKernelIr ir = unit.portablePlan().kernelIr();
            var representedMatmul=unit.portablePlan().specialization().matmulIr();
            boolean changed = false;
            for (CpuMaterializationPlan copy : representation.materializations()) {
                for (var consumer : copy.consumers()) {
                    if (consumer.unitPosition() != unitIndex) continue;
                    bindings.set(consumer.boundaryPosition(), copy.consumerBinding());
                    carriers.set(consumer.boundaryPosition(),
                            CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                    ir = adjustedIr(ir, consumer.boundaryPosition(),
                            copy.consumerBinding().plan());
                    if(representedMatmul.isPresent()) {
                        var matmul=representedMatmul.orElseThrow().withInputAccess(
                                consumer.boundaryPosition(),copy.consumerBinding().plan());
                        representedMatmul=Optional.of(matmul);ir=matmul.encodedKernelIr();
                    }
                    changed = true;
                }
            }
            CpuPortableRoutePlan route = unit.portablePlan();
            if (changed) {
                var old = route.specialization();
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),                         old.executionStrategy(), old.boundaryDataTypes(), carriers,
                        old.vectorSpeciesBitSize(), representation.materializations().size() == 1
                            ? representation.materializations().getFirst().consumers().stream()
                                .filter(value -> value.unitPosition() == representedUnitIndex)
                                .mapToInt(value -> value.boundaryPosition()).findFirst().orElse(-1)
                            : -1,
                        old.scalarPowerRealizations(), old.scratchParameter(),
                        old.classIdentitySchema(),representedMatmul);
                route=new CpuPortableRoutePlan(ir,specialization);
            }
            representedUnits.add(new CpuPartitionPreparationPlan.RepresentationUnitPlan(unitIndex,
                    route, bindings, carriers));
        }
        var requirements = new ArrayList<PreparationResourceRequirement>(analysis.requirements());
        for (CpuMaterializationPlan copy : representation.materializations()) requirements.add(
                new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                        copy.byteCount(), copy.byteAlignment()));
        var representedPlan = new CpuPartitionPreparationPlan(
                plan.units(), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), plan.specializedSubgraphs(), plan.fusionDecisions(),
                plan.publicationBoundaryPositions(), representation.materializations(),
                representation.materializations().isEmpty() ? List.of() : representedUnits,
                representation.decisions(), plan.partialReductionRecipe(),
                plan.openBlasPlan(), plan.openBlasTuningBatch(),
                plan.selectedOpenBlasTuningCandidate(), plan.retainedBaselineBinding());
        return new BackendPartitionAnalysis<>(analysis.partition(), representedPlan, requirements);
    }

    /**
     * Chooses existing one-unit analysis or complete DAG resource assembly for selected units.
     * Output-only count units require DAG assembly even alone because original source declarations
     * belong to their containing partition, not their computational entry.
     * @param context non-null immutable complete CPU projection, borrowed
     * @param units non-null selected disjoint units in stable topological order, borrowed
     * @param carrierReferenceValues original carrier-order value IDs, or {@code null} to derive
     *     carrier order from this partition's declarations
     * @return non-null immutable analysis with declarations selected before shared slot assignment
     * @throws IllegalArgumentException if unit or requested carrier facts cannot be analyzed
     * @throws ArithmeticException if checked resource geometry overflows
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeTopology(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> units, List<ValueId> carrierReferenceValues) {
        if (units.size() != 1 || units.getFirst().lowering().recurrentScan().isPresent()
                || units.getFirst().lowering().generalScan().isPresent()
                || units.getFirst().lowering().portableKernelIr()
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr)
            return analyzeDag(context, units, carrierReferenceValues);
        var unit = units.getFirst();
        PrepareContext<CpuPartitionAnalysisInputs> unitContext = context;
        if (carrierReferenceValues != null && !context.backendInputs().carrierPattern().isEmpty()) {
            var inputs = candidateInputs(context, unit.lowering().boundaryValues(),
                    carrierReferenceValues, true);
            unitContext = decomposer.unitContext(context, unit.nodes(), inputs);
        }
        return annotateSingle(analyzeUnit(unitContext, unit.lowering(), 0, true), unit);
    }

    private static CpuPartitionAnalysisInputs candidateInputs(
            PrepareContext<CpuPartitionAnalysisInputs> context, List<ValueId> boundaryValues,
            List<ValueId> carrierReferenceValues, boolean allowMaterialization) {
        List<CpuKernelSpecialization.CarrierAccess> requested =
                context.backendInputs().carrierPattern();
        if (requested.size() != carrierReferenceValues.size()) {
            throw new IllegalArgumentException(
                    "carrier pattern count must match the exact 0008B baseline buffers");
        }
        var carriers = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        for (int index = 0; index < carrierReferenceValues.size(); index++) {
            carriers.put(carrierReferenceValues.get(index), requested.get(index));
        }
        var projected = boundaryValues.stream().map(value -> carriers.getOrDefault(value,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)).toList();
        return new CpuPartitionAnalysisInputs(
                context.backendInputs().loweringManifestEnabled(), projected,
                context.backendInputs().portableExecution(),
                allowMaterialization ? context.backendInputs().materializationPolicy()
                        : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED,
                allowMaterialization
                        && context.backendInputs().conv2dMaterializedSuffixUnit(),
                context.backendInputs().partialReductionEvidence());
    }

    private static boolean sameTopology(List<CpuPartitionDagDecomposer.Unit> left,
            List<CpuPartitionDagDecomposer.Unit> right) {
        return left.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList()
                .equals(right.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList());
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withMetadata(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            List<CpuSpecializedSubgraph> facts,
            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision> decisions,
            CpuPartitionPreparationPlan baseline) {
        var plan = analysis.plan();
        var publications = new java.util.HashSet<ValueId>();
        context.memoryRequirements().stream().filter(
                io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement::graphOutput)
                .map(io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement::valueId)
                .forEach(publications::add);
        List<Integer> publicationBoundaryPositions = java.util.stream.IntStream.range(
                0, plan.boundaryValues().size()).filter(position ->
                        publications.contains(plan.boundaryValues().get(position))).boxed().toList();
        var enriched = new CpuPartitionPreparationPlan(
                plan.units(), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), facts, decisions, publicationBoundaryPositions,
                plan.materializations(), plan.representationUnits(),
                plan.representationDecisions(), plan.partialReductionRecipe(),
                plan.openBlasPlan(), plan.openBlasTuningBatch(),
                plan.selectedOpenBlasTuningCandidate(),
                baseline == null ? Optional.empty() : Optional.of(
                        CpuPartitionPreparationPlan.RetainedBaselineBinding.capture(
                                context.partitionDag(), baseline, plan.units(), publications)));
        return new BackendPartitionAnalysis<>(analysis.partition(), enriched,
                analysis.requirements());
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> annotateSingle(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuPartitionDagDecomposer.Unit topology) {
        var plan = analysis.plan();
        var unit = plan.units().getFirst();
        var facts = new CpuPartitionPreparationPlan.UnitRuntimeFacts(plan.affineAddressPairs(),
                plan.materialization(), plan.workspaceDeclaration(), plan.workspaceUse(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry());
        var enriched = new CpuPartitionPreparationPlan.ExecutionUnitPlan(unit.portablePlan(),
                unit.boundaryValues(), unit.accessBindings(), unit.carrierPattern(),
                unit.generatedCarrierPattern(), unit.extents(), unit.elementCount(),
                unit.executionStrategy(), unit.selectedRangeCount(), unit.minimumElementsPerWorker(),
                unit.vectorSpeciesBitSize(), unit.conv2dGeometry(), unit.conv3dGeometry(),
                unit.matmulGeometry(), unit.pool2dGeometry(), unit.pool3dGeometry(),
                unit.attentionGeometry(), unit.outputCount(),
                unit.fusionReason(), topology.dependencies(),
                topology.memberNodeOrdinals(), facts);
        var annotated = new CpuPartitionPreparationPlan(
                List.of(enriched), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), plan.specializedSubgraphs(), plan.fusionDecisions(),
                plan.publicationBoundaryPositions(), plan.materializations(),
                plan.representationUnits(), plan.representationDecisions(),
                plan.partialReductionRecipe());
        return new BackendPartitionAnalysis<>(analysis.partition(), annotated,
                analysis.requirements());
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeUnit(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering.LoweredPartition lowered, int workspaceRequirementId,
            boolean allowMaterialization) {
        return analyzeUnit(context, lowered, workspaceRequirementId, allowMaterialization, false);
    }

    /**
     * Analyzes real ordinary resources while preserving the selected affine pack protocol.
     * @param context exact projected ordinary unit
     * @param lowered actual physical lowering, including its unchanged address pack
     * @param workspaceRequirementId declaration ID for any family workspace
     * @param allowMaterialization whether ordinary external-read materialization may be selected
     * @param controlled whether a Scan helper uses controlled distinct-write ordinal triples
     * @return complete ordinary resource analysis with the actual pair or triple pack
     * @throws IllegalArgumentException if physical or resource facts disagree
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeUnit(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering.LoweredPartition lowered, int workspaceRequirementId,
            boolean allowMaterialization, boolean controlled) {
        if (lowered.recurrentScan().isPresent()) return analyzeRecurrent(context, lowered, workspaceRequirementId);
        CpuPartitionLowering.requirePermittedLowPrecisionUse(lowered.portableKernelIr());
        if (lowered.generalScan().isPresent()) return analyzeGeneralScan(context, lowered, workspaceRequirementId);
        var requestedCarriers = context.backendInputs().carrierPattern().isEmpty()
                ? java.util.Collections.nCopies(lowered.boundaryValues().size(),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : context.backendInputs().carrierPattern();
        if (requestedCarriers.size() != lowered.boundaryValues().size()) {
            throw new IllegalArgumentException("carrier pattern count must match boundary count");
        }
        var budget = new CpuSpecializationBudget(4, 1, 0, 0);
        boolean movement = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
        boolean indexing = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
        boolean scatter = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
        boolean fold = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
        boolean ordering = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
        boolean random = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
        boolean scan = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
        boolean aggregate = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
        boolean aggregateFusion = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr;
        boolean softmaxBackward = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr;
        boolean constantWrite = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr;
        boolean argExtrema = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
        boolean maskedReduction = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
        boolean advancedReduction = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
        boolean softmax = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
        boolean trailingNormalization = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
        boolean batchNormalization = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
        boolean batchNormTraining = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr;
        boolean conv2d = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
        boolean conv3d = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
        boolean matmul = lowered.matmulIr().isPresent();
        boolean rowRegion = lowered.portableKernelIr() instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr;
        boolean pool2d = lowered.pool2dGeometry().isPresent();
        boolean pool3d = lowered.pool3dGeometry().isPresent();
        boolean attention = lowered.attentionGeometry().isPresent();
        boolean loss = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
        Optional<CpuMaterializationPlan> materialization = Optional.empty();
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(lowered.boundaryValues().size());
        for (int i = 0; i < lowered.boundaryValues().size(); i++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(i),
                        Math.multiplyExact(lowered.referencedElementSpans().get(i),
                                lowered.boundaryDataTypes().get(i).byteWidth()),
                        lowered.boundaryDataTypes().get(i).byteWidth()));
        var bindings = new ArrayList<>(lowered.accessBindings());
        var carriers = new ArrayList<>(requestedCarriers);
        CpuKernelIr kernelIr = lowered.kernelIr();
        boolean affineCopy = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
        if (materialization.isPresent()) {
            var selected = materialization.orElseThrow();
            bindings.set(selected.sourceBoundaryIndex(), selected.consumerBinding());
            carriers.set(selected.sourceBoundaryIndex(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
            kernelIr = adjustedIr(kernelIr, selected.sourceBoundaryIndex(),
                    selected.consumerBinding().plan());
        }
        var config = context.backendInputs().portableExecution();
        DataType vectorType = config.computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                ? vectorLaneType(kernelIr) : null;
        int lanes = speciesLanes(vectorType);
        int speciesBits = speciesBits(vectorType);
        boolean matmulVector = matmul && config.computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && (lowered.matmulIr().orElseThrow().realization()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_N_VECTOR
                ||lowered.matmulIr().orElseThrow().realization()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.TILED_N_VECTOR_2X2);
        boolean lossVector = loss && config.computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorType != null && lanes > 1 && lowered.elementCount() >= lanes
                && vectorLossEligible((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr)
                        lowered.portableKernelIr(), bindings);
        boolean directConvOperation = !context.nodes().isEmpty()
                && (conv2d && context.nodes().getFirst().operation().kind()
                    == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D
                || conv3d && context.nodes().getFirst().operation().kind()
                    == io.github.pho001.synaptik.model.operation.convolution.Conv3dKind.CONV3D);
        boolean convVector = directConvOperation
                && config.computePreference()
                    == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorType != null && lanes > 1 && lowered.elementCount() >= lanes
                && convVectorEligible(lowered, bindings, lanes);
        boolean vectorEligible = !softmaxBackward && !aggregateFusion && !constantWrite && !rowRegion && (matmulVector || lossVector || convVector
                || !loss && !matmul && !pool2d && !pool3d && !attention && !affineCopy && !movement
                && !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate
                && !argExtrema && !maskedReduction && !advancedReduction && !softmax
                && !trailingNormalization && !batchNormalization && !batchNormTraining
                && !conv2d && !conv3d && config.computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorType != null && lanes > 1 && lowered.elementCount() >= lanes
                && vectorTopologyEligible(kernelIr, vectorType)
                && (!hasDenseFloatingMaskBoundary(kernelIr, vectorType) || lanes <= Long.SIZE)
                && bindings.stream().allMatch(binding -> vectorEligible(binding, lanes)));
        int usableParallelism = Math.min(config.configuredMaximumParallelism(),
                config.availableParallelism());
        var rowExec = rowRegion ? rowEmission(context,
                (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)lowered.portableKernelIr(),null)
                : Optional.<CpuKernelSpecialization.RowFusionEmission>empty();
        if (rowExec.isPresent()) vectorEligible = config.computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorRowEligible((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)lowered.portableKernelIr());
        long iterationCount = lowered.elementCount();
        long minimumRangeItemsPerWorker = config.minimumElementsPerWorker();
        long[] selectedExtents = lowered.extents();
        var selectedPortableIr = lowered.portableKernelIr();
        var selectedBatchGeometry = lowered.batchNormInferenceGeometry();
        var selectedTrainingGeometry = lowered.batchNormTrainingGeometry();
        int selectedRangeCount;
        if (aggregateFusion || softmaxBackward) {
            selectedRangeCount = 1;
            minimumRangeItemsPerWorker = 1;
        } else if (rowRegion) {
            selectedRangeCount = rowExec.map(CpuKernelSpecialization.RowFusionEmission::maximumPartialCount).orElse(1);
            minimumRangeItemsPerWorker = rowExec.isPresent() ? config.minimumElementsPerWorker() : 1;
        } else if (matmul) {
            var geometry = lowered.matmulGeometry().orElseThrow();
            var form=lowered.matmulIr().orElseThrow().realization();
            iterationCount = switch (form) {
                case DIRECT_SCALAR -> geometry.outputCount();
                case DIRECT_N_VECTOR -> Math.multiplyExact(geometry.batchCount(), geometry.m());
                case TILED_SCALAR_2X2 -> Math.multiplyExact(geometry.batchCount(),
                        Math.multiplyExact(ceilDiv(geometry.m(), 2), ceilDiv(geometry.n(), 2)));
                case TILED_N_VECTOR_2X2 -> Math.multiplyExact(geometry.batchCount(),
                        Math.multiplyExact(ceilDiv(geometry.m(), 2),
                                ceilDiv(geometry.n(), Math.multiplyExact(2L, lanes))));
            };
            long coordinatesPerWorkUnit = iterationCount == 0 ? 1
                    : Math.max(1, geometry.outputCount() / iterationCount);
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(), coordinatesPerWorkUnit));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            selectedExtents = new long[] {iterationCount};
        } else if (batchNormalization) {
            var geometry = lowered.batchNormInferenceGeometry().orElseThrow();
            var form = usableParallelism >= 2 && geometry.channelCount() < geometry.nonChannelCount()
                    ? io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE
                    : io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE;
            geometry = geometry.withRangeForm(form);
            iterationCount = geometry.rangeItemCount();
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(),
                            geometry.coordinatesPerRangeItem()));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            if (selectedRangeCount < 2
                    && form != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE) {
                geometry = geometry.withRangeForm(
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE);
                iterationCount = geometry.rangeItemCount();
                minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                        ceilDiv(config.minimumElementsPerWorker(),
                                geometry.coordinatesPerRangeItem()));
                selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                        Math.toIntExact(Math.min(Integer.MAX_VALUE,
                                ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            }
            selectedBatchGeometry = Optional.of(geometry);
            selectedPortableIr = ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr)
                    lowered.portableKernelIr()).withRangeForm(geometry.rangeForm());
            kernelIr = ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr)
                    selectedPortableIr).encodedKernelIr();
            selectedExtents = new long[] {iterationCount};
        } else if (batchNormTraining) {
            var geometry = lowered.batchNormTrainingGeometry().orElseThrow();
            iterationCount = geometry.channelCount();
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(), geometry.reductionCount()));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            selectedExtents = new long[]{iterationCount};
        } else {
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
        }
        Optional<CpuPartitionPreparationPlan.PartialReductionRecipe> partialRecipe = Optional.empty();
        if (aggregate && lowered.boundaryValues().size() == 2
                && lowered.virtualValues().isEmpty() && bindings.size() == 2
                && bindings.stream().allMatch(binding -> binding.plan().regime()
                    == CpuAccessPlan.Regime.DENSE_LINEAR)) {
            var aggregateIr = (CpuAggregateIr) lowered.portableKernelIr();
            var evidence = context.backendInputs().partialReductionEvidence();
            /*
             * 0008P has no valid, frozen evidence root.  Analysis inputs are supplied by normal
             * callers and are therefore not an evidence-verification boundary: accepting their
             * boolean would let a caller manufacture a passing profitability result.  Leave the
             * recipe unreachable until a later, separately scoped trusted-evidence reader is
             * implemented and binds every required hash-valid row to this exact selector.
             */
            boolean matchingEvidence = false;
            boolean primitiveArrays = aggregateIr.dataType() == DataType.INT32
                    ? carriers.equals(List.of(CpuKernelSpecialization.CarrierAccess.INT_ARRAY,
                            CpuKernelSpecialization.CarrierAccess.INT_ARRAY))
                    : carriers.equals(List.of(CpuKernelSpecialization.CarrierAccess.LONG_ARRAY,
                            CpuKernelSpecialization.CarrierAccess.LONG_ARRAY));
            if (matchingEvidence && primitiveArrays) {
                partialRecipe = partialReductionLowering.admit(
                        new CpuPartialReductionLowering.AdmissionInputs(aggregateIr.kind(),
                                aggregateIr.dataType(), aggregateIr.form(), lowered.elementCount(),
                                aggregateIr.domainCount(), evidence.partialCount(),
                                config.minimumElementsPerWorker(), usableParallelism, true, true,
                                true, true)).map(ir -> new CpuPartitionPreparationPlan
                                        .PartialReductionRecipe(ir,
                                                CpuPartitionPreparationPlan.PartialReductionRecipe
                                                        .artifactIdentity(ir)));
            }
        }
        if (partialRecipe.isPresent()) selectedRangeCount = partialRecipe.orElseThrow().ir()
                .partialCount();
        boolean parallel = selectedRangeCount >= 2;
        var strategy = vectorEligible
                ? (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)
                : (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
        var artifactStrategy = vectorEligible
                ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        if (rowExec.isPresent()) artifactStrategy = strategy;
        if (matmul) {
            var ir = lowered.matmulIr().orElseThrow();
            boolean vectorForm = ir.realization()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_N_VECTOR
                    || ir.realization()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
            if (vectorForm != vectorEligible
                    || ir.preferredSpeciesBitSize() != (vectorEligible ? speciesBits : 0)) {
                throw new IllegalArgumentException("MATMUL realization and generated strategy disagree");
            }
        }
        if (partialRecipe.isPresent()) {
            strategy = CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR;
            artifactStrategy = CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        }
        var specialization = ordinarySpecialization(kernelIr, selectedPortableIr, lowered.matmulIr(),
                lowered.boundaryDataTypes(), carriers, artifactStrategy,
                vectorEligible ? speciesBits : 0,
                materialization.map(CpuMaterializationPlan::sourceBoundaryIndex).orElse(-1),
                rowExec.isPresent() || ordinaryScratchParameter(selectedPortableIr),
                convVector, vectorType, rowExec);
        selectedPortableIr = matmul||materialization.isPresent()?kernelIr:selectedPortableIr;
        var routePlan = new CpuPortableRoutePlan(selectedPortableIr, specialization);
        String manifest = "unit=0;fusion=" + lowered.fusionReason()
                + ";access=" + bindings.stream()
                        .map(binding -> binding.plan().regime().name()).toList()
                + ";carriers=" + carriers
                + ";route=PORTABLE;strategy=" + strategy + ";speciesBits="
                + (vectorEligible ? speciesBits : 0) + ";power="
                + powerRealizations(kernelIr) + ";key="
                + specialization.structuralKey() + ";buffers=" + lowered.boundaryValues()
                + partialRecipe.map(recipe -> ";partialReduction="
                        + recipe.generatedArtifactIdentity()).orElse("");
        Optional<PreparationResourceRequirement.Workspace> workspace = materialization.map(copy ->
                    new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                            copy.byteCount(), copy.byteAlignment()));
        if (rowExec.isPresent()) workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                rowWorkspaceBytes((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)lowered.portableKernelIr(),rowExec.orElseThrow()),8));
        if (lowered.scatterGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent()) {
            var scatterGeometry = lowered.scatterGeometry().orElseThrow();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    scatterGeometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        if (lowered.orderingGeometry().isPresent()) {
            var geometry = lowered.orderingGeometry().orElseThrow();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    geometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        if (lowered.aggregateGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            var geometry = lowered.aggregateGeometry().orElseThrow();
            long aggregateBytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && aggregateBytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "aggregate exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    aggregateBytes, Long.BYTES));
        }
        if (lowered.maskedReductionGeometry().filter(g -> g.outputCount() > 0).isPresent()) {
            var geometry = lowered.maskedReductionGeometry().orElseThrow();
            long maskedBytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && maskedBytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "masked-reduction exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    maskedBytes, Long.BYTES));
        }
        if (lowered.advancedReductionGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            var geometry = lowered.advancedReductionGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "advanced-reduction exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    bytes, Long.BYTES));
        }
        if (lowered.trailingNormalizationGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent()) {
            var geometry = lowered.trailingNormalizationGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "trailing-normalization exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId, bytes,
                    Long.BYTES));
        }
        if (lowered.batchNormTrainingGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent()) {
            var geometry = lowered.batchNormTrainingGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "batch-normalization training exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId, bytes, Long.BYTES));
        }
        if (lowered.attentionGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent()) {
            var geometry=lowered.attentionGeometry().orElseThrow();
            workspace=Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    geometry.workspaceBytes(selectedRangeCount),Long.BYTES));
        }
        if (partialRecipe.isPresent()) {
            long partialBytes = partialRecipe.orElseThrow().ir().workspaceBytes();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    partialBytes, Long.BYTES));
        }
        var workspaceUse = materialization.isPresent()
                ? CpuPartitionPreparationPlan.WorkspaceUse.MATERIALIZATION
                : workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.SCATTER_PRODUCT
                : CpuPartitionPreparationPlan.WorkspaceUse.NONE;
        if (rowExec.isPresent()) workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.GENERAL_SCAN_BANKS;
        if (lowered.orderingGeometry().isPresent() && workspace.isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.ORDERING_INDICES;
        if (lowered.aggregateGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.maskedReductionGeometry().filter(g -> g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.advancedReductionGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.trailingNormalizationGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.batchNormTrainingGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.attentionGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.ATTENTION_ROW_STATE;
        if (partialRecipe.isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        boolean affineOrdinals = controlled && lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr affine
                && affine.writeDomain()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES;
        var rowRuntime = affineOrdinals ? CpuPartitionPreparationPlan.UnitRuntimeFacts.controlledAffineCopy()
                : rowExec.isEmpty() ? CpuPartitionPreparationPlan.UnitRuntimeFacts.EMPTY
                : new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0],Optional.empty(),workspace,workspaceUse,
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty());
        var plan = new CpuPartitionPreparationPlan(
                                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.boundaryValues(), bindings, requestedCarriers, carriers,
                        selectedExtents, iterationCount, strategy, selectedRangeCount,
                        minimumRangeItemsPerWorker, vectorEligible ? speciesBits : 0,
                        lowered.conv2dGeometry(), lowered.conv3dGeometry(), lowered.matmulGeometry(),
                        lowered.pool2dGeometry(), lowered.pool3dGeometry(), lowered.attentionGeometry(),
                        (int) kernelIr.values().stream()
                            .filter(value -> value.kind() == CpuKernelIr.Value.Kind.OUTPUT).count(),
                        lowered.fusionReason(), List.of(), List.of(),
                        rowRuntime)),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy,
                declarations, lowered.boundaryValues(), bindings,
                requestedCarriers, carriers,
                selectedExtents, iterationCount, lowered.affineAddressPairs(),
                selectedRangeCount, minimumRangeItemsPerWorker,
                vectorEligible ? speciesBits : 0,
                context.backendInputs().loweringManifestEnabled() ? manifest : "",
                materialization, workspace, workspaceUse, budget,
                lowered.movementGeometry(), lowered.indexingGeometry(), lowered.scatterGeometry(),
                lowered.foldGeometry(), lowered.orderingGeometry(), lowered.randomGeometry(),
                lowered.scanGeometry(), lowered.aggregateGeometry(), lowered.argExtremaGeometry(),
                lowered.maskedReductionGeometry(), lowered.advancedReductionGeometry(),
                lowered.softmaxGeometry(), lowered.trailingNormalizationGeometry(),
                selectedBatchGeometry, selectedTrainingGeometry, lowered.conv2dGeometry(),
                List.of(), List.of(), List.of(), List.of(), List.of(), List.of(),
                partialRecipe);

        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.workspaceDeclaration().ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    /**
     * Selects the existing ordinary code-shaping namespace from typed IR, not numeric extents.
     * Static analysis and symbolic controlled helpers share this constructor so family math,
     * conversion roles and scalar identities cannot diverge.
     * @param kernel exact ordinary generation envelope
     * @param ir exact represented portable family
     * @param matrix separate typed MATMUL facts, empty for every other family
     * @param types ordered physical boundary types
     * @param carriers ordered physical carrier forms
     * @param strategy already selected scalar/vector orchestration
     * @param speciesBits selected vector width, or zero for scalar
     * @param materializedSource copied source position, or minus one
     * @param scratch whether the family entry ABI includes scratch
     * @param convVector whether the qualified direct-convolution vector projection applies
     * @param vectorType selected vector lane type, nullable for scalar
     * @param rowEmission selected row execution facts, empty for inline ordinary helpers
     * @return exact ordinary specialization; dense heap affine copies alone use projection 73
     *     and its selective empty-range compatibility marker
     */
    private static CpuKernelSpecialization ordinarySpecialization(CpuKernelIr kernel,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr ir,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr> matrix,
            List<DataType> types, List<CpuKernelSpecialization.CarrierAccess> carriers,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy, int speciesBits,
            int materializedSource, boolean scratch, boolean convVector, DataType vectorType,
            Optional<CpuKernelSpecialization.RowFusionEmission> rowEmission) {
        boolean vector = speciesBits != 0;
        boolean crossTypeCast = kernel.instructions().stream().anyMatch(instruction ->
                instruction.opcode() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST
                        && kernel.values().get(instruction.inputs().getFirst()).dataType()
                        != kernel.values().get(instruction.output()).dataType());
        boolean pointwise = kernel.familyIdentity().equals("pointwise");
        var row = ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr value
                ? Optional.of(value) : Optional.<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr>empty();
        var fusion = ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr value
                ? Optional.of(value) : Optional.<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr>empty();
        var backward = ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr value
                ? Optional.of(value) : Optional.<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr>empty();
        boolean denseArrayAffine = ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr
                && !carriers.contains(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && kernel.values().stream().allMatch(value ->
                        value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                        || value.accessPlan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
        int schema = row.isPresent() ? 72 : convVector ? 63
                : denseArrayAffine ? CpuKernelSpecialization.DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA
                : ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr ? 57
                : ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr ? 56
                : ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr ? 55
                : matrix.isPresent() ? 54
                : ir instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr ? vector ? 62 : 58
                : pointwise && kernel.values().stream().anyMatch(value -> value.dataType() == DataType.FLOAT16) ? 68
                : pointwise && crossTypeCast ? 60
                : vector && hasDenseFloatingMaskBoundary(kernel, vectorType) ? 61
                : pointwise && kernel.values().stream().anyMatch(value -> value.dataType() == DataType.BFLOAT16) ? 59 : 52;
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(kernel.structuralKey()),
                strategy, types, carriers, speciesBits, materializedSource, powerRealizations(kernel),
                scratch, schema, matrix, Optional.empty(), false, false, row, rowEmission, fusion, backward);
    }

    /**
     * Retains each ordinary family's scratch ABI independently of an empty invocation domain.
     * @param ir exact portable code-shaping facts
     * @return whether the standalone family method declares its existing scratch parameter
     */
    private static boolean ordinaryScratchParameter(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr ir) {
        return switch (ir) {
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr ignored -> true;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr ignored -> true;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr ignored -> true;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr value ->
                    value.aggregate().scratchSliceBytes() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr value ->
                    value.aggregate().scratchSliceBytes() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr value -> value.scratchSignature() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr value -> value.scratchSliceBytes() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr value -> value.scratchSliceBytes() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr value -> value.scratchSliceBytes() > 0;
            case io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr value -> value.scratchSliceBytes() > 0;
            default -> false;
        };
    }

    /**
     * Identifies direct decompositions outside complete generated-plan discovery. This is a
     * candidate-protocol limit, not a limit on supported ordinary graphs or variadic movement.
     * One indivisible single occurrence retains its independently checked wider entry budget;
     * all other bounded candidates keep sixteen boundaries per unit and sixty-four in total.
     * @param context non-null complete CPU projection, inspected without reading storage
     * @param units exact already-lowered complete direct decomposition
     * @return whether the complete decomposition requires direct analysis without candidate facts
     */
    private static boolean directPartition(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> units) {
        if (context.nodes().size() > 8 || context.nodes().stream().anyMatch(n ->
                n.operation().kind() instanceof io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind
                || n.operation().kind() == io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN)) return true;
        if (units.size() == 1 && units.getFirst().nodes().size() == 1
                && !units.getFirst().pointwise()) return false;
        int boundaryPositions = 0;
        for (var unit : units) {
            int count = unit.lowering().boundaryValues().size();
            if (count > 16) return true;
            boundaryPositions = Math.addExact(boundaryPositions, count);
        }
        return boundaryPositions > 64;
    }
    /** Authenticates source roles, including every nested body, before resource declarations. */
    private static void requirePermittedSource(PrepareContext<CpuPartitionAnalysisInputs> context) {
        requirePermittedSource(context.nodes(), context.values(), context.scanActivityPlans());
    }

    /**
     * Checks each lexical body independently; local value IDs never resolve against outer values.
     * @param nodes exact nodes in this lexical graph
     * @param graphValues exact descriptors in the same graph
     * @param activity activity annotations belonging only to this graph
     * @throws IllegalArgumentException if any nested occurrence performs locked low arithmetic
     */
    private static void requirePermittedSource(
            List<io.github.pho001.synaptik.model.graph.CompiledNode> nodes,
            List<io.github.pho001.synaptik.model.graph.GraphValue> graphValues,
            Map<io.github.pho001.synaptik.model.graph.NodeId,
                    io.github.pho001.synaptik.planning.capability.ScanActivityPlan> activity) {
        boolean lowValue = graphValues.stream().anyMatch(value ->
                value.descriptor().dataType() == DataType.BFLOAT16
                        || value.descriptor().dataType() == DataType.FLOAT16);
        var values = new java.util.HashMap<ValueId, io.github.pho001.synaptik.model.tensor.TensorDescriptor>();
        if (lowValue) for (var value : graphValues) values.put(value.id(), value.descriptor());
        for (var node : nodes) {
            if (lowValue) {
                var query = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(
                        node.operation(), node.inputs().stream().map(values::get).toList(),
                        node.outputs().stream().map(values::get).toList(), node.scanBody(),
                        Optional.ofNullable(activity.get(node.id())));
                if (!CpuPartitionLowering.permitsLowPrecisionUse(query))
                    throw new IllegalArgumentException("CPU low numerical computation is temporarily unsupported");
            }
            node.scanBody().ifPresent(body ->
                    requirePermittedSource(body.graph().nodes(), body.graph().values(), Map.of()));
        }
    }


    /**
     * Declares referenced scan storage spans and two packed working rows before assignment.
     * Workspace width/alignment is eight bytes for double work and four otherwise; low storage
     * never makes the working projections low precision. Empty time/batch omits scratch.
     * @param context non-null single-occurrence projection; no values are read
     * @param lowered non-null direct scan lowering
     * @param workspaceId non-negative final or temporary unit workspace identifier
     * @return immutable direct analysis, containing no generated plan or artifact
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeRecurrent(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering.LoweredPartition lowered, int workspaceId) {
        var scan = lowered.recurrentScan().orElseThrow();
        if (scan.geometry().dataType() != DataType.FLOAT32 && scan.geometry().dataType() != DataType.FLOAT64)
            throw new IllegalArgumentException("CPU low recurrent computation is temporarily unsupported");
        var carriers = context.backendInputs().carrierPattern().isEmpty()
                ? java.util.Collections.nCopies(scan.boundaryValues().size(),
                    CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : context.backendInputs().carrierPattern();
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>();
        for (int p = 0; p < scan.boundaryValues().size(); p++) declarations.add(
                new PreparationResourceRequirement.Buffer(scan.boundaryValues().get(p),
                    Math.multiplyExact(scan.spans().get(p), scan.boundaryTypes().get(p).byteWidth()),
                    scan.boundaryTypes().get(p).byteWidth()));
        Optional<PreparationResourceRequirement.Workspace> workspace = scan.geometry().scratchBytes() == 0
                ? Optional.empty() : Optional.of(new PreparationResourceRequirement.Workspace(
                    workspaceId, scan.geometry().scratchBytes(), scan.geometry().workingBytes()));
        var use = workspace.isEmpty() ? CpuPartitionPreparationPlan.WorkspaceUse.NONE
                : CpuPartitionPreparationPlan.WorkspaceUse.RECURRENT_ROW_STATE;
        var facts = new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0],
                Optional.empty(), workspace, use, Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(),
                Optional.of(scan), Optional.empty());
        var unit = new CpuPartitionPreparationPlan.ExecutionUnitPlan(null, scan.boundaryValues(),
                scan.accessBindings(), carriers, carriers, lowered.extents(), lowered.elementCount(),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, 1, 1, 0,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), scan.boundaryValues().size() - scan.readCount(),
                lowered.fusionReason(), List.of(), List.of(0), facts);
        var plan = new CpuPartitionPreparationPlan(List.of(unit),
                CpuPartitionPreparationPlan.Route.PORTABLE, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                declarations, scan.boundaryValues(), scan.accessBindings(), carriers, carriers,
                lowered.extents(), lowered.elementCount(), new long[0], 1, 1, 0, "direct fixed recurrent scan",
                Optional.empty(), workspace, use, new CpuSpecializationBudget(4, 1, 0, 0),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(),
                List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), Optional.empty());
        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        workspace.ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    /**
     * Declares demanded outputs, banks, live cuts, controlled ordinary-helper scratch and native
     * control before assignment. Composite helpers reuse exact local ordinary analysis, or the
     * family's symbolic numeric recipe, with scalar single-range non-native inputs. Their typed
     * specializations and owner cursor resets become
     * immutable generation facts. Their control starts after the physical value/carry inventory,
     * and each helper's aligned scratch follows the complete control slabs. Retained staged scans
     * keep their existing resource/strategy policy. No storage or control values are read here.
     * @param context exact single-unit projection, inspected without storage reads
     * @param lowered proved generated scan and positional boundaries
     * @param workspaceId exact unit-local workspace declaration identity
     * @return immutable generated analysis with checked complete resource declarations
     * @throws IllegalArgumentException if requested carriers or exact resource facts disagree
     * @throws ArithmeticException if declared typed byte counts overflow
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeGeneralScan(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering.LoweredPartition lowered, int workspaceId) {
        var scan = lowered.generalScan().orElseThrow();
        var compositionResources = scan.ir().composition().isPresent()
                ? Optional.of(analyzeCompositionResources(context, scan)) : Optional.<CompositionResources>empty();
        if (compositionResources.isPresent()) scan = compositionResources.orElseThrow().scan();
        var region = scan.ir().stages().flatMap(s -> s.rowMapReductionIr());
        var sourceScan = scan.ir();
        var emission = region.flatMap(r -> rowEmission(context,r,sourceScan));
        boolean vectorRow = emission.isPresent() && context.backendInputs().portableExecution().computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorRowEligible(region.orElseThrow());
        if (emission.isPresent()) scan = rowScanResources(scan,emission.orElseThrow());
        int width = compositionResources.map(CompositionResources::alignment)
                .orElseGet(() -> lowered.generalScan().orElseThrow().ir().dataType().byteWidth());
        var carriers = java.util.Collections.nCopies(lowered.boundaryValues().size(),
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        if (!context.backendInputs().carrierPattern().isEmpty()
                && !context.backendInputs().carrierPattern().equals(carriers))
            throw new IllegalArgumentException("general scan requires checked segment carriers");
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>();
        var batchSizes = scan.ir().symbolic() && scan.ir().composition().isEmpty()
                ? CpuPartitionLowering.batchByteSizes(context) : java.util.Map
                        .<ValueId,io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize>of();
        for (int p = 0; p < lowered.boundaryValues().size(); p++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(p),
                    scan.ir().symbolic() ? scan.ir().composition().isPresent()
                            ? scan.ir().composition().orElseThrow().boundarySizes().get(p)
                            : batchSizes.get(lowered.boundaryValues().get(p))
                        : new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(
                            Math.multiplyExact(scan.spans().get(p), (long) scan.ir().boundaryType(p).byteWidth())),
                    scan.ir().boundaryType(p).byteWidth()));
        Optional<PreparationResourceRequirement.Workspace> workspace = !scan.ir().symbolic() && scan.geometry().scratchBytes() == 0
                ? Optional.empty() : Optional.of(new PreparationResourceRequirement.Workspace(
                    workspaceId, scan.geometry().workspaceSize(), emission.isPresent() ? 8 : width));
        var use = workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.GENERAL_SCAN_BANKS
                : CpuPartitionPreparationPlan.WorkspaceUse.NONE;
        int ranges = emission.map(CpuKernelSpecialization.RowFusionEmission::maximumPartialCount).orElse(1);
        var strategy = vectorRow ? ranges > 1 ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR : ranges > 1
                ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        int bits = vectorRow ? speciesBits(scan.ir().dataType()) : 0;
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(scan.ir().structuralKey()),
                strategy, lowered.boundaryDataTypes(), carriers,
                bits, -1, List.of(), workspace.isPresent(), 71, Optional.empty(), Optional.of(scan.ir()),
                false,false,Optional.empty(),emission,Optional.empty(),Optional.empty(),
                compositionResources.map(CompositionResources::units).orElseGet(List::of));
        var route = new CpuPortableRoutePlan(scan.ir(), specialization);
        var runtime = new CpuPartitionPreparationPlan.UnitRuntimeFacts(new long[0], Optional.empty(), workspace, use,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(scan));
        var unit = new CpuPartitionPreparationPlan.ExecutionUnitPlan(route, lowered.boundaryValues(),
                scan.accessBindings(), carriers, carriers, lowered.extents(), lowered.elementCount(),
                strategy, ranges, emission.isPresent() ? context.backendInputs().portableExecution().minimumElementsPerWorker() : 1, bits,
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), scan.ir().demandedPorts().size(), lowered.fusionReason(), List.of(), List.of(0), runtime);
        var plan = new CpuPartitionPreparationPlan(List.of(unit), CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy, declarations, lowered.boundaryValues(),
                scan.accessBindings(), carriers, carriers, lowered.extents(), lowered.elementCount(),
                new long[0], ranges, unit.minimumElementsPerWorker(), bits, lowered.fusionReason(), Optional.empty(), workspace, use,
                new CpuSpecializationBudget(4, 1, 0, 0), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), List.of(), List.of(), List.of(),
                List.of(), List.of(), List.of(), Optional.empty());
        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        workspace.ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    /**
     * Immutable cold inventory; numeric offsets are workspace-relative and never slot identities.
     * @param scan unchanged physical IR with the complete appended workspace declaration
     * @param units exact ordinary specializations in physical composition order
     * @param alignment maximum typed/helper/control byte alignment, at least eight
     */
    private record CompositionResources(
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts scan,
            List<CpuKernelSpecialization.CompositionUnit> units, int alignment) {}

    /**
     * Reuses ordinary resource analysis in the exact local lexical scope. Symbolic families supply
     * their checked range-capacity/scratch/reset recipes directly, without resolving a sampled
     * B. Binding retains the exact range end in the numeric pack when empty-domain rules make it
     * piecewise. The physical inventory ends at the control origin; appended storage changes Geometry only,
     * not that origin. Control contains one helper slab and one slab per distinct carried/seed
     * declaration root. Both affine coefficients are aligned before each disjoint helper scratch.
     * @param context containing single-Scan projection, with its exact activity association
     * @param scan independently proved composition and ordinary lowerings
     * @return complete typed helper inventory and total declaration, without resource realization
     * @throws IllegalArgumentException if local membership, scalar analysis or resource facts disagree
     * @throws ArithmeticException if resource coefficients overflow
     */
    private CompositionResources analyzeCompositionResources(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts scan) {
        var composition = scan.ir().composition().orElseThrow();
        if (composition.units().size() != scan.ordinaryUnits().size())
            throw new IllegalArgumentException("Scan ordinary helper inventory disagrees");
        var node = context.nodes().getFirst();
        var descriptors = new LinkedHashMap<ValueId, io.github.pho001.synaptik.model.tensor.TensorDescriptor>();
        context.values().forEach(value -> descriptors.put(value.id(), value.descriptor()));
        var query = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(
                node.operation(), node.inputs().stream().map(descriptors::get).toList(),
                node.outputs().stream().map(descriptors::get).toList(), node.scanBody(),
                Optional.ofNullable(context.scanActivityPlans().get(node.id())));
        var local = composition.units().isEmpty() ? null
                : io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanCompositionLowering
                        .bodyContext(query, CpuPartitionAnalysisInputs.DEFAULT);
        if (local != null) requirePermittedSource(local);
        int alignment = Long.BYTES;
        for (var value : composition.values()) alignment = Math.max(alignment, value.descriptor().dataType().byteWidth());
        int roots = composition.activity().map(activity ->
                Math.addExact(activity.carries().size(), activity.seedFormals().size())).orElse(0);
        long slabs = Math.addExact(1L, roots);
        var batch = compositionBatchSize(composition);
        var cursor = addCompositionBytes(composition.workspaceSize(),
                new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                        Math.multiplyExact(slabs, Math.addExact(16L, batch.constantBytes())),
                        Math.multiplyExact(slabs, batch.bytesPerBatch()), 1));
        var units = new ArrayList<CpuKernelSpecialization.CompositionUnit>(composition.units().size());
        for (int ordinal = 0; ordinal < composition.units().size(); ordinal++) {
            var physical = composition.units().get(ordinal);
            var ordinary = scan.ordinaryUnits().get(ordinal);
            CpuPartitionLowering.requirePermittedLowPrecisionUse(physical.ir());
            CpuPartitionLowering.requirePermittedLowPrecisionUse(ordinary.portableKernelIr());
            var members = physical.memberNodeOrdinals().stream().map(local.nodes()::get).toList();
            var unitContext = decomposer.unitContext(local, members, CpuPartitionAnalysisInputs.DEFAULT);
            CpuKernelSpecialization helper;
            io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes;
            io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterations;
            List<CpuKernelSpecialization.PackReset> resets;
            int scratchAlignment;
            if (ordinary.batchGeometry().isPresent()) {
                var numeric = ordinary.batchGeometry().orElseThrow();
                CpuPartitionLowering.requirePermittedLowPrecisionUse(numeric.ir());
                scratchBytes = numeric.scratchBytes();
                iterations = numeric.iterationCount();
                scratchAlignment = Math.max(Long.BYTES, scratchBytes.elementByteWidth());
                var carriers = java.util.Collections.nCopies(numeric.boundaryTypes().size(),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                helper = ordinarySpecialization(ordinary.kernelIr(), numeric.ir(), numeric.matrixIr(),
                        numeric.boundaryTypes(), carriers, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        0, -1, ordinaryScratchParameter(numeric.ir())
                                || scratchBytes.constantBytes() != 0 || scratchBytes.bytesPerBatch() != 0,
                        false, null, Optional.empty());
                resets = numeric.resets();
            } else {
                var analysis = analyzeUnit(unitContext, ordinary, 0, false, true).plan();
                var unit = analysis.units().getFirst();
                if (analysis.units().size() != 1
                        || unit.executionStrategy() != CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR
                        || unit.selectedRangeCount() != 1 || analysis.materialization().isPresent())
                    throw new IllegalArgumentException("Scan helper must retain one scalar ordinary unit");
                helper = unit.portablePlan().specialization();
                var scratch = analysis.workspaceDeclaration();
                scratchAlignment = scratch.map(w -> Math.toIntExact(w.byteAlignment())).orElse(1);
                var size = scratch.map(PreparationResourceRequirement.Workspace::size);
                scratchBytes = size.isEmpty()
                        ? new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(0, 0, 1)
                        : size.orElseThrow() instanceof io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize affine
                            ? affine : new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                                    size.orElseThrow().fixedByteSize().orElseThrow(), 0, 1);
                iterations = new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                        unit.elementCount(), 0, 1);
                resets = io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable.temporalResets(ordinary);
            }
            CpuPartitionLowering.requirePermittedLowPrecisionUse(helper, ordinary.kernelIr());
            alignment = Math.max(alignment, scratchAlignment);
            cursor = alignCompositionBytes(cursor, Math.max(Long.BYTES, scratchAlignment));
            var scratchOffset = cursor;
            cursor = addCompositionBytes(cursor, scratchBytes);
            units.add(new CpuKernelSpecialization.CompositionUnit(helper, resets, scratchOffset, iterations));
        }
        cursor = alignCompositionBytes(cursor, alignment);
        io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize total = scan.ir().symbolic()
                ? cursor : new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(cursor.constantBytes());
        var geometry = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Geometry(
                scan.geometry().time(), scan.geometry().fixedElements(), total);
        return new CompositionResources(
                new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts(
                        scan.ir(), geometry, scan.spans(), scan.accessBindings(), scan.batchBindings(), scan.ordinaryUnits()),
                List.copyOf(units), alignment);
    }

    /**
     * Selects the actual numeric control-row recipe without inventing ROW provenance.
     * @param composition exact typed body facts
     * @return proved length/ROW extent coefficients, or zero for SHARED-only bodies
     */
    private static io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize
            compositionBatchSize(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Composition composition) {
        long constant = 0, factor = 0;
        if (composition.lengthAware()) {
            var lengths = composition.boundaryMappings().get(composition.lengthRead());
            constant = lengths.extentConstants().getFirst();
            factor = lengths.extentBatchFactors().getFirst();
        } else for (var value : composition.values()) if (value.rowAxis() >= 0) {
            constant = value.access().extentConstants().get(value.rowAxis());
            factor = value.access().extentBatchFactors().get(value.rowAxis());
            break;
        }
        return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(constant, factor, 1);
    }

    /**
     * Adds disjoint byte recipes without resolving any run batch.
     * @param left non-negative preceding byte recipe
     * @param right non-negative appended byte recipe
     * @return checked affine sum, expressed in individual bytes
     */
    private static io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize
            addCompositionBytes(
                    io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize left,
                    io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize right) {
        return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                Math.addExact(left.constantBytes(), right.constantBytes()),
                Math.addExact(left.bytesPerBatch(), right.bytesPerBatch()), 1);
    }

    /**
     * Aligns both coefficients so every actual B preserves helper alignment.
     * @param bytes exact non-negative affine byte origin
     * @param alignment positive power-of-two required byte alignment
     * @return aligned origin, retaining all preceding storage
     * @throws ArithmeticException if either rounded coefficient overflows
     */
    private static io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize
            alignCompositionBytes(
                    io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize bytes,
                    int alignment) {
        return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                Math.multiplyExact(ceilDiv(bytes.constantBytes(), alignment), alignment),
                Math.multiplyExact(ceilDiv(bytes.bytesPerBatch(), alignment), alignment), alignment);
    }

    /**
     * Derives only selected code/resource shape from the already proved region. Worker policy
     * remains on the execution unit; no invocation B or length participates. Transpose lifetime
     * is invariant only for an immutable invariant formal, otherwise exactly one active step.
     * Scalar staging also authenticates the shared query's source scopes and both phase bases
     * before the inline/no-exec-v1 exit. Each extra demanded row wrapper contributes its own
     * carrier and two-slot base; staging failure changes neither strategy nor resource layout.
     * All demanded histories, including forwarded sources, additionally retain their separate
     * LONG caller bases across invocation staging. Unproved multi-stage lifetimes stay checked.
     * @param context cold configuration owner
     * @param region complete CPU0073 proof
     * @param scan Scan binder, or null for an ordinary region
     * @return selected immutable execution layout, empty for inline scalar or failed resource proof;
     *     scalar load/address reuse is independently bounded before cuts and needs no payload
     */
    private static Optional<CpuKernelSpecialization.RowFusionEmission> rowEmission(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr scan) {
        var config = context.backendInputs().portableExecution();
        int ranges = Math.min(config.configuredMaximumParallelism(),config.availableParallelism());
        boolean vector = config.computePreference() == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorRowEligible(region);
        if (!vector) {
            var invariantSources = new ArrayList<Integer>(); var temporalSources = new ArrayList<Integer>();
            if (scan == null) {
                for (var v : region.arithmetic().values()) if (v.kind() == CpuKernelIr.Value.Kind.INPUT
                        && region.viewOrigins().get(v.ordinal()) == v.ordinal()) invariantSources.add(v.ordinal());
            } else for (int f = scan.carryCount(); f < scan.formalOrdinals().size(); f++) {
                int root = scan.formalOrdinals().get(f);
                if (scan.scalarReads().get(scan.actualReads().get(f))) continue;
                if (f >= scan.carryCount()+scan.sequenceCount()) invariantSources.add(root);
                else if (region.rowAxes().get(root) < 0) temporalSources.add(root);
            }
            // Also executes for scalar without exec-v1. No resource/route is conditional on
            // staging success: the same source scopes and phase ceilings drive cache/emission.
            long duplicateBindings = 0;
            if (scan != null) for (int v : region.rowSites()) if (region.retainedStores().contains(v)) {
                long wrappers = 0;
                for (int p : scan.demandedPorts()) if (p < scan.carryCount() ? scan.nextOrdinals().get(p) == v
                        : region.viewOrigins().get(scan.portOrdinals().get(p)) == v) wrappers++;
                duplicateBindings += Math.max(0,wrappers-1);
            }
            if (scan != null) {
                var finish = scan.stages().orElseThrow().order().stream().filter(s -> s.values().containsAll(region.rowSites())
                        && s.values().containsAll(region.terminalOutputs())).findFirst().orElseThrow().epilogue();
                for (int v = scan.formalOrdinals().size(); v < region.viewOrigins().size(); v++) if (region.viewOrigins().get(v) == v
                        && !region.rowSites().contains(v) && !region.terminalOutputs().contains(v) && !finish.contains(v)) {
                    invariantSources.clear(); temporalSources.clear(); break;
                }
            }
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLowering
                    .invariantLoadStaging(region,
                        invariantSources,temporalSources,
                        112+3*duplicateBindings+2*(scan == null ? 0 : scan.demandedPorts().stream().filter(p -> p >= scan.carryCount()).count()),
                        116+3*duplicateBindings);
        }
        if (!vector && ranges < 2) return Optional.empty();
        if (scan != null) {
            var s = scan.stages().orElseThrow();
            if (s.scratchBytes()%8 != 0 || s.scratchBytesPerBatch()%8 != 0) return Optional.empty();
        }
        var invariant = new ArrayList<Integer>(); var temporal = new ArrayList<Integer>();
        if (vector) for (var contraction : region.contractions()) if (region.rowSites().contains(contraction.output())) {
            int v = contraction.right(); var a = region.accesses().get(v);
            if (a.strideConstants().get(1) == 1) continue;
            if (invariant.contains(v) || temporal.contains(v)) continue;
            int origin = region.viewOrigins().get(v);
            int formal = scan == null ? -1 : scan.formalOrdinals().indexOf(origin);
            if (scan == null || formal >= scan.carryCount()+scan.sequenceCount())
                invariant.add(v);
            else temporal.add(v);
        }
        int values = region.arithmetic().values().size();
        // Two carriers (read/write) per ordinal, scratch and lengths. Frame: original time, R,
        // private scratch base and actual N, then read/write bases and two effective byte strides
        // per value, followed by demanded-wrapper bases. No policy enters the frame.
        int operands = scan == null ? 0 : Math.addExact(Math.addExact(Math.multiplyExact(values,2),2),scan.demandedPorts().size());
        int frame = scan == null ? 0 : Math.addExact(Math.addExact(4,Math.multiplyExact(values,4)),scan.demandedPorts().size());
        var result = new CpuKernelSpecialization.RowFusionEmission(Math.max(1,ranges),invariant,temporal,operands,frame);
        long terminals = 0, rowCells = 0, staging = 0;
        for (int v : region.terminalOutputs()) terminals += rowCells(region,v,false);
        for (int v : region.rowSites()) rowCells += rowCells(region,v,true);
        for (var list : List.of(invariant,temporal)) for (int v : list) staging += rowCells(region,v,false);
        int width = region.arithmetic().values().getFirst().dataType().byteWidth();
        // Retain CPU0073's carrier/base/typed-temporary allowance. Scalar range control adds
        // two INT bounds and the two-slot validity flag; the caller merge is disjoint from
        // producer liveness and has fewer floating cells. Do not sum caller and helper locals.
        long bindingSlots = 5+3L*region.arithmetic().values().stream().filter(v -> v.kind()
                != CpuKernelIr.Value.Kind.VIRTUAL).count()+6L*(width/4);
        long localBound = bindingSlots+4+(rowCells+terminals)*(width/4);
        long chunks = 0; int maskCount = 0;
        if (vector) {
            int lanes = speciesBits(region.arithmetic().values().getFirst().dataType())/(width*8);
            var masks = new java.util.HashSet<Integer>();
            for (int v : java.util.stream.Stream.concat(region.rowSites().stream(),region.terminalOutputs().stream()).toList()) {
                var a = region.accesses().get(v); long n = a.plan().iterationRank() == 0 ? 1 : a.extentConstants().getLast();
                if (n > 0) chunks += rowCells(region,v,region.rowAxes().get(v) >= 0)/n*((n+lanes-1)/lanes);
                for (int column = 0; column < n; column += lanes) masks.add((int)Math.min(lanes,n-column));
            }
            // One reference per chunk, at most one typed extraction per producing feature,
            // exactly the used fixed masks, and eight additional staging/range/control slots. Staging
            // raw-copy temporaries and the caller-only scalar merge are not live in this loop.
            localBound = bindingSlots+chunks+rowScalarExtractionCells(region)*(width/4)+masks.size()+8
                    +rowInputCacheSlots(region,lanes)+rowStagedVectorSlots(region,lanes)+rowInvariantVectorSlots(region,lanes);
            maskCount = masks.size();
        }
        if (scan != null) {
            long caller = bindingSlots+terminals*(width/4)+12;
            localBound = ranges > 1 ? Math.max(caller,scanRowHelperSlots(region,scan,vector,rowCells,terminals,chunks,maskCount))
                    : Math.max(caller,localBound);
        }
        // Only scalar row phases use the recycled affine offset and last-use value pool.
        // Caller merge and range helper are disjoint; never add their live sets together.
        // The binders cap their one LONG affine offset and typed last-use pool at exactly
        // this remaining capacity. Unused capacity is not allocated or held live.
        if (!vector) localBound += Math.max(0,128-localBound);
        // Existing CPU0073 method bound plus bounded raw-copy and partial publication guards.
        long bytes = 1024+256L*region.arithmetic().values().stream().filter(v -> v.kind()
                != CpuKernelIr.Value.Kind.VIRTUAL).count()+1024*rowCells+512*terminals+128*staging+2048;
        for (var c : region.contractions()) if (region.rowSites().contains(c.output()))
            bytes += 512*rowCells(region,c.output(),true)*region.accesses().get(c.left()).extentConstants().get(1);
        return localBound <= 128 && bytes <= 48000 ? Optional.of(result) : Optional.empty();
    }

    /**
     * Bounds the range helper's simultaneous liveness separately from its caller's merge.
     * The fixed 24 slots cover arrays, checked layouts, scratch, original time, R/private base,
     * bounds/induction, address, partial offset/validity, lengths and vector species/byte order.
     * Only required source roots, producing destinations, duplicate wrappers and nonconstant
     * structural stride facts survive the preheader. Dead Shaped-owner allocations and caller
     * temporaries do not add simultaneously live slots to this helper. External contraction-left
     * sources retain their carrier/base allowance even when row-invariant: a scalar load is not
     * replaced merely by proving its address invariant.
     * @param region complete selected region
     * @param scan owning temporal binding proof
     * @param vector whether cells are held in vector chunks
     * @param rows fixed producer feature cells, excluding R
     * @param terminals fixed terminal scalar cells
     * @param chunks fixed producer/terminal vector reference count
     * @param masks exact number of used fixed masks
     * @return conservative helper live-slot bound; over-limit rank returns a rejecting bound
     */
    private static long scanRowHelperSlots(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr scan,boolean vector,
            long rows,long terminals,long chunks,int masks) {
        var reads = new java.util.LinkedHashSet<Integer>();
        for (var i : region.arithmetic().instructions()) if (region.rowSites().contains(i.output())) reads.addAll(i.inputs());
        for (var c : region.contractions()) { reads.add(c.left()); reads.add(c.right()); }
        for (var sum : region.sums()) reads.add(sum.input());
        reads.removeIf(v -> region.rowSites().contains(region.viewOrigins().get(v)));
        if (vector) for (var c : region.contractions()) if (region.rowSites().contains(c.output())) {
            var a = region.accesses().get(c.right());
            if (a.strideConstants().getLast() != 1 && a.extentConstants().getLast() > 0
                    && region.contractions().stream().noneMatch(other -> other.left() == c.right())) reads.remove(c.right());
        }
        var roots = new java.util.HashSet<Integer>(); var facts = new java.util.HashSet<List<Long>>();
        var stages = scan.stages().orElseThrow();
        for (int v : reads) {
            int root = region.viewOrigins().get(v), formal = scan.formalOrdinals().indexOf(root); roots.add(root);
            var a = region.accesses().get(v);
            if (a.plan().iterationRank() > 2) return 129;
            for (int axis = 0; axis < a.plan().iterationRank(); axis++) {
                long c = a.strideConstants().get(axis), f = a.strideBatchFactors().get(axis), ic = c, iff = f;
                if (scan.symbolic()) {
                    var initial = stages.views().get(v).initialRead().orElse(stages.views().get(v).batch().orElseThrow());
                    ic = initial.strideConstants().get(axis); iff = initial.strideBatchFactors().get(axis);
                }
                if (formal >= 0 && scan.scalarReads().get(scan.actualReads().get(formal))) { ic = 0; iff = 0; }
                boolean changes = formal >= 0 && formal < scan.carryCount();
                if ((changes && (c != ic || f != iff)) || iff != 0) facts.add(List.of((long)root,c,f,ic,iff));
            }
        }
        long writes = 0, duplicates = 0;
        for (int v : region.rowSites()) {
            var cut = stages.cuts().stream().filter(c -> c.value() == v).findFirst();
            if (cut.isEmpty()) continue;
            writes++;
            for (int p : scan.demandedPorts()) if (cut.orElseThrow().port() != p
                    && (p < scan.carryCount() ? scan.lengthAware() && stages.carryPorts().get(p) == p && scan.nextOrdinals().get(p) == v
                        : (scan.symbolic() || scan.lengthAware()) && scan.portOrdinals().get(p) == v)) duplicates++;
        }
        int typedWidth = scan.dataType().byteWidth()/4;
        int lanes = speciesBits(scan.dataType())/scan.dataType().bitWidth();
        long invariantSlots = vector ? rowInvariantVectorSlots(region,lanes) : 0;
        var loopRoots = new java.util.HashSet<Integer>();
        for (int v : reads) if (!vector || !rowInvariantView(region,v)) loopRoots.add(region.viewOrigins().get(v));
        if (vector) for (var c : region.contractions()) {
            int root = region.viewOrigins().get(c.left());
            if (!region.rowSites().contains(root)) loopRoots.add(root);
        }
        long header = 24+3*roots.size()+3*writes+3*duplicates+2*facts.size()+masks+2+3L*typedWidth
                +(vector ? invariantSlots+rowStagedVectorSlots(region,lanes) : 0);
        long body = 24+3*loopRoots.size()+3*writes+3*duplicates+2*facts.size()+invariantSlots
                +(vector ? chunks+rowScalarExtractionCells(region)*typedWidth+masks+2+rowInputCacheSlots(region,
                    speciesBits(scan.dataType())/scan.dataType().bitWidth())+rowStagedVectorSlots(region,
                    speciesBits(scan.dataType())/scan.dataType().bitWidth()) : (rows+terminals)*typedWidth+3L*typedWidth);
        return Math.max(header,body);
    }

    /**
     * Proves a complete external view's feature reads independent of R. A changing ROW carry's
     * scalar initial does not suffice: its nominal row stride must be structurally zero.
     * @param region complete selected access/origin proof
     * @param v external source view ordinal
     * @return whether all its vector feature requests can be retained before the row loop
     */
    private static boolean rowInvariantView(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int v) {
        int root = region.viewOrigins().get(v); if (region.rowSites().contains(root)) return false;
        var a = region.accesses().get(v); int row = region.rowAxes().get(v);
        return row < 0 || a.strideConstants().get(row) == 0 && a.strideBatchFactors().get(row) == 0;
    }

    /**
     * Bounds dedicated external pointwise/broadcast vector references, excluding transpose
     * vectors already counted separately. A vector-only binding may die before producer/terminal
     * liveness; scalar contraction-left uses retain their separate carrier/base allowance.
     * @param region complete region and fixed-feature recipes
     * @param lanes positive preferred typed lane count
     * @return conservative additional simultaneous reference slots
     */
    private static long rowInvariantVectorSlots(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int lanes) {
        var loaded = new java.util.HashSet<Integer>(); var staged = new java.util.HashSet<Integer>();
        for (var i : region.arithmetic().instructions()) if (region.rowSites().contains(i.output())) loaded.addAll(i.inputs());
        for (var c : region.contractions()) {
            loaded.add(c.right());
            if (region.rowSites().contains(c.output()) && region.accesses().get(c.right()).strideConstants().getLast() != 1) staged.add(c.right());
        }
        for (var sum : region.sums()) loaded.add(sum.input());
        long slots = 0;
        for (int v : loaded) if (!staged.contains(v) && rowInvariantView(region,v)) {
            var a = region.accesses().get(v); int rank = a.plan().iterationRank();
            if (rank == 0) slots++;
            else {
                long n = a.extentConstants().getLast();
                if (n > 0) slots += rowCells(region,v,region.rowAxes().get(v) >= 0)/n*((n+lanes-1)/lanes);
            }
        }
        return slots;
    }

    /**
     * Counts dedicated row-invariant transpose vector locals, deduplicating exact source views.
     * Includes width-one masked chunks. Worker source bindings used only through these staged
     * views are dead before the row loop and are excluded separately above; scalar uses of the
     * same root must retain their binding allowance. Coefficient vectors remain live with
     * producer/terminal vectors.
     * @param region complete selected region and staging proof
     * @param lanes positive preferred typed lane count
     * @return simultaneous coefficient reference slots, excluding raw staging temporaries
     */
    private static long rowStagedVectorSlots(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int lanes) {
        var staged = new java.util.HashSet<Integer>(); long slots = 0;
        for (var c : region.contractions()) if (region.rowSites().contains(c.output())) {
            int v = c.right(); var a = region.accesses().get(v);
            long n = a.extentConstants().getLast();
            if (n > 0 && a.strideConstants().getLast() != 1 && staged.add(v))
                slots += rowCells(region,v,false)/n*((n+lanes-1)/lanes);
        }
        return slots;
    }

    /**
     * Bounds dedicated cached row-input vectors and at most one typed extraction per feature.
     * Scalar-only contraction-left reads do not create vector caches. Roots are deduplicated
     * structurally; no operand temporary may stand in for a retained vector.
     * @param region complete source/view proof
     * @param lanes positive typed preferred lane count
     * @return conservative additional simultaneous JVM slots
     */
    private static long rowInputCacheSlots(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int lanes) {
        var loaded = new java.util.HashSet<Integer>();
        for (var i : region.arithmetic().instructions()) if (region.rowSites().contains(i.output())) loaded.addAll(i.inputs());
        for (var c : region.contractions()) loaded.add(c.right());
        for (var sum : region.sums()) loaded.add(sum.input());
        var cached = new java.util.HashSet<Integer>();
        for (int v : loaded) {
            int root = region.viewOrigins().get(v); var a = region.accesses().get(v); var source = region.accesses().get(root);
            if (!region.rowSites().contains(root) && region.rowAxes().get(v) == 0 && region.rowAxes().get(root) == 0
                    && a.plan().iterationRank() == 2 && source.plan().iterationRank() == 2
                    && a.extentConstants().getLast().equals(source.extentConstants().getLast())
                    && a.extentConstants().getLast() > 1 && a.strideConstants().getLast() == 1 && a.strideBatchFactors().getLast() == 0) cached.add(root);
        }
        long slots = 0; int typedWidth = region.arithmetic().values().getFirst().dataType().byteWidth()/4;
        for (int root : cached) {
            long features = region.accesses().get(root).extentConstants().getLast();
            slots += (features+lanes-1)/lanes+features*typedWidth;
        }
        return slots;
    }

    /**
     * Bounds typed lane extractions only for producer roots actually consumed as scalars.
     * Ordinary vector-to-vector pointwise edges do not allocate scalar feature locals.
     * @param region complete origin and opcode/contraction proof
     * @return conservative fixed producing-feature count, excluding R
     */
    private static long rowScalarExtractionCells(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region) {
        var scalar = new java.util.HashSet<Integer>(); var vector = new java.util.HashSet<Integer>();
        for (var c : region.contractions()) { scalar.add(c.left()); vector.add(c.right()); }
        for (var i : region.arithmetic().instructions()) if (region.rowSites().contains(i.output())) vector.addAll(i.inputs());
        for (var sum : region.sums()) vector.add(sum.input());
        for (int v : vector) {
            var a = region.accesses().get(v);
            if (a.plan().iterationRank() == 0 || a.extentConstants().getLast() == 1 && a.extentBatchFactors().getLast() == 0
                    || a.strideConstants().getLast() == 0 && a.strideBatchFactors().getLast() == 0) scalar.add(v);
        }
        var roots = new java.util.HashSet<Integer>();
        for (int v : scalar) if (region.rowSites().contains(region.viewOrigins().get(v))) roots.add(region.viewOrigins().get(v));
        long cells = 0; for (int root : roots) cells += rowCells(region,root,true);
        return cells;
    }

    /**
     * Proves contiguous feature chunks, scalar broadcasts and bounded right-transpose staging.
     * Feature K remains ordered and represented SUM may reduce only rows or singleton features.
     * @param region existing complete row/origin/alias proof
     * @return whether every selected access and opcode has this vector binding
     */
    private static boolean vectorRowEligible(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region) {
        for (int v : region.rowSites()) {
            var a = region.accesses().get(v);
            if (region.rowAxes().get(v) != 0 || a.plan().iterationRank() != 2
                    || a.strideConstants().get(1) != 1 || a.strideBatchFactors().get(1) != 0) return false;
        }
        for (var i : region.arithmetic().instructions()) if (region.rowSites().contains(i.output())) {
            if (i.opcode().vectorForm() != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.VectorForm.VALUE)
                return false;
            for (int v : i.inputs()) {
                var a = region.accesses().get(v); int rank = a.plan().iterationRank();
                if (rank > 2 || region.rowAxes().get(v) > 0) return false;
                if (rank > 0 && (a.strideBatchFactors().get(rank-1) != 0
                        || a.strideConstants().get(rank-1) != 0 && a.strideConstants().get(rank-1) != 1)) return false;
            }
        }
        for (var c : region.contractions()) if (region.rowSites().contains(c.output())) {
            var a = region.accesses().get(c.right());
            if (a.strideBatchFactors().stream().anyMatch(f -> f != 0)) return false;
            if (a.strideConstants().get(1) != 1 && a.strideConstants().get(0) != 1) return false;
        }
        for (var s : region.sums()) {
            var a = region.accesses().get(s.input());
            if (region.rowAxes().get(s.input()) != 0 || a.plan().iterationRank() != 2) return false;
            for (int axis : s.axes()) if (axis != 0 && a.extentConstants().get(axis) != 1) return false;
        }
        return true;
    }

    /**
     * Counts fixed feature cells without substituting invocation B.
     * @param region proved access inventory
     * @param value value ordinal
     * @param omitRow whether to omit its authenticated row coordinate
     * @return exact bounded cell count
     */
    private static long rowCells(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int value,boolean omitRow) {
        long cells = 1; var a = region.accesses().get(value);
        for (int axis = 0; axis < a.plan().iterationRank(); axis++) if (!omitRow || axis != region.rowAxes().get(value))
            cells = Math.multiplyExact(cells,a.extentConstants().get(axis));
        return cells;
    }

    /**
     * Extends an already declared-body recipe with bounded staging and W private partial slots.
     * Existing cuts/banks and their inclusive lifetimes are unchanged; all additions precede slots.
     * @param facts immutable scalar body/access facts
     * @param emission selected code shape with proved transpose lifetimes
     * @return fresh typed facts with complete exact workspace declaration
     */
    private static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts rowScanResources(
            io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts facts,
            CpuKernelSpecialization.RowFusionEmission emission) {
        var ir = facts.ir(); var s = ir.stages().orElseThrow(); var r = s.rowMapReductionIr().orElseThrow();
        int width = ir.dataType().byteWidth(); long extra = rowWorkspaceBytes(r,emission);
        var stages = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(s.views(),s.order(),s.cuts(),s.carryOffsets(),
                s.carryPorts(),Math.addExact(s.scratchBytes(),extra),s.carryOffsetFactors(),s.scratchBytesPerBatch(),s.symbolic(),
                s.sequenceStrides(),s.rowAxes(),s.terminal(),s.rowCarryCount(),s.lengthRead(),s.batch(),s.rowMapReductionIr());
        var selected = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),ir.actualReads(),
                ir.scalarReads(),ir.nextOrdinals(),ir.portOrdinals(),ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),
                ir.sequenceCount(),ir.reverse(),1,Optional.of(stages));
        var geometry = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Geometry(
                facts.geometry().time(),facts.geometry().fixedElements(),s.symbolic() ? stages.workspaceSize(width)
                    : new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(stages.scratchBytes()));
        return new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts(selected,geometry,
                facts.spans(),facts.accessBindings(),facts.batchBindings());
    }

    /**
     * Sizes bounded transpose and partial/validity areas without any B-by-parameter bank.
     * @param region proved fixed feature inventory
     * @param emission selected maximum partial count and transpose sources
     * @return exact eight-byte aligned private workspace bytes
     * @throws ArithmeticException if resource sizing overflows
     */
    private static long rowWorkspaceBytes(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,
            CpuKernelSpecialization.RowFusionEmission emission) {
        long cells = 0, transposes = 0; int width = region.arithmetic().values().getFirst().dataType().byteWidth();
        for (int v : region.terminalOutputs()) cells = Math.addExact(cells,rowCells(region,v,false));
        for (var list : List.of(emission.invariantTransposeValues(),emission.temporalTransposeValues()))
            for (int v : list) transposes = Math.addExact(transposes,rowCells(region,v,false));
        long stage = Math.multiplyExact((Math.addExact(Math.multiplyExact(transposes,width),7))/8,8);
        long slice = Math.addExact(8,Math.multiplyExact(Math.addExact(Math.multiplyExact(cells,width),7)/8,8));
        return Math.addExact(stage,Math.multiplyExact(emission.maximumPartialCount(),slice));
    }

    /**
     * Combines unit-local computations with a deduplicated partition resource inventory.
     * Original sources retained by a count fold are declared before carrier validation and slot
     * assignment, but have no generated child argument or output role. Each child carrier list
     * is projected by ValueId from the selected partition/reference order.
     * @param context non-null complete immutable CPU projection, borrowed
     * @param topology non-null stable selected unit topology with original node membership
     * @param carrierReferenceValues exact baseline carrier-order IDs, or {@code null} to use
     *     this partition's declaration order including supplemental original sources
     * @return non-null immutable general plan and exact shared resource declarations
     * @throws IllegalArgumentException if carrier cardinality, unit or resource facts disagree
     * @throws ArithmeticException if checked resource geometry overflows
     */
    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeDag(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> topology,
            List<ValueId> carrierReferenceValues) {
        var baseInputs = new CpuPartitionAnalysisInputs(
                context.backendInputs().loweringManifestEnabled(), List.of(),
                context.backendInputs().portableExecution(),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, false);
        var analyses = new ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>>();
        for (int index = 0; index < topology.size(); index++) {
            var unit = topology.get(index);
            analyses.add(analyzeUnit(decomposer.unitContext(context, unit.nodes(), baseInputs),
                    unit.lowering(), 0, false));
        }
        var declarations = retainCountSources(context, topology, mergeDeclarations(analyses));
        List<CpuKernelSpecialization.CarrierAccess> requested =
                context.backendInputs().carrierPattern();
        if (!requested.isEmpty()) {
            if (carrierReferenceValues == null && requested.size() != declarations.size()) {
                throw new IllegalArgumentException(
                        "carrier pattern count must match distinct partition buffers: requested="
                                + requested.size() + ", buffers=" + declarations.size()
                                + ", values=" + declarations.stream()
                                    .map(PreparationResourceRequirement.Buffer::valueId).toList()
                                + ", members=" + topology.stream()
                                    .map(unit -> unit.memberNodeOrdinals() + ":"
                                            + (unit.lowering().recurrentScan().isPresent() ? "RecurrentScan"
                                                : unit.lowering().portableKernelIr().getClass().getSimpleName()))
                                    .toList());
            }
            List<ValueId> reference = carrierReferenceValues == null
                    ? declarations.stream().map(PreparationResourceRequirement.Buffer::valueId)
                            .toList()
                    : carrierReferenceValues;
            if (requested.size() != reference.size()) throw new IllegalArgumentException(
                    "carrier pattern count must match the exact 0008B baseline buffers");
            analyses.clear();
            for (int index = 0; index < topology.size(); index++) {
                var unit = topology.get(index);
                var inputs = candidateInputs(context, unit.lowering().boundaryValues(), reference,
                        false);
                analyses.add(analyzeUnit(decomposer.unitContext(context, unit.nodes(), inputs),
                        unit.lowering(), 0, false));
            }
            declarations = retainCountSources(context, topology, mergeDeclarations(analyses));
        }

        var bindingByValue = new java.util.LinkedHashMap<ValueId, CpuAccessPlan.Binding>();
        var requestedByValue = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        var generatedByValue = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        var units = new ArrayList<CpuPartitionPreparationPlan.ExecutionUnitPlan>();
        var requirements = new ArrayList<PreparationResourceRequirement>();
        requirements.addAll(declarations);
        for (int index = 0; index < analyses.size(); index++) {
            CpuPartitionPreparationPlan selected = analyses.get(index).plan();
            mergeUnitFacts(selected, bindingByValue, requestedByValue, generatedByValue);
            int unitIndex = index;
            Optional<PreparationResourceRequirement.Workspace> workspace =
                    selected.workspaceDeclaration().map(declaration ->
                            new PreparationResourceRequirement.Workspace(unitIndex,
                                    declaration.byteSize(), declaration.byteAlignment()));
            workspace.ifPresent(requirements::add);
            var unit = selected.units().getFirst();
            var runtimeFacts = new CpuPartitionPreparationPlan.UnitRuntimeFacts(
                    selected.affineAddressPairs(), selected.materialization(),
                    workspace, selected.workspaceUse(),
                    selected.movementGeometry(), selected.indexingGeometry(),
                    selected.scatterGeometry(), selected.foldGeometry(),
                    selected.orderingGeometry(), selected.randomGeometry(),
                    selected.scanGeometry(), selected.aggregateGeometry(),
                    selected.argExtremaGeometry(), selected.maskedReductionGeometry(),
                    selected.advancedReductionGeometry(), selected.softmaxGeometry(),
                    selected.trailingNormalizationGeometry(),
                    selected.batchNormInferenceGeometry(), selected.batchNormTrainingGeometry(),
                    selected.units().getFirst().runtimeFacts().recurrentScan(),
                    selected.units().getFirst().runtimeFacts().generalScan());
            CpuPartitionDagDecomposer.Unit topologyUnit = topology.get(index);
            units.add(new CpuPartitionPreparationPlan.ExecutionUnitPlan(unit.portablePlan(),
                    unit.boundaryValues(), unit.accessBindings(), unit.carrierPattern(),
                    unit.generatedCarrierPattern(), unit.extents(), unit.elementCount(),
                    unit.executionStrategy(), unit.selectedRangeCount(),
                    unit.minimumElementsPerWorker(), unit.vectorSpeciesBitSize(),
                    unit.conv2dGeometry(), unit.conv3dGeometry(), unit.matmulGeometry(),
                    unit.pool2dGeometry(), unit.pool3dGeometry(), unit.attentionGeometry(),
                    unit.outputCount(),
                    unit.fusionReason(), topologyUnit.dependencies(),
                    topologyUnit.memberNodeOrdinals(), runtimeFacts));
        }
        // These original explicit sources still require initialized representations. They are
        // partition declarations, never computational inputs to a constant-write executable.
        for (int position = 0; position < declarations.size(); position++) {
            var declaration = declarations.get(position); var id = declaration.valueId();
            if (bindingByValue.containsKey(id)) continue;
            var value = context.values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow();
            var descriptor = value.descriptor(); var layout = descriptor.layout().orElseThrow();
            long[] extents = descriptor.shape().toLongArray(); long count = descriptor.shape().knownElementCount().orElseThrow();
            var access = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.GENERAL_ODOMETER,
                    extents.length, java.util.Collections.nCopies(extents.length, CpuAccessPlan.AxisRole.STRIDED), 0);
            bindingByValue.put(id, CpuAccessPlan.Binding.create(access, extents, layout.storageOffset(),
                    layout.strides(), count, 0, count, layout.referencedElementSpan()));
            var carrier = requested.isEmpty() ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : requested.get(position);
            requestedByValue.put(id, carrier); generatedByValue.put(id, carrier);
        }
        var boundaryValues = declarations.stream()
                .map(PreparationResourceRequirement.Buffer::valueId).toList();
        String manifest = context.backendInputs().loweringManifestEnabled()
                ? "form=GENERAL_PARTITION;units=" + units.size() + ";members="
                    + units.stream().map(CpuPartitionPreparationPlan.ExecutionUnitPlan::memberNodeOrdinals)
                        .toList() : "";
        var combined = new CpuPartitionPreparationPlan(
                units,
                CpuPartitionPreparationPlan.Route.PORTABLE,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                declarations, boundaryValues,
                boundaryValues.stream().map(bindingByValue::get).toList(),
                boundaryValues.stream().map(requestedByValue::get).toList(),
                boundaryValues.stream().map(generatedByValue::get).toList(),
                new long[0], 0, new long[0], 1, 1, 0, manifest,
                Optional.empty(), Optional.empty(), CpuPartitionPreparationPlan.WorkspaceUse.NONE,
                new CpuSpecializationBudget(4, 1, 0, 0), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), List.of(), List.of(), List.of(), List.of(),
                List.of(), List.of(), Optional.empty());
        return new BackendPartitionAnalysis<>(context.partition(), combined, requirements);
    }

    /**
     * Retains mandatory original constant-source initialization after private count elimination.
     * @param context non-null complete immutable original source/constant projection, borrowed
     * @param topology non-null selected units with original node membership, not mutated
     * @param declarations non-null computational boundary declarations in first-use order,
     *     not mutated
     * @return immutable deduplicated declarations in supplied order followed by any missing
     *     original folded sources in unit order; never {@code null}
     * @throws NullPointerException if a required reference is {@code null}
     * @throws ArithmeticException if an original source byte span overflows
     */
    private static List<PreparationResourceRequirement.Buffer> retainCountSources(
            PrepareContext<CpuPartitionAnalysisInputs> context, List<CpuPartitionDagDecomposer.Unit> topology,
            List<PreparationResourceRequirement.Buffer> declarations) {
        var result = new LinkedHashMap<ValueId, PreparationResourceRequirement.Buffer>();
        declarations.forEach(d -> result.put(d.valueId(), d));
        for (var unit : topology) if (unit.lowering().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr) {
            var id = unit.nodes().getFirst().inputs().getFirst();
            var descriptor = context.values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
            result.putIfAbsent(id, new PreparationResourceRequirement.Buffer(id,
                    Math.multiplyExact(descriptor.layout().orElseThrow().referencedElementSpan(), descriptor.dataType().byteWidth()),
                    descriptor.dataType().byteWidth()));
        }
        return List.copyOf(result.values());
    }

    private static List<PreparationResourceRequirement.Buffer> mergeDeclarations(
            List<BackendPartitionAnalysis<CpuPartitionPreparationPlan>> analyses) {
        var result = new java.util.LinkedHashMap<ValueId,
                PreparationResourceRequirement.Buffer>();
        for (var analysis : analyses) for (var declaration : analysis.plan().bufferDeclarations()) {
            var old = result.putIfAbsent(declaration.valueId(), declaration);
            if (old != null && (old.byteSize() != declaration.byteSize()
                    || old.byteAlignment() != declaration.byteAlignment())) {
                throw new IllegalArgumentException("unit buffer declarations disagree");
            }
        }
        return List.copyOf(result.values());
    }

    private static void mergeUnitFacts(CpuPartitionPreparationPlan unit,
            java.util.Map<ValueId, CpuAccessPlan.Binding> bindings,
            java.util.Map<ValueId, CpuKernelSpecialization.CarrierAccess> requested,
            java.util.Map<ValueId, CpuKernelSpecialization.CarrierAccess> generated) {
        for (int i = 0; i < unit.boundaryValues().size(); i++) {
            bindings.putIfAbsent(unit.boundaryValues().get(i), unit.accessBindings().get(i));
            requested.putIfAbsent(unit.boundaryValues().get(i), unit.carrierPattern().get(i));
            generated.putIfAbsent(unit.boundaryValues().get(i),
                    unit.generatedCarrierPattern().get(i));
        }
    }

    private static List<CpuKernelIr.PowerRealization> powerRealizations(CpuKernelIr kernelIr) {
        return kernelIr.instructions().stream()
                .filter(instruction -> instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
    }

    private static DataType vectorLaneType(CpuKernelIr kernelIr) {
        var numeric = kernelIr.values().stream().map(CpuKernelIr.Value::dataType)
                .filter(type -> type != DataType.BOOL).distinct().toList();
        if (numeric.size() > 1) return null;
        if (numeric.isEmpty()) return kernelIr.values().stream()
                .allMatch(value -> value.dataType() == DataType.BOOL) ? DataType.BOOL : null;
        DataType type = numeric.getFirst();
        return type == DataType.FLOAT32 || type == DataType.FLOAT64
                || type == DataType.INT32 || type == DataType.INT64 ? type : null;
    }

    private static boolean vectorTopologyEligible(CpuKernelIr ir, DataType laneType) {
        if (laneType == null) return false;
        if (ir.instructions().stream().anyMatch(instruction ->
                instruction.opcode() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST
                        && ir.values().get(instruction.inputs().getFirst()).dataType()
                                != ir.values().get(instruction.output()).dataType())) return false;
        boolean mixedMasks = laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64
                ? ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL) : false;
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.dataType() == DataType.BOOL && mixedMasks
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && !(value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && value.accessPlan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                if (value.accessPlan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR) return false;
            }
        }
        for (CpuKernelIr.Instruction instruction : ir.instructions()) {
            if (!instruction.opcode().vectorEligible()
                    || instruction.opcode() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization() == CpuKernelIr.PowerRealization.DIRECT) {
                return false;
            }
            switch (instruction.opcode().vectorForm()) {
                case VALUE -> {
                    if (!valueOpcodeEligible(instruction.opcode(), laneType)) return false;
                }
                case MASK_PRODUCER -> {
                    if (!mixedMasks || !isProducedFloatingMask(
                            ir.values().get(instruction.output()))) return false;
                }
                case VALUE_OR_MASK -> {
                    boolean byteValues = laneType == DataType.BOOL;
                    boolean floatingMasks = mixedMasks
                            && instruction.inputs().stream().allMatch(input ->
                                isProducedFloatingMask(ir.values().get(input)))
                            && isProducedFloatingMask(ir.values().get(instruction.output()));
                    if (!byteValues && !floatingMasks) return false;
                }
                case MASK_CONSUMER -> {
                    if (!mixedMasks) return false;
                    CpuKernelIr.Value condition = ir.values().get(instruction.inputs().getFirst());
                    if (!isProducedFloatingMask(condition)
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == CpuAccessPlan.Regime.SCALAR_ALL_ZERO)
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == CpuAccessPlan.Regime.DENSE_LINEAR)) return false;
                }
                case NONE -> { return false; }
            }
        }
        return true;
    }

    private static boolean isProducedFloatingMask(CpuKernelIr.Value value) {
        return value.dataType() == DataType.BOOL
                && (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL
                    || value.kind() == CpuKernelIr.Value.Kind.OUTPUT
                        && value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR);
    }

    private static boolean hasDenseFloatingMaskBoundary(CpuKernelIr ir, DataType laneType) {
        return (laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64)
                && ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR);
    }

    private static boolean valueOpcodeEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode opcode,
            DataType laneType) {
        if (laneType == DataType.BOOL) return opcode
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST;
        if (laneType == DataType.INT32 || laneType == DataType.INT64) return switch (opcode) {
            case ADD, SUB, MUL, MIN, MAX, SCALAR_ADD, SCALAR_SUB, SCALAR_MUL,
                    SCALAR_MIN, SCALAR_MAX, CAST -> true;
            default -> false;
        };
        return laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64;
    }

    private static int speciesLanes(DataType type) {
        if (type == null) return 0;
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case INT32 -> IntVector.SPECIES_PREFERRED.length();
            case INT64 -> LongVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            default -> 0;
        };
    }

    private static int speciesBits(DataType type) {
        if (type == null) return 0;
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> LongVector.SPECIES_PREFERRED.vectorBitSize();
            case BOOL -> ByteVector.SPECIES_PREFERRED.vectorBitSize();
            default -> 0;
        };
    }

    private static CpuAccessPlan.Binding denseBinding(long[] extents, long elementCount) {
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int i = 0; i < extents.length; i++) roles.add(CpuAccessPlan.AxisRole.CONTIGUOUS);
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, extents.length, roles, extents.length);
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, Math.max(1, extents[axis]));
        }
        return CpuAccessPlan.Binding.create(plan, extents, 0, strides, elementCount,
                0, elementCount, elementCount);
    }

    private static CpuKernelIr adjustedIr(CpuKernelIr source, int boundaryIndex,
            CpuAccessPlan plan) {
        var values = new ArrayList<>(source.values());
        CpuKernelIr.Value old = values.get(boundaryIndex);
        values.set(boundaryIndex, new CpuKernelIr.Value(old.ordinal(), old.dataType(), old.kind(), plan));
        return new CpuKernelIr(values, source.instructions(), source.loop(), source.stores());
    }

    private static boolean vectorEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding,
            int lanes) {
        return switch (binding.plan().regime()) {
            case GENERAL_ODOMETER -> false;
            case SCALAR_ALL_ZERO -> binding.plan().accessKind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.READ;
            case DENSE_LINEAR -> true;
            case LAST_AXIS_BIAS, BLOCK_OUTER -> contiguousRun(binding) >= lanes;
        };
    }

    private static boolean vectorLossEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr loss,
            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding> bindings) {
        return loss.kind() == io.github.pho001.synaptik.model.operation.loss.LossKind.MEAN_SQUARED_ERROR
                && loss.reduction()
                    == io.github.pho001.synaptik.model.operation.loss.LossReduction.NONE
                && (loss.predictionType() == DataType.FLOAT32
                    || loss.predictionType() == DataType.FLOAT64)
                && loss.predictionType() == loss.targetType()
                && loss.predictionType() == loss.resultType()
                && loss.boundaryAccesses().stream().allMatch(access -> access.regime()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR)
                && bindings.stream().allMatch(binding -> binding.plan().regime()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR);
    }

    private static boolean convVectorEligible(CpuPartitionLowering.LoweredPartition lowered,
            List<CpuAccessPlan.Binding> bindings, int lanes) {
        if (bindings.stream().anyMatch(binding -> binding.plan().regime()
                != CpuAccessPlan.Regime.DENSE_LINEAR)) {
            return false;
        }
        if (lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr conv) {
            if (conv.epilogue()
                    != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr.Epilogue.NONE
                    || !sameVectorConvType(conv.inputTypes(), conv.resultType())
                    || conv.strideWidth() != 1 || conv.dilationWidth() != 1) {
                return false;
            }
            var boundaries = lowered.conv2dGeometry().orElseThrow().boundaries();
            return hasInteriorWidthBlock(boundaries.getFirst().extents()[3],
                    boundaries.get(1).extents()[3], boundaries.getLast().extents()[3],
                    conv.paddingWidth(), lanes);
        }
        if (lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr conv) {
            if (!sameVectorConvType(conv.inputTypes(), conv.resultType())
                    || conv.strideWidth() != 1 || conv.dilationWidth() != 1) {
                return false;
            }
            var boundaries = lowered.conv3dGeometry().orElseThrow().boundaries();
            return hasInteriorWidthBlock(boundaries.getFirst().extents()[4],
                    boundaries.get(1).extents()[4], boundaries.getLast().extents()[4],
                    conv.paddingWidth(), lanes);
        }
        return false;
    }

    private static boolean sameVectorConvType(List<DataType> inputTypes, DataType resultType) {
        return (resultType == DataType.FLOAT32 || resultType == DataType.FLOAT64)
                && inputTypes.stream().allMatch(type -> type == resultType);
    }

    private static boolean hasInteriorWidthBlock(long inputWidth, long kernelWidth,
            long outputWidth, long paddingWidth, int lanes) {
        try {
            long first = Math.max(0L, paddingWidth);
            long lastByOutput = Math.subtractExact(outputWidth, lanes);
            long lastByInput = Math.addExact(inputWidth, paddingWidth);
            lastByInput = Math.subtractExact(lastByInput, lanes);
            lastByInput = Math.subtractExact(lastByInput, kernelWidth);
            lastByInput = Math.addExact(lastByInput, 1L);
            return first <= Math.min(lastByOutput, lastByInput);
        } catch (ArithmeticException incompatibleGeometry) {
            return false;
        }
    }

    private static long contiguousRun(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding) {
        long result = 1;
        int start = binding.extents().size() - binding.plan().contiguousSuffix();
        for (int axis = start; axis < binding.extents().size(); axis++) {
            result = Math.multiplyExact(result, binding.extents().get(axis));
        }
        return result;
    }

    private static long ceilDiv(long value, long divisor) {
        return value == 0 ? 0 : 1 + (value - 1) / divisor;
    }
}
