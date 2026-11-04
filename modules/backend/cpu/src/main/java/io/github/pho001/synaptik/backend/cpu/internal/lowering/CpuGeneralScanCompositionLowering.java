package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Composes a captured PURE Scan through the existing ordinary physical owners. The exact local
 * graph supplies logical requirements, producer/consumer occurrences and unit contexts; outer
 * constants and coincident outer value identities never enter that scope. This cold owner retains
 * typed layouts and declares simultaneous old/new state and ordinary cuts before slot assignment.
 * Low floating values retain their byte/short carriers in the same physical composition;
 * ordinary numerical helpers own FLOAT32 work and each declared operation-result narrowing.
 * It neither executes a body nor defines a second operation-family inventory.
 */
public final class CpuGeneralScanCompositionLowering {
    /** Creates a stateless ordinary-composition boundary. */
    public CpuGeneralScanCompositionLowering() {}

    /**
     * Independently proves scoped descriptors, roles and ordinary feasibility without storage access.
     * Controlled MATMUL selects its qualified direct-scalar realization through the existing
     * matrix owner before retaining physical kernels, identities or iteration geometry.
     * Absorbed matrix reads transport their original logical operand IDs from that same owner,
     * so ROW control follows effective operand coordinates rather than backing storage axes.
     * @param query exact body-bearing Scan occurrence
     * @return complete immutable physical facts, including an empty legitimate unit inventory
     * @throws IllegalArgumentException if declarations, PURE semantics or ordinary feasibility fail
     * @throws ArithmeticException if represented geometry or byte recipes overflow
     */
    public static CpuGeneralScanLowering.Facts analyze(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.operation().kind() != ScanKind.SCAN
                || !(query.operation().attrs() instanceof ScanAttrs attrs)) throw invalid("not a Scan");
        var body = query.scanBody().orElseThrow(() -> invalid("missing captured body"));
        var graph = body.graph();
        if (query.inputs().size() != attrs.inputCount() || query.outputs().size() != attrs.outputCount()
                || graph.inputs().size() != attrs.inputCount() - (attrs.lengthAware() ? 1 : 0)
                || body.nextCarry().size() != attrs.carryCount()
                || body.emissions().size() != attrs.emissionCount()
                || body.residuals().size() != attrs.residualCount()) throw invalid("header cardinality");
        CpuGeneralScanLowering.validateInventory(body, attrs.carryCount());
        var descriptors = new LinkedHashMap<ValueId, TensorDescriptor>();
        var ordinals = new LinkedHashMap<ValueId, Integer>();
        for (var value : graph.values()) {
            ordinals.put(value.id(), ordinals.size()); descriptors.put(value.id(), value.descriptor());
        }
        Optional<DynamicDimension> batch = validateDeclaredShapes(query, graph);
        boolean symbolic = batch.isPresent();
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        checkBindings(query, attrs, body, descriptors, roles);
        // The shared projection proves logical declarations before completing computed layouts.
        var local = graph.nodes().isEmpty() ? null : bodyContext(query, CpuPartitionAnalysisInputs.DEFAULT);
        var semantics = semantics(query, attrs, body, descriptors);
        validateActivity(query, attrs, body, descriptors, semantics);
        var physicalDescriptors = new LinkedHashMap<>(descriptors);
        if (local != null) local.values().forEach(value ->
                physicalDescriptors.put(value.id(), value.descriptor()));
        var ordinary = new ArrayList<CpuPartitionLowering.LoweredPartition>();
        var memberships = new ArrayList<List<Integer>>();
        var dependencies = new ArrayList<List<Integer>>();
        var effectiveBoundaries = new IdentityHashMap<CpuPartitionLowering.LoweredPartition, List<ValueId>>();
        Map<ValueId, CpuAccessPlan.BatchBinding> layouts;
        var backingSizes = new HashMap<ValueId, io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize>();
        if (graph.nodes().isEmpty()) {
            // PlannedPartition deliberately forbids empty membership; do not weaken that invariant.
            layouts = new LinkedHashMap<>();
            for (var value : graph.values()) layouts.put(value.id(), access(value.descriptor(), CpuAccessPlan.AccessKind.READ));
        } else {
            layouts = new LinkedHashMap<>();
            if (symbolic) {
                layouts.putAll(CpuPartitionLowering.batchLayouts(local));
                backingSizes.putAll(CpuPartitionLowering.batchByteSizes(local));
            } else for (var value : local.values()) layouts.put(value.id(), access(value.descriptor(), CpuAccessPlan.AccessKind.READ));
            var decomposer = new CpuPartitionDagDecomposer(); var lowerer = new CpuPartitionLowering(layouts);
            var nodeOrdinals = new HashMap<NodeId, Integer>();
            for (int n = 0; n < graph.nodes().size(); n++) nodeOrdinals.put(graph.nodes().get(n).id(), n);
            for (var group : compatibleGroups(query, semantics)) {
                var scope = decomposer.unitContext(local, group, CpuPartitionAnalysisInputs.DEFAULT);
                for (var unit : decomposer.decompose(scope, lowerer)) {
                    var context = decomposer.unitContext(local, unit.nodes(), CpuPartitionAnalysisInputs.DEFAULT);
                    var lowered = lowerer.lower(context);
                    if (lowered.batchGeometry().isEmpty() && lowered.matmulIr().isPresent()) {
                        var effective = new ArrayList<ValueId>();
                        lowered = new CpuMatmulLowering().lowerControlled(context,
                                lowered.matmulIr().orElseThrow().epilogue(), effective);
                        effectiveBoundaries.put(lowered, List.copyOf(effective));
                    }
                    ordinary.add(lowered);
                    memberships.add(unit.nodes().stream().map(n -> nodeOrdinals.get(n.id())).toList());
                }
            }
            sortUnits(ordinary, memberships, graph);
        }
        var producerUnits = new HashMap<ValueId, Integer>();
        for (int u = 0; u < ordinary.size(); u++) {
            var lowered = ordinary.get(u);
            var deps = new TreeSet<Integer>();
            for (var id : lowered.boundaryValues()) {
                Integer producer = producerUnits.get(id); if (producer != null) deps.add(producer);
            }
            dependencies.add(List.copyOf(deps));
            for (int n : memberships.get(u)) for (var id : graph.nodes().get(n).outputs()) producerUnits.put(id, u);
        }
        var birth = new int[graph.values().size()]; Arrays.fill(birth, -1);
        var death = new int[graph.values().size()]; Arrays.fill(death, -1);
        for (int u = 0; u < ordinary.size(); u++) {
            for (int n : memberships.get(u)) for (var id : graph.nodes().get(n).outputs()) birth[ordinals.get(id)] = u;
            for (var id : ordinary.get(u).boundaryValues()) death[ordinals.get(id)] = u;
        }
        for (var id : body.nextCarry()) death[ordinals.get(id)] = ordinary.size();
        for (int port : body.demandedOutputIndices()) death[ordinals.get(roles.get(port))] = ordinary.size();
        for (var id : graph.inputs().subList(0, attrs.carryCount())) death[ordinals.get(id)] = ordinary.size();
        var physical = new ArrayList<CpuGeneralScanIr.PhysicalValue>();
        BatchByteSize cursor = bytes(0, 0, 8);
        for (var value : graph.values()) {
            int ordinal = ordinals.get(value.id()); var mapping = layouts.get(value.id());
            var descriptor = physicalDescriptors.get(value.id());
            var size = backingSizes.containsKey(value.id())
                    ? affineSize(backingSizes.get(value.id()), descriptor.dataType().byteWidth())
                    : storageSize(descriptor, mapping);
            cursor = align(cursor, descriptor.dataType().byteWidth());
            var fact = semantics.get(value.id());
            physical.add(new CpuGeneralScanIr.PhysicalValue(ordinal, descriptor, mapping, size, cursor,
                    birth[ordinal], Math.max(birth[ordinal], death[ordinal]), fact.rowAxis().orElse(-1), fact.terminal()));
            cursor = add(cursor, size);
        }
        var temporal = new ArrayList<CpuGeneralScanIr.TemporalCarry>();
        var carries = new ArrayList<CpuGeneralScanIr.Carry>();
        for (int c = 0; c < attrs.carryCount(); c++) {
            int formal = ordinals.get(graph.inputs().get(c)), next = ordinals.get(body.nextCarry().get(c));
            var size = physical.get(formal).bytes(); cursor = align(cursor, size.elementByteWidth());
            var oldOffset = physical.get(formal).offset(); var newOffset = cursor; cursor = add(cursor, size);
            temporal.add(new CpuGeneralScanIr.TemporalCarry(formal, next, oldOffset, newOffset));
            carries.add(new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.SCRATCH, c));
        }
        cursor = align(cursor, 8);
        var units = new ArrayList<CpuGeneralScanIr.PhysicalUnit>();
        for (int u = 0; u < ordinary.size(); u++) {
            var lowered = ordinary.get(u); var members = memberships.get(u);
            var boundaryRows = effectiveBoundaries.getOrDefault(lowered, lowered.boundaryValues()).stream()
                    .map(id -> row(descriptors.get(id), semantics.get(id))).toList();
            var valueRows = new ArrayList<CpuScanControl.Row>();
            if (lowered.portableKernelIr() instanceof CpuKernelIr kernel) {
                int read = 0, write = (int) kernel.values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count(), virtual = 0;
                for (var value : kernel.values()) {
                    ValueId id = switch (value.kind()) {
                        case INPUT -> lowered.boundaryValues().get(read++);
                        case OUTPUT -> lowered.boundaryValues().get(write++);
                        case VIRTUAL -> lowered.virtualValues().get(virtual++);
                    };
                    valueRows.add(normalizedRow(descriptors.get(id), semantics.get(id),
                            descriptors.get(lowered.boundaryValues().getLast()).shape()));
                }
            } else {
                valueRows.addAll(boundaryRows);
                if (lowered.portableKernelIr() instanceof CpuAggregateFusionIr fusion && fusion.producer().isPresent()) {
                    var producer = fusion.producer().orElseThrow();
                    int anchor = 0;
                    while (anchor < members.size() && !(graph.nodes().get(members.get(anchor)).operation().kind()
                            instanceof io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind)) anchor++;
                    var produced = members.subList(0, anchor).stream().map(n -> graph.nodes().get(n).outputs().getFirst()).toList();
                    var inputs = fusion.producerInputs().stream().map(p -> lowered.boundaryValues().get(p)).toList();
                    int input = 0, output = 0;
                    var iteration = Shape.of(fusion.inputExtents().stream().mapToLong(Long::longValue).toArray());
                    for (var value : producer.values()) {
                        var id = value.kind() == CpuKernelIr.Value.Kind.INPUT ? inputs.get(input++) : produced.get(output++);
                        valueRows.add(normalizedRow(descriptors.get(id), semantics.get(id), iteration));
                    }
                }
            }
            int scope = query.scanActivityPlan().map(p -> p.nodeScopes().get(graph.nodes().get(members.getFirst()).id())).orElse(-1);
            var soleNode = members.size() == 1 ? graph.nodes().get(members.getFirst()) : null;
            var merge = query.scanActivityPlan().flatMap(p -> Optional.ofNullable(soleNode == null ? null : p.merges().get(soleNode.id())));
            var accumulator = query.scanActivityPlan().flatMap(p -> p.accumulators().stream().filter(a -> soleNode != null
                    && soleNode.outputs().contains(body.nextCarry().get(a.numericSharedCarryPosition()))).findFirst());
            units.add(new CpuGeneralScanIr.PhysicalUnit(lowered.portableKernelIr(), lowered.kernelIr(),
                    lowered.boundaryValues().stream().map(ordinals::get).toList(), dependencies.get(u), members,
                    new CpuScanControl(boundaryRows, valueRows), scope, merge, accumulator,
                    lowered.batchGeometry().map(CpuPartitionLowering.BatchUnitGeometry::identity).orElse("")));
        }
        var boundaryDescriptors = new ArrayList<>(query.inputs());
        for (int port : body.demandedOutputIndices()) boundaryDescriptors.add(query.outputs().get(port));
        var boundaryMappings = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int p = 0; p < boundaryDescriptors.size(); p++) boundaryMappings.add(access(boundaryDescriptors.get(p),
                p < query.inputs().size() ? CpuAccessPlan.AccessKind.READ : CpuAccessPlan.AccessKind.WRITE));
        var sequenceStrides = new ArrayList<BatchByteSize>();
        for (int s = 0; s < attrs.sequenceCount(); s++) {
            int p = attrs.carryCount() + s; var m = boundaryMappings.get(p); int width = query.inputs().get(p).dataType().byteWidth();
            sequenceStrides.add(bytes(Math.multiplyExact(m.strideConstants().getFirst(), width),
                    Math.multiplyExact(m.strideBatchFactors().getFirst(), width), width));
        }
        var composition = new CpuGeneralScanIr.Composition(physical, units, temporal,
                boundaryDescriptors.stream().map(TensorDescriptor::dataType).toList(), boundaryDescriptors,
                java.util.stream.IntStream.range(0, boundaryDescriptors.size()).mapToObj(p ->
                        storageSize(boundaryDescriptors.get(p), boundaryMappings.get(p))).toList(), boundaryMappings, sequenceStrides,
                cursor, symbolic, attrs.lengthAware(), attrs.lengthAware() ? query.inputs().size() - 1 : -1,
                attrs.lengthAware() ? Optional.of(query.inputs().getLast().shape().dimension(0)) : Optional.empty(),
                query.scanActivityPlan().map(p -> new CpuGeneralScanIr.PhysicalActivity(
                        p.carries(), p.seedFormals(), p.scopes(), p.accumulators())));
        var arithmetic = new CpuKernelIr(List.of(), List.of(), new CpuKernelIr.Loop("start", "end"), List.of());
        var formals = graph.inputs().stream().map(ordinals::get).toList();
        var ir = new CpuGeneralScanIr(arithmetic, formals,
                java.util.stream.IntStream.range(0, formals.size()).boxed().toList(),
                boundaryMappings.subList(0, query.inputs().size()).stream().map(m -> m.plan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO).toList(),
                body.nextCarry().stream().map(ordinals::get).toList(), roles.stream().map(ordinals::get).toList(),
                body.demandedOutputIndices(), carries, boundaryMappings.stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                attrs.carryCount(), attrs.sequenceCount(), attrs.reverse(),
                cursor.constantBytes() == 0 && cursor.bytesPerBatch() == 0 ? 0 : 1, Optional.empty(), Optional.of(composition));
        var geometry = new CpuGeneralScanLowering.Geometry(attrs.timeCount(), symbolic ? OptionalLong.empty() : OptionalLong.of(0), symbolic ? cursor
                : new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(cursor.constantBytes()));
        if (symbolic) return new CpuGeneralScanLowering.Facts(ir, geometry, List.of(), List.of(), boundaryMappings, ordinary);
        var bound = boundaryMappings.stream().map(m -> m.bind(0)).toList();
        return new CpuGeneralScanLowering.Facts(ir, geometry, bound.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                bound, List.of(), ordinary);
    }

    /**
     * Projects the exact nonempty body through ordinary computed logical-layout completion.
     * Scoped Model semantics are checked before each completion. Only produced descriptors are
     * overlaid: caller/formal layouts, the captured graph, node/value identities and activity
     * associations are unchanged. Exact inferred views retain their offsets/strides; unresolved
     * static results that the shared owner cannot prove remain physically unsupported.
     * Both decomposition and helper resource analysis consume this same projection.
     * Constants are local ingress only; captured Scan bodies have no implicit constant ingress.
     * @param query exact containing occurrence and optional scoped activity
     * @param inputs immutable scalar/helper analysis configuration
     * @return exact local DAG with projected computed values and matching logical requirements
     * @throws IllegalArgumentException if declarations, scoped semantics or physical completion
     *     are invalid, or the body has no nodes or no captured graph
     * @throws ArithmeticException if declared geometry or exact layout arithmetic overflows
     */
    public static PrepareContext<CpuPartitionAnalysisInputs> bodyContext(
            OperationCapabilityQuery query, CpuPartitionAnalysisInputs inputs) {
        var body = query.scanBody().orElseThrow(() -> invalid("missing captured body"));
        var graph = body.graph();
        if (graph.nodes().isEmpty()) throw invalid("empty body has no partition or ordinary units");
        var batch = validateDeclaredShapes(query, graph);
        var projected = new LinkedHashMap<ValueId, GraphValue>();
        graph.values().forEach(value -> projected.put(value.id(), value));
        for (var node : graph.nodes()) {
            var operands = node.inputs().stream().map(id -> projected.get(id).descriptor()).toList();
            var outputs = node.outputs().stream().map(id -> projected.get(id).descriptor()).toList();
            var described = ScanBody.describeOrdinary(node.operation(), operands, outputs.size(), batch);
            // Logical type/shape/gradient agreement does not promise a physical result layout.
            for (int port = 0; port < outputs.size(); port++) {
                var expected = described.get(port); var declared = outputs.get(port);
                if (declared.dataType() != expected.dataType() || !declared.shape().equals(expected.shape())
                        || declared.requiresGrad() != expected.requiresGrad())
                    throw invalid("ordinary declared semantics");
            }
            var completed = ScanBody.closeOrdinaryLayouts(node.operation(), operands, outputs);
            for (int port = 0; port < outputs.size(); port++) {
                if (!completed.get(port).equals(outputs.get(port))) {
                    ValueId id = node.outputs().get(port);
                    projected.put(id, new GraphValue(id, completed.get(port)));
                }
            }
        }
        var physicalSymbols = new HashSet<DynamicDimension>();
        for (var value : projected.values()) checkShape(value.descriptor(), physicalSymbols, false);
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                graph.nodes().stream().map(CompiledNode::id).toList());
        var logical = LogicalMemoryPlanning.plan(graph, List.of(partition));
        var protectedValues = activityBarriers(query, graph);
        protectedValues.addAll(body.nextCarry()); protectedValues.addAll(body.emissions());
        var roots = new ArrayList<>(body.nextCarry()); roots.addAll(body.emissions()); roots.addAll(body.residuals());
        for (int port : body.demandedOutputIndices()) protectedValues.add(roots.get(port));
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (var requirement : logical.requirements()) requirements.add(new LogicalMemoryRequirement(
                requirement.valueId(), projected.get(requirement.valueId()).descriptor(), requirement.producerPartition(),
                requirement.consumerPartitions(), protectedValues.contains(requirement.valueId())));
        return new PrepareContext<>(new PartitionDag(partition, graph.nodes()), List.copyOf(projected.values()), requirements,
                Map.of(), Objects.requireNonNull(inputs, "inputs"), Map.of());
    }

    private static List<List<CompiledNode>> compatibleGroups(OperationCapabilityQuery query,
            Map<ValueId, Scan.ValueSemantics> semantics) {
        var groups = new ArrayList<List<CompiledNode>>(); var current = new ArrayList<CompiledNode>();
        for (var node : query.scanBody().orElseThrow().graph().nodes()) {
            var proposed = new ArrayList<>(current); proposed.add(node);
            boolean compatible = compatibleActivity(query, proposed);
            if (query.scanActivityPlan().isPresent() || ((ScanAttrs) query.operation().attrs()).lengthAware()) {
                var outputs = proposed.stream().flatMap(n -> n.outputs().stream()).map(semantics::get).toList();
                var first = outputs.getFirst();
                compatible &= outputs.stream().allMatch(f -> f.rowAxis().equals(first.rowAxis()) && f.terminal() == first.terminal());
            }
            if (!compatible && !current.isEmpty()) { groups.add(List.copyOf(current)); current.clear(); }
            current.add(node);
            if (node.outputs().size() != 1) { groups.add(List.copyOf(current)); current.clear(); }
        }
        if (!current.isEmpty()) groups.add(List.copyOf(current));
        return List.copyOf(groups);
    }

    /**
     * Deduplicates outer bindings without admitting outer IDs into the body inventory.
     * @param context exact containing single-Scan projection
     * @return existing lowering result carrying complete ordinary composite facts
     * @throws IllegalArgumentException if the projection is not one qualified Scan
     */
    public CpuPartitionLowering.LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context) {
        return lower(context, context.values().stream().anyMatch(v -> !v.descriptor().shape().isFullyStatic())
                ? CpuPartitionLowering.batchLayouts(context) : Map.of());
    }

    /**
     * Preserves actual outer affine provenance supplied by the containing partition.
     * @param context exact one-Scan projection
     * @param layouts authenticated containing-partition mappings, never sampled descriptors
     * @return complete composite lowering with deduplicated outer reads
     * @throws IllegalArgumentException if a mapping or positional row declaration disagrees
     */
    public CpuPartitionLowering.LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        if (context.nodes().size() != 1) throw invalid("not one projected Scan");
        var node = context.nodes().getFirst(); var values = new HashMap<ValueId, TensorDescriptor>();
        context.values().forEach(v -> values.put(v.id(), v.descriptor()));
        var facts = analyze(new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(values::get).toList(),
                node.outputs().stream().map(values::get).toList(), node.scanBody(), Optional.ofNullable(context.scanActivityPlans().get(node.id()))));
        var old = facts.ir(); var original = old.composition().orElseThrow();
        var mappings = new ArrayList<>(original.boundaryMappings());
        var sequenceStrides = new ArrayList<BatchByteSize>();
        var boundarySizes = new ArrayList<>(original.boundarySizes());
        if (old.symbolic()) {
            var outerSizes = CpuPartitionLowering.batchByteSizes(context);
            for (int p = 0; p < node.inputs().size(); p++) boundarySizes.set(p,
                    affineSize(outerSizes.get(node.inputs().get(p)), values.get(node.inputs().get(p)).dataType().byteWidth()));
        }
        for (int p = 0; p < node.inputs().size(); p++) {
            var mapped = layouts.get(node.inputs().get(p));
            if (mapped != null) mappings.set(p, mapped);
            if (p >= old.carryCount() && p < old.carryCount() + old.sequenceCount()) {
                var actual = mappings.get(p); int width = values.get(node.inputs().get(p)).dataType().byteWidth();
                sequenceStrides.add(bytes(Math.multiplyExact(actual.strideConstants().getFirst(), width),
                        Math.multiplyExact(actual.strideBatchFactors().getFirst(), width), width));
            }
        }
        var unique = new LinkedHashMap<ValueId, Integer>(); var selected = new ArrayList<Integer>();
        var actuals = new ArrayList<Integer>();
        for (int p = 0; p < node.inputs().size(); p++) {
            var id = node.inputs().get(p);
            if (!unique.containsKey(id)) { unique.put(id, unique.size()); selected.add(p); }
            if (p < old.formalOrdinals().size()) actuals.add(unique.get(id));
            if (original.lengthAware() && p < old.formalOrdinals().size()) {
                int role = original.values().get(old.formalOrdinals().get(p)).rowAxis();
                if (role >= 0 && p >= old.carryCount() && p < old.carryCount() + old.sequenceCount()) role++;
                for (int earlier = 0; earlier < p; earlier++) if (node.inputs().get(earlier).equals(id)) {
                    int previous = original.values().get(old.formalOrdinals().get(earlier)).rowAxis();
                    if (previous >= 0 && earlier >= old.carryCount() && earlier < old.carryCount() + old.sequenceCount()) previous++;
                    if (previous != role) throw invalid("tied actual row roles differ");
                }
            }
            if (original.lengthAware() && p == node.inputs().size() - 1 && node.inputs().subList(0, p).contains(id))
                throw invalid("lengths aliases a numerical formal");
        }
        var boundaries = new ArrayList<>(unique.keySet());
        for (int w = 0; w < old.demandedPorts().size(); w++) {
            var id = node.outputs().get(old.demandedPorts().get(w));
            if (boundaries.contains(id)) throw invalid("aliased outer write");
            boundaries.add(id); selected.add(node.inputs().size() + w);
        }
        var composition = new CpuGeneralScanIr.Composition(original.values(), original.units(), original.carries(),
                select(original.boundaryTypes(), selected), select(original.boundaryDescriptors(), selected), select(boundarySizes, selected),
                select(mappings, selected), sequenceStrides, original.workspaceSize(), original.symbolic(), original.lengthAware(),
                original.lengthAware() ? unique.get(node.inputs().getLast()) : -1, original.batch(), original.activity());
        var ir = new CpuGeneralScanIr(old.arithmetic(), old.formalOrdinals(), actuals,
                selected.subList(0, unique.size()).stream().map(p -> old.scalarReads().get(p)).toList(), old.nextOrdinals(),
                old.portOrdinals(), old.demandedPorts(), old.carries(), select(mappings, selected).stream().map(CpuAccessPlan.BatchBinding::plan).toList(), old.carryCount(),
                old.sequenceCount(), old.reverse(), old.scratchBanks(), Optional.empty(), Optional.of(composition));
        return new CpuPartitionLowering.LoweredPartition(new CpuGeneralScanLowering.Facts(ir, facts.geometry(),
                old.symbolic() ? List.of() : select(mappings, selected).stream().map(m -> m.bind(0).referencedElementSpan()).toList(),
                old.symbolic() ? List.of() : select(mappings, selected).stream().map(m -> m.bind(0)).toList(),
                old.symbolic() ? select(mappings, selected) : List.of(), facts.ordinaryUnits()), boundaries);
    }

    private static <T> List<T> select(List<T> values, List<Integer> selected) {
        return selected.stream().map(values::get).toList();
    }

    /**
     * Validates actual-only coordinate authority and role-aware unresolved local declarations.
     * Only genuine producer results may defer a static layout until shared physical completion.
     * @param query containing occurrence whose actual inputs declare the possible batch symbol
     * @param graph exact captured local graph, retained without mutation
     * @return the sole actual-declared coordinate, or empty for a static occurrence
     * @throws IllegalArgumentException if symbols, ingress layouts or physical shape bounds fail
     * @throws ArithmeticException if fixed geometry overflows
     */
    private static Optional<DynamicDimension> validateDeclaredShapes(
            OperationCapabilityQuery query, CompiledGraphModel graph) {
        var symbols = new HashSet<DynamicDimension>();
        query.inputs().forEach(descriptor -> checkShape(descriptor, symbols, false));
        if (symbols.size() > 1) throw invalid("independent batch coordinates");
        var produced = new HashSet<ValueId>();
        graph.nodes().forEach(node -> produced.addAll(node.outputs()));
        var localSymbols = new HashSet<DynamicDimension>();
        // Ordinary ranks apply to step values; checkBindings proves the extra outer time axis.
        for (var value : graph.values()) {
            if (value.descriptor().shape().rank() > 16)
                throw invalid("step rank exceeds ordinary physical envelope");
            checkShape(value.descriptor(), localSymbols, produced.contains(value.id()));
        }
        query.outputs().forEach(descriptor -> checkShape(descriptor, localSymbols, false));
        if (!symbols.containsAll(localSymbols)) throw invalid("undeclared batch coordinate");
        return symbols.isEmpty() ? Optional.empty() : Optional.of(symbols.iterator().next());
    }

    /**
     * Checks an original declaration or a completed physical descriptor without binding B.
     * @param descriptor exact represented descriptor
     * @param symbols collection receiving declared coordinates
     * @param computed whether this original node-produced value may defer its static layout
     * @throws IllegalArgumentException if shape, layout or unresolved-ingress rules fail
     * @throws ArithmeticException if fixed geometry exceeds the supported integer domain
     */
    private static void checkShape(TensorDescriptor descriptor, Set<DynamicDimension> symbols, boolean computed) {
        long features = 1; int named = 0;
        for (var dimension : descriptor.shape().dimensions()) {
            if (dimension instanceof DynamicDimension dynamic) { symbols.add(dynamic); named++; }
            else features = Math.multiplyExact(features, dimension.staticSize().orElseThrow(() -> invalid("unknown dimension")));
        }
        if (named > 1 || named > 0 && descriptor.layout().isPresent()) throw invalid("unproved named layout");
        Math.multiplyExact(features, descriptor.dataType().byteWidth());
        if (named == 0 && descriptor.layout().isEmpty() && !computed) throw invalid("missing fixed layout");
    }

    private static void checkBindings(OperationCapabilityQuery query, ScanAttrs attrs, CompiledScanBody body,
            Map<ValueId, TensorDescriptor> local, List<ValueId> roots) {
        boolean gradient = query.inputs().stream().anyMatch(TensorDescriptor::requiresGrad);
        for (int p = 0; p < body.graph().inputs().size(); p++) {
            var formal = local.get(body.graph().inputs().get(p)); var actual = query.inputs().get(p);
            boolean sequence = p >= attrs.carryCount() && p < attrs.carryCount() + attrs.sequenceCount();
            var expected = sequence ? stacked(formal.shape(), attrs.timeCount()) : formal.shape();
            if (formal.dataType() != actual.dataType() || !expected.equals(actual.shape())
                    || formal.requiresGrad() != actual.requiresGrad() || !canonical(formal)) throw invalid("actual/formal mismatch");
        }
        for (int c = 0; c < attrs.carryCount(); c++) {
            var formal = local.get(body.graph().inputs().get(c)); var next = local.get(body.nextCarry().get(c));
            if (formal.dataType() != next.dataType() || !formal.shape().equals(next.shape())) throw invalid("next-state descriptor mismatch");
        }
        for (int p = 0; p < roots.size(); p++) {
            var root = local.get(roots.get(p)); var outer = query.outputs().get(p);
            Shape expected = p < attrs.carryCount() ? root.shape() : stacked(root.shape(), attrs.timeCount());
            if (outer.dataType() != root.dataType() || !outer.shape().equals(expected) || !canonical(outer)
                    || outer.requiresGrad() != (outer.dataType().isDifferentiable() && gradient)) throw invalid("outer role descriptor mismatch");
        }
        if (attrs.lengthAware()) {
            var lengths = query.inputs().getLast();
            if (lengths.dataType() != DataType.INT64 || lengths.requiresGrad() || lengths.shape().rank() != 1)
                throw invalid("lengths must be separate INT64[B]");
        }
    }

    private static Map<ValueId, Scan.ValueSemantics> semantics(OperationCapabilityQuery query, ScanAttrs attrs,
            CompiledScanBody body, Map<ValueId, TensorDescriptor> local) {
        var result = new LinkedHashMap<ValueId, Scan.ValueSemantics>();
        if (!attrs.lengthAware()) {
            for (var value : body.graph().values()) result.put(value.id(), new Scan.ValueSemantics(OptionalInt.empty(), false, List.of()));
            return result;
        }
        Dimension batch = query.inputs().getLast().shape().dimension(0);
        var sharedNext = new HashMap<ValueId, ValueId>(); var sharedDependent = new HashSet<ValueId>();
        for (int f = 0; f < body.graph().inputs().size(); f++) {
            int axis = f < attrs.carryCount() ? f < attrs.rowCarryCount() ? 0 : -1
                    : f < attrs.carryCount() + attrs.sequenceCount() ? attrs.sequenceRowAxes().get(f - attrs.carryCount()).orElse(-1) : -1;
            var id = body.graph().inputs().get(f);
            result.put(id, Scan.declaredSemantics(local.get(id), axis < 0 ? OptionalInt.empty() : OptionalInt.of(axis), batch));
            if (f >= attrs.rowCarryCount() && f < attrs.carryCount()) { sharedNext.put(id, body.nextCarry().get(f)); sharedDependent.add(id); }
        }
        var producers = new HashMap<ValueId, CompiledNode>();
        for (var node : body.graph().nodes()) {
            var facts = Scan.ordinarySemantics(node.operation(), node.inputs().stream().map(local::get).toList(),
                    node.inputs().stream().map(result::get).toList(), node.outputs().stream().map(local::get).toList(), batch);
            for (int o = 0; o < node.outputs().size(); o++) {
                var id = node.outputs().get(o); var fact = facts.get(o); result.put(id, fact); producers.put(id, node);
                for (int input : fact.directInputs()) if (sharedDependent.contains(node.inputs().get(input))) {
                    if (node.operation().kind() != BinaryArithmeticKind.ADD || input != 0 || node.outputs().size() != 1
                            || !id.equals(sharedNext.get(node.inputs().get(input)))) throw invalid("shared-state feedback");
                    sharedDependent.add(id);
                }
            }
        }
        var accumulations = new HashSet<NodeId>();
        for (int c = 0; c < attrs.carryCount(); c++) {
            var root = body.nextCarry().get(c); var fact = result.get(root);
            if (c < attrs.rowCarryCount()) {
                if (fact.rowAxis().orElse(-1) != 0 || fact.terminal()) throw invalid("ROW next-state provenance");
            } else {
                var producer = producers.get(root);
                if (producer == null || producer.operation().kind() != BinaryArithmeticKind.ADD || producer.inputs().size() != 2
                        || !producer.inputs().getFirst().equals(body.graph().inputs().get(c))
                        || sharedDependent.contains(producer.inputs().get(1)) || fact.rowAxis().isPresent()
                        || result.get(producer.inputs().get(1)).rowAxis().isPresent()
                        || !local.get(root).shape().equals(local.get(producer.inputs().get(1)).shape())
                        || !accumulations.add(producer.id())) throw invalid("ordered shared accumulation");
            }
        }
        for (var id : body.emissions()) if (result.get(id).rowAxis().orElse(-1) != 0 || result.get(id).terminal())
            throw invalid("ROW emission provenance");
        return result;
    }

    private static void validateActivity(OperationCapabilityQuery query, ScanAttrs attrs, CompiledScanBody body,
            Map<ValueId, TensorDescriptor> local, Map<ValueId, Scan.ValueSemantics> semantics) {
        if (query.scanActivityPlan().isEmpty()) return;
        var plan = query.scanActivityPlan().orElseThrow();
        if (plan.graph() != body.graph()) throw invalid("activity graph identity");
        var carryPositions = new HashSet<Integer>(); var seedPositions = new HashSet<Integer>();
        for (var carry : plan.carries()) if (carry.numericCarryPosition() < 0
                || (attrs.lengthAware() && carry.numericCarryPosition() >= attrs.rowCarryCount())
                || carry.numericCarryPosition() >= attrs.carryCount()
                || !carryPositions.add(carry.numericCarryPosition())) throw invalid("activity carry role");
        for (var carry : plan.carries()) if (carry.nextScope() < 0 || carry.nextScope() >= plan.scopes().size()
                || !local.get(body.graph().inputs().get(carry.numericCarryPosition())).dataType().isDifferentiable())
            throw invalid("activity cotangent carry descriptor/scope");
        for (int seed : plan.seedFormals()) if (seed < attrs.carryCount() || seed >= attrs.carryCount() + attrs.sequenceCount()
                || !seedPositions.add(seed)) throw invalid("activity seed role");
        for (int seed : plan.seedFormals()) if (!local.get(body.graph().inputs().get(seed)).dataType().isDifferentiable())
            throw invalid("activity seed descriptor");
        var byOutput = new HashMap<ValueId, CompiledNode>();
        body.graph().nodes().forEach(n -> n.outputs().forEach(id -> byOutput.put(id, n)));
        int roots = plan.carries().size() + plan.seedFormals().size();
        for (var scope : plan.scopes()) {
            int previous = -1;
            for (int root : scope) {
                if (root <= previous || root >= roots) throw invalid("activity root order");
                previous = root;
            }
        }
        var expectedNodes = body.graph().nodes().stream().map(CompiledNode::id).toList();
        if (!new ArrayList<>(plan.nodeScopes().keySet()).equals(expectedNodes)) throw invalid("activity occurrence order");
        for (int scope : plan.nodeScopes().values()) if (scope < 0 || scope >= plan.scopes().size())
            throw invalid("activity owner scope");
        for (var entry : plan.merges().entrySet()) {
            var node = body.graph().nodes().stream().filter(n -> n.id().equals(entry.getKey())).findFirst()
                    .orElseThrow(() -> invalid("activity merge membership"));
            if (node.operation().kind() != BinaryArithmeticKind.ADD || node.inputs().size() != 2 || node.outputs().size() != 1)
                throw invalid("activity optional ADD");
            if (entry.getValue().leftScope() < 0 || entry.getValue().leftScope() >= plan.scopes().size()
                    || entry.getValue().rightScope() < 0 || entry.getValue().rightScope() >= plan.scopes().size())
                throw invalid("activity merge operand scope");
            var output = local.get(node.outputs().getFirst());
            for (var id : node.inputs()) if (local.get(id).dataType() != output.dataType()
                    || !local.get(id).shape().equals(output.shape())) throw invalid("optional merge raw-copy descriptor");
            var union = new TreeSet<>(plan.scopes().get(entry.getValue().leftScope()));
            union.addAll(plan.scopes().get(entry.getValue().rightScope()));
            if (!new ArrayList<>(union).equals(plan.scopes().get(plan.nodeScopes().get(node.id()))))
                throw invalid("activity merge union");
        }
        var accumulatorPositions = new HashSet<Integer>();
        for (var accumulator : plan.accumulators()) {
            int c = accumulator.numericSharedCarryPosition();
            if (c < 0 || c >= attrs.carryCount() || attrs.lengthAware() && c < attrs.rowCarryCount()
                    || carryPositions.contains(c) || !accumulatorPositions.add(c)) throw invalid("activity accumulator role");
            if (accumulator.contributionScope() < 0 || accumulator.contributionScope() >= plan.scopes().size()
                    || !local.get(body.graph().inputs().get(c)).dataType().isDifferentiable())
                throw invalid("activity accumulator descriptor/scope");
            var node = byOutput.get(body.nextCarry().get(c));
            if (node == null || node.operation().kind() != BinaryArithmeticKind.ADD || node.inputs().size() != 2
                    || !node.inputs().getFirst().equals(body.graph().inputs().get(c))
                    || !local.get(node.inputs().get(1)).shape().equals(local.get(body.nextCarry().get(c)).shape())
                    || semantics.get(node.inputs().get(1)).rowAxis().isPresent()) throw invalid("activity temporal ADD");
            if (plan.merges().containsKey(node.id())) throw invalid("temporal ADD cannot be an optional merge");
        }
        if (carryPositions.size() + accumulatorPositions.size() != attrs.carryCount())
            throw invalid("activity carry declaration coverage");
        // Independently re-prove root-to-producer implication and temporal result bindings.
        var valueScopes = new HashMap<ValueId, List<Integer>>();
        for (int r = 0; r < plan.carries().size(); r++) valueScopes.put(body.graph().inputs().get(
                plan.carries().get(r).numericCarryPosition()), List.of(r));
        for (int s = 0; s < plan.seedFormals().size(); s++) valueScopes.put(body.graph().inputs().get(
                plan.seedFormals().get(s)), List.of(plan.carries().size() + s));
        for (var node : body.graph().nodes()) {
            var owner = plan.scopes().get(plan.nodeScopes().get(node.id()));
            var merge = plan.merges().get(node.id());
            if (merge != null) {
                var left = valueScopes.get(node.inputs().get(0)); var right = valueScopes.get(node.inputs().get(1));
                if (left == null || right == null || !left.equals(plan.scopes().get(merge.leftScope()))
                        || !right.equals(plan.scopes().get(merge.rightScope()))) throw invalid("activity merge producer binding");
            } else for (var input : node.inputs()) {
                var producer = valueScopes.get(input);
                if (producer != null && !producer.containsAll(owner)) throw invalid("activity producer implication");
            }
            for (var output : node.outputs()) valueScopes.put(output, owner);
        }
        for (var carry : plan.carries()) if (!Objects.equals(
                valueScopes.get(body.nextCarry().get(carry.numericCarryPosition())), plan.scopes().get(carry.nextScope())))
            throw invalid("activity next-state scope binding");
        for (var accumulator : plan.accumulators()) {
            var node = byOutput.get(body.nextCarry().get(accumulator.numericSharedCarryPosition()));
            if (!Objects.equals(valueScopes.get(node.inputs().get(1)), plan.scopes().get(accumulator.contributionScope()))
                    || !Objects.equals(plan.nodeScopes().get(node.id()), accumulator.contributionScope()))
                throw invalid("activity contribution-completion scope binding");
        }
    }

    private static Set<ValueId> activityBarriers(OperationCapabilityQuery query, CompiledGraphModel graph) {
        var result = new HashSet<ValueId>();
        if (query.scanActivityPlan().isEmpty()) return result;
        var plan = query.scanActivityPlan().orElseThrow(); var producerScope = new HashMap<ValueId, Integer>();
        for (var node : graph.nodes()) {
            int scope = plan.nodeScopes().get(node.id());
            for (var id : node.inputs()) if (producerScope.containsKey(id) && producerScope.get(id) != scope) result.add(id);
            boolean special = plan.merges().containsKey(node.id()) || plan.accumulators().stream().anyMatch(a ->
                    node.outputs().contains(query.scanBody().orElseThrow().nextCarry().get(a.numericSharedCarryPosition())));
            if (special) { result.addAll(node.inputs()); result.addAll(node.outputs()); }
            for (var id : node.outputs()) producerScope.put(id, scope);
        }
        return result;
    }

    private static boolean compatibleActivity(OperationCapabilityQuery query, List<CompiledNode> nodes) {
        if (query.scanActivityPlan().isEmpty() || nodes.size() == 1) return true;
        var plan = query.scanActivityPlan().orElseThrow(); int scope = plan.nodeScopes().get(nodes.getFirst().id());
        for (var node : nodes) if (plan.nodeScopes().get(node.id()) != scope || plan.merges().containsKey(node.id())
                || plan.accumulators().stream().anyMatch(a -> node.outputs().contains(
                        query.scanBody().orElseThrow().nextCarry().get(a.numericSharedCarryPosition())))) return false;
        return true;
    }

    private static void sortUnits(List<CpuPartitionLowering.LoweredPartition> units, List<List<Integer>> members,
            CompiledGraphModel graph) {
        var pending = new ArrayList<Integer>(); for (int u = 0; u < units.size(); u++) pending.add(u);
        var producer = new HashMap<ValueId, Integer>();
        for (int u = 0; u < units.size(); u++) for (int n : members.get(u)) for (var id : graph.nodes().get(n).outputs()) producer.put(id, u);
        var done = new HashSet<Integer>(); var ordered = new ArrayList<CpuPartitionLowering.LoweredPartition>();
        var orderedMembers = new ArrayList<List<Integer>>();
        while (!pending.isEmpty()) {
            Integer ready = null;
            for (int u : pending) {
                boolean available = true;
                for (int n : members.get(u)) for (var id : graph.nodes().get(n).inputs()) {
                    Integer p = producer.get(id); if (p != null && p != u && !done.contains(p)) available = false;
                }
                if (available && (ready == null || members.get(u).getFirst() < members.get(ready).getFirst())) ready = u;
            }
            if (ready == null) throw invalid("ordinary unit cycle");
            ordered.add(units.get(ready)); orderedMembers.add(members.get(ready)); done.add(ready); pending.remove(ready);
        }
        units.clear(); units.addAll(ordered); members.clear(); members.addAll(orderedMembers);
    }

    private static CpuScanControl.Row row(TensorDescriptor descriptor, Scan.ValueSemantics semantics) {
        if (semantics.rowAxis().isEmpty()) return CpuScanControl.Row.shared();
        int axis = semantics.rowAxis().orElseThrow(); long stride = 1;
        for (int i = descriptor.shape().rank() - 1; i > axis; i--)
            stride = Math.multiplyExact(stride, descriptor.shape().dimension(i).staticSize().orElseThrow());
        return new CpuScanControl.Row(axis, stride);
    }

    private static CpuAccessPlan.BatchBinding access(TensorDescriptor descriptor, CpuAccessPlan.AccessKind kind) {
        if (!descriptor.shape().isFullyStatic()) return CpuPartitionLowering.batchCanonical(descriptor, kind);
        var layout = descriptor.layout().orElseThrow();
        var extents = Arrays.stream(descriptor.shape().toLongArray()).boxed().toList();
        var strides = Arrays.stream(layout.strides()).boxed().toList();
        var base = CpuPartitionLowering.batchAccess(kind, extents, Collections.nCopies(extents.size(), 0L), strides,
                Collections.nCopies(extents.size(), 0L));
        return new CpuAccessPlan.BatchBinding(base.plan(), base.extentConstants(), base.extentBatchFactors(),
                layout.storageOffset(), base.strideConstants(), base.strideBatchFactors());
    }

    private static BatchByteSize storageSize(TensorDescriptor descriptor, CpuAccessPlan.BatchBinding access) {
        int width = descriptor.dataType().byteWidth();
        if (descriptor.shape().isFullyStatic()) return bytes(Math.multiplyExact(descriptor.layout().orElseThrow().referencedElementSpan(), width), 0, width);
        long constant = 1, factor = 0;
        for (int a = access.plan().iterationRank() - 1; a >= 0; a--) {
            long c = access.extentConstants().get(a), f = access.extentBatchFactors().get(a);
            if (factor != 0 && f != 0) throw invalid("nonaffine backing");
            factor = Math.addExact(Math.multiplyExact(factor, c), Math.multiplyExact(constant, f));
            constant = Math.multiplyExact(constant, c);
        }
        return bytes(Math.multiplyExact(constant, width), Math.multiplyExact(factor, width), width);
    }

    private static CpuScanControl.Row normalizedRow(TensorDescriptor descriptor, Scan.ValueSemantics semantics,
            Shape iterationShape) {
        if (semantics.rowAxis().isEmpty()) return CpuScanControl.Row.shared();
        int axis = semantics.rowAxis().orElseThrow() + iterationShape.rank() - descriptor.shape().rank();
        if (axis < 0 || axis >= iterationShape.rank()) throw invalid("unproved normalized ROW axis");
        long stride = 1;
        for (int a = iterationShape.rank() - 1; a > axis; a--)
            stride = Math.multiplyExact(stride, iterationShape.dimension(a).staticSize().orElseThrow());
        return new CpuScanControl.Row(axis, stride);
    }

    private static boolean canonical(TensorDescriptor descriptor) {
        return descriptor.shape().isFullyStatic() ? descriptor.layout().equals(Optional.of(LayoutDescriptor.contiguous(descriptor.shape())))
                : descriptor.layout().isEmpty();
    }

    private static Shape stacked(Shape shape, long time) {
        var dimensions = new ArrayList<Dimension>(); dimensions.add(new StaticDimension(time)); dimensions.addAll(shape.dimensions());
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    private static BatchByteSize affineSize(
            io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize size, int width) {
        if (size instanceof BatchByteSize affine) return bytes(affine.constantBytes(), affine.bytesPerBatch(), width);
        return bytes(size.fixedByteSize().orElseThrow(() -> invalid("non-affine physical backing")), 0, width);
    }

    private static BatchByteSize bytes(long constant, long factor, long alignment) { return new BatchByteSize(constant, factor, Math.toIntExact(alignment)); }
    private static BatchByteSize add(BatchByteSize left, BatchByteSize right) {
        return bytes(Math.addExact(left.constantBytes(), right.constantBytes()), Math.addExact(left.bytesPerBatch(), right.bytesPerBatch()), 1);
    }
    private static BatchByteSize align(BatchByteSize size, long alignment) {
        return bytes(round(size.constantBytes(), alignment), round(size.bytesPerBatch(), alignment), alignment);
    }
    private static long round(long n, long alignment) { return Math.multiplyExact(Math.floorDiv(Math.addExact(n, alignment - 1), alignment), alignment); }
    private static IllegalArgumentException invalid(String reason) { return new IllegalArgumentException("unsupported ordinary Scan composition: " + reason); }
}
