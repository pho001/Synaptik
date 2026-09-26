package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Analyzes and lowers one complete maximal Metal-owned profile-qualified partition.
 *
 * <p>The deterministic analysis assigns stable native value indices, retains every node kind and
 * ordered operand, and derives unique feeds and targets before selecting a closed private route.
 * For both profiles, it walks explicit unavailable/canonical/affine-view states in node order for
 * the retained six exact unary operations, affine, CONTIGUOUS, UNFOLD_AXIS, GATHER, ONE_HOT,
 * replacement SCATTER_ELEMENTS, and exact classification/BOOL-logic/WHERE domain. Under
 * {@code ACCELERATOR}, it additionally accepts existing arithmetic, reduction, and rank-two
 * MATMUL plus the exact Task-0052 comparisons, tensor/scalar extrema, clamp, reduction extrema,
 * cumulative scans, no-gradient scalar ADD/SUB/MUL/DIV, and no-gradient RECIPROCAL. An affine
 * MATMUL operand is authenticated to the exact earlier local rank-two {@code PERMUTE [1,0]} on
 * that consuming edge. Schema-fourteen lowering emits one bounded self-describing image over the
 * stable type wires 1..6, complete operation registry 1..115, attribute registry 0..41, and
 * explicit prepared route. Production capability is exactly 50 operation kinds; the additional
 * structural recipes remain inaccessible to this analysis.
 * Graph feeds are canonical and explicitly typed. Rank-zero values participate only where the
 * existing non-BOOL operation capability permits them. Exact BOOL results may feed the newly
 * admitted logic and selection nodes or cross owner boundaries.
 * Analysis freshly regenerates the complete candidate batch. Every supplied handoff authenticates
 * its exact partition, schema, workload, profile, and session target; an absent decision preserves
 * the singleton-NEG heuristic, while a present decision must additionally authenticate its
 * candidate identity. Any shared custom-program node fixes the whole partition to its custom
 * program route before exact declarations, including a declared run-owned buffer for every
 * internal logical value. Package-private tests may force only another candidate already approved
 * by that freshly validated batch; production has no corresponding input or switch.
 * Published affine views retain logical descriptors while declarations use full dense represented-
 * order byte geometry. Analysis allocates no physical resource and never changes partition
 * ownership or capability.</p>
 */
final class MetalNegPartitionPreparer implements BackendPartitionPreparer<
        MetalNegAnalysisInputs, MetalNegPreparationPlan> {
    private static final long UINT32_MAX = 0xffff_ffffL;

    /**
     * Produces one immutable whole-partition lowering and its exact shared declarations.
     *
     * @param context non-null complete partition-local facts and borrowed Metal context
     * @return non-null analysis retaining the exact planned partition
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if ownership, operation descriptors, topology, logical
     *     boundaries, constants, or checked geometry are outside the exact route contract
     */
    @Override
    public BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            PrepareContext<MetalNegAnalysisInputs> context) {
        return analyzeInternal(context, null);
    }

    /**
     * Exercises an exact currently approved route without exposing a production selector.
     *
     * <p>The ordinary analysis, capability checks, candidate regeneration, and any supplied
     * handoff authentication complete first. The force succeeds only when the exact route is a
     * member of that freshly generated batch; otherwise analysis fails before native creation.</p>
     *
     * @param context non-null complete partition-local facts and borrowed Metal context
     * @param forcedRoute non-null exact route to require
     * @return non-null analysis retaining the exact planned partition and forced route
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if normal analysis rejects the partition or the route is
     *     not currently approved for it
     */
    BackendPartitionAnalysis<MetalNegPreparationPlan> analyzeForTesting(
            PrepareContext<MetalNegAnalysisInputs> context, MetalPreparedRoute forcedRoute) {
        return analyzeInternal(context, Objects.requireNonNull(forcedRoute, "forcedRoute"));
    }

    private BackendPartitionAnalysis<MetalNegPreparationPlan> analyzeInternal(
            PrepareContext<MetalNegAnalysisInputs> context, MetalPreparedRoute forcedRoute) {
        Objects.requireNonNull(context, "context");
        NumericalProfile numericalProfile = context.numericalProfile();
        if (!context.partition().owner().equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be Metal");
        }
        MetalDeviceContext deviceContext = context.backendInputs().context();

        Map<ValueId, GraphValue> graphValues = new LinkedHashMap<>();
        context.values().forEach(value -> graphValues.put(value.id(), value));
        Map<ValueId, LogicalMemoryRequirement> requirements = new LinkedHashMap<>();
        context.memoryRequirements().forEach(value -> requirements.put(value.valueId(), value));

        var valueIndexes = new LinkedHashMap<ValueId, Integer>();
        var valueIds = new ArrayList<ValueId>();
        var descriptors = new ArrayList<TensorDescriptor>();
        var states = new LinkedHashMap<ValueId, MetalMpsGraphProgram.ValueState>();
        var feeds = new ArrayList<ValueId>();
        var localTranspose = new LinkedHashMap<ValueId, Boolean>();
        int nodeCount = context.nodes().size();
        var programNodes = new ArrayList<MetalMpsGraphProgram.Node>(nodeCount);
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            var node = context.nodes().get(nodeIndex);
            var inputDescriptors = new ArrayList<TensorDescriptor>(node.inputs().size());
            var inputStates = new ArrayList<MetalMpsGraphProgram.ValueState>(node.inputs().size());
            for (ValueId inputId : node.inputs()) {
                GraphValue inputValue = graphValues.get(inputId);
                if (inputValue == null) {
                    throw new IllegalArgumentException(
                            "partition value is missing: " + inputId);
                }
                inputDescriptors.add(inputValue.descriptor());
                MetalMpsGraphProgram.ValueState state = states.get(inputId);
                if (state == null) {
                    if (context.partitionDag().producer(inputId).isPresent()) {
                        throw new IllegalArgumentException(
                                "Metal view producer must precede its consumer: " + inputId);
                    }
                    if (!canonicalFeed(inputValue.descriptor())) {
                        throw new IllegalArgumentException(
                                "Metal graph feeds must be canonical: " + inputId);
                    }
                    state = MetalMpsGraphProgram.ValueState.CANONICAL;
                    states.put(inputId, state);
                    feeds.add(inputId);
                    localTranspose.put(inputId, false);
                }
                inputStates.add(state);
            }
            if (node.outputs().size() != 1) {
                throw new IllegalArgumentException(
                        "Metal supported-operation partition requires one output per node");
            }
            ValueId outputId = node.outputs().getFirst();
            GraphValue outputValue = graphValues.get(outputId);
            if (outputValue == null) {
                throw new IllegalArgumentException(
                        "partition value is missing: " + outputId);
            }
            if (states.containsKey(outputId) || valueIndexes.containsKey(outputId)) {
                throw new IllegalArgumentException(
                        "Metal output must be produced exactly once in topological order");
            }
            if (!MetalCapabilityProvider.supportsOccurrence(
                    numericalProfile,
                    node.operation(),
                    inputDescriptors,
                    List.of(outputValue.descriptor()))) {
                throw new IllegalArgumentException(
                        "Metal occurrence is outside the capability domain");
            }
            int[] inputIndices = new int[node.inputs().size()];
            for (int inputIndex = 0; inputIndex < inputIndices.length; inputIndex++) {
                inputIndices[inputIndex] = index(
                        node.inputs().get(inputIndex),
                        graphValues,
                        valueIndexes,
                        valueIds,
                        descriptors);
            }
            int outputIndex = index(
                    outputId, graphValues, valueIndexes, valueIds, descriptors);
            MetalMpsGraphProgram.Node lowered = lower(
                    node.operation(), inputIndices, outputIndex);
            if (lowered.kind() == MetalMpsGraphProgram.NodeKind.MATMUL) {
                for (int inputIndex = 0; inputIndex < inputStates.size(); inputIndex++) {
                    if (inputStates.get(inputIndex)
                                    == MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                            && !Boolean.TRUE.equals(
                                    localTranspose.get(node.inputs().get(inputIndex)))) {
                        throw new IllegalArgumentException(
                                "Metal MATMUL affine input is not an authenticated local transpose");
                    }
                }
            }
            if (!lowered.kind().acceptsCardinality(inputStates.size(), 1)
                    || inputStates.stream().anyMatch(state -> !lowered.kind().accepts(state))) {
                throw new IllegalArgumentException(
                        "Metal node input value state is unavailable or incompatible");
            }
            programNodes.add(lowered);
            boolean exactLocalTranspose =
                    lowered.kind() == MetalMpsGraphProgram.NodeKind.PERMUTE
                            && inputStates.getFirst()
                                    == MetalMpsGraphProgram.ValueState.CANONICAL
                            && inputDescriptors.getFirst().shape().rank() == 2
                            && outputValue.descriptor().shape().rank() == 2
                            && ((PermutationAttrs) node.operation().attrs())
                                    .axes().equals(List.of(1, 0));
            localTranspose.put(outputId, exactLocalTranspose);
            states.put(outputId, lowered.kind().outputState());
        }
        var graphProgram = new MetalMpsGraphProgram(programNodes);

        validateLogicalPartitionFacts(
                context.partitionDag(), context.partition(), valueIds, requirements);

        var valueStates = new ArrayList<MetalMpsGraphProgram.ValueState>(valueIds.size());
        for (ValueId valueId : valueIds) {
            MetalMpsGraphProgram.ValueState state = states.get(valueId);
            if (state == null || state == MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                throw new IllegalArgumentException("Metal value state is unavailable: " + valueId);
            }
            valueStates.add(state);
        }
        var targets = new ArrayList<ValueId>();
        for (var node : context.partitionDag().nodes()) {
            for (ValueId output : node.outputs()) {
                LogicalMemoryRequirement requirement = require(requirements, output);
                boolean outsideConsumer = requirement.consumerPartitions().stream()
                        .anyMatch(partition -> partition != context.partition());
                if ((requirement.graphOutput() || outsideConsumer) && !targets.contains(output)) {
                    targets.add(output);
                }
            }
        }
        if (targets.isEmpty()) {
            throw new IllegalArgumentException(
                    "Metal supported-operation partition requires at least one target");
        }
        var feedSplats = new ArrayList<Optional<ScalarValue>>(feeds.size());
        boolean[] feedSplatSources = new boolean[feeds.size()];
        for (int index = 0; index < feeds.size(); index++) {
            ValueId feed = feeds.get(index);
            ScalarValue scalar = context.constants().get(feed);
            DataType dataType = graphValues.get(feed).descriptor().dataType();
            if (scalar != null && scalar.dataType() != dataType) {
                throw new IllegalArgumentException(
                        "Metal splat feed must exactly match its descriptor carrier type");
            }
            LogicalMemoryRequirement requirement = require(requirements, feed);
            if (scalar != null && (requirement.producerPartition().isPresent()
                    || requirement.consumerPartitions().isEmpty())) {
                throw new IllegalArgumentException(
                        "Metal splat feed must be a consumed producer-free constant");
            }
            feedSplats.add(Optional.ofNullable(scalar));
            feedSplatSources[index] = scalar != null
                    && requirement.consumerPartitions().getFirst() == context.partition();
        }
        for (ValueId constant : context.constants().keySet()) {
            if (!feeds.contains(constant)) {
                throw new IllegalArgumentException("Metal constant must be a boundary feed");
            }
        }

        int[] feedIndices = indices(feeds, valueIndexes);
        int[] targetIndices = indices(targets, valueIndexes);
        long[] feedBytes = requiredBytes(feeds, graphValues);
        long[] targetBytes = requiredBytes(targets, graphValues);
        boolean containsCustomProgram = graphProgram.nodes().stream()
                .anyMatch(node -> node.kind().isCustomProgramOperation());
        long singletonElements = feedBytes.length == 1 ? feedBytes[0] / Float.BYTES : 0L;
        MetalPreparedRoute route = containsCustomProgram
                ? MetalPreparedRoute.CUSTOM_PROGRAM
                : nodeCount == 1
                        && graphProgram.nodes().getFirst().kind()
                                == MetalMpsGraphProgram.NodeKind.NEG
                        && feeds.size() == 1
                        && targets.size() == 1
                        && singletonElements >= 1L
                        && singletonElements <= UINT32_MAX
                        ? MetalPreparedRoute.CUSTOM_SINGLE_NEG
                        : MetalPreparedRoute.MPSGRAPH;
        var internalValues = new ArrayList<ValueId>();
        if (route == MetalPreparedRoute.CUSTOM_PROGRAM) {
            for (ValueId valueId : valueIds) {
                if (!feeds.contains(valueId) && !targets.contains(valueId)) {
                    internalValues.add(valueId);
                }
            }
        }
        int[] internalIndices = indices(internalValues, valueIndexes);
        long[] internalBytes = requiredBytes(internalValues, graphValues);
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(
                feeds.size() + targets.size() + internalValues.size());
        for (int index = 0; index < feeds.size(); index++) {
            int alignment = graphValues.get(feeds.get(index)).descriptor().dataType().byteWidth();
            declarations.add(new PreparationResourceRequirement.Buffer(
                    feeds.get(index), feedBytes[index], alignment));
        }
        for (int index = 0; index < targets.size(); index++) {
            int alignment = graphValues.get(targets.get(index)).descriptor().dataType().byteWidth();
            declarations.add(new PreparationResourceRequirement.Buffer(
                    targets.get(index), targetBytes[index], alignment));
        }
        for (int index = 0; index < internalValues.size(); index++) {
            int alignment = graphValues.get(internalValues.get(index))
                    .descriptor().dataType().byteWidth();
            declarations.add(new PreparationResourceRequirement.Buffer(
                    internalValues.get(index), internalBytes[index], alignment));
        }
        Optional<PreparationResourceRequirement.Workspace> heuristicWorkspace = workspace(
                route, feeds.size(), targets.size(), valueIds.size());
        var heuristicPlan = new MetalNegPreparationPlan(
                context.numericalProfile(),
                context.partition(), context.partitionDag(), deviceContext,
                route,
                valueIds, descriptors, valueStates, graphProgram,
                feeds, feedIndices, targets, targetIndices,
                internalValues, internalIndices, internalBytes, declarations, feedSplats,
                feedSplatSources, heuristicWorkspace, feedBytes, targetBytes);
        MetalNegTuningBatch freshBatch = new MetalNegRouteCandidateGenerator()
                .generate(context, heuristicPlan, MetalNegTuningBatch.Candidate.values().length);
        var suppliedHandoff = context.backendInputs().tuningHandoff();
        if (suppliedHandoff.isPresent()) {
            var handoff = suppliedHandoff.orElseThrow();
            if (handoff.partition() != context.partition()
                    || !handoff.candidateBatch().compatibility()
                            .equals(freshBatch.compatibility())) {
                throw new IllegalArgumentException(
                        "Metal NEG tuning handoff is stale or foreign");
            }
            if (handoff.selectedDecision().isPresent()) {
                var decision = handoff.selectedDecision().orElseThrow();
                if (handoff.candidateBatch().find(decision.selectedCandidate()).isEmpty()) {
                    throw new IllegalArgumentException(
                            "Metal NEG tuning handoff is stale or foreign");
                }
                route = decision.match(freshBatch)
                        .orElseThrow(() -> new IllegalArgumentException(
                                "Metal NEG tuning decision is incompatible"))
                        .route();
            }
        }
        if (forcedRoute != null) {
            boolean approved = freshBatch.candidates().stream()
                    .anyMatch(candidate -> candidate.route() == forcedRoute);
            if (!approved) {
                throw new IllegalArgumentException(
                        "forced Metal route is not approved for this partition");
            }
            route = forcedRoute;
        }

        Optional<PreparationResourceRequirement.Workspace> selectedWorkspace = workspace(
                route, feeds.size(), targets.size(), valueIds.size());
        MetalTraceProducer traceProducer = context.backendInputs().traceProducer();
        MetalTraceProducer.PreparedUnit traceUnit = traceProducer == null
                ? null
                : traceProducer.prepareUnit(context.numericalProfile(), route);
        var plan = traceProducer == null && route == heuristicPlan.route()
                ? heuristicPlan
                : new MetalNegPreparationPlan(
                        context.numericalProfile(),
                        context.partition(), context.partitionDag(), deviceContext,
                        route,
                        valueIds, descriptors, valueStates, graphProgram,
                        feeds, feedIndices, targets, targetIndices,
                        internalValues, internalIndices, internalBytes, declarations, feedSplats,
                        feedSplatSources, selectedWorkspace, feedBytes, targetBytes, traceUnit);
        var allDeclarations = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.addressWorkspace().ifPresent(allDeclarations::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, allDeclarations);
    }

    private static Optional<PreparationResourceRequirement.Workspace> workspace(
            MetalPreparedRoute route,
            int feedCount,
            int targetCount,
            int valueCount) {
        if (route == MetalPreparedRoute.CUSTOM_SINGLE_NEG) return Optional.empty();
        long pointerCount = route == MetalPreparedRoute.CUSTOM_PROGRAM
                ? Math.addExact((long) valueCount, targetCount)
                : Math.addExact((long) feedCount, targetCount);
        long workspaceBytes = Math.multiplyExact(pointerCount, Long.BYTES);
        return Optional.of(new PreparationResourceRequirement.Workspace(
                0L, workspaceBytes, Long.BYTES));
    }

    private static int index(ValueId id, Map<ValueId, GraphValue> values,
            Map<ValueId, Integer> indexes, List<ValueId> ids,
            List<TensorDescriptor> descriptors) {
        Integer existing = indexes.get(id);
        if (existing != null) return existing;
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is missing: " + id);
        int result = ids.size();
        indexes.put(id, result);
        ids.add(id);
        descriptors.add(value.descriptor());
        return result;
    }

    private static LogicalMemoryRequirement require(
            Map<ValueId, LogicalMemoryRequirement> requirements, ValueId id) {
        LogicalMemoryRequirement result = requirements.get(id);
        if (result == null) throw new IllegalArgumentException("logical requirement is missing: " + id);
        return result;
    }

    private static void validateLogicalPartitionFacts(
            io.github.pho001.synaptik.prepare.analysis.PartitionDag dag,
            io.github.pho001.synaptik.planning.partition.PlannedPartition partition,
            List<ValueId> valueIds,
            Map<ValueId, LogicalMemoryRequirement> requirements) {
        for (ValueId valueId : valueIds) {
            LogicalMemoryRequirement requirement = require(requirements, valueId);
            var producer = requirement.producerPartition();
            if (producer.isPresent()
                    && producer.orElseThrow() != partition
                    && producer.orElseThrow().equals(partition)) {
                throw new IllegalArgumentException(
                        "logical producer uses a structural copy of the analyzed partition: "
                                + valueId);
            }
            boolean locallyProduced = dag.producer(valueId).isPresent();
            if (locallyProduced != (producer.isPresent() && producer.orElseThrow() == partition)) {
                throw new IllegalArgumentException(
                        "logical producer fact disagrees with the partition DAG: " + valueId);
            }

            boolean exactLocalConsumer = false;
            for (var consumer : requirement.consumerPartitions()) {
                if (consumer == partition) {
                    exactLocalConsumer = true;
                } else if (consumer.equals(partition)) {
                    throw new IllegalArgumentException(
                            "logical consumer uses a structural copy of the analyzed partition: "
                                    + valueId);
                }
            }
            boolean locallyConsumed = !dag.consumers(valueId).isEmpty();
            if (locallyConsumed != exactLocalConsumer) {
                throw new IllegalArgumentException(
                        "logical consumer fact disagrees with the partition DAG: " + valueId);
            }
        }
    }

    private static boolean canonicalFeed(TensorDescriptor descriptor) {
        return descriptor.layout().isPresent()
                && descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()));
    }

    private static int[] indices(List<ValueId> ids, Map<ValueId, Integer> indexes) {
        int[] result = new int[ids.size()];
        for (int index = 0; index < result.length; index++) {
            Integer value = indexes.get(ids.get(index));
            if (value == null) throw new IllegalArgumentException("boundary value is not indexed");
            result[index] = value;
        }
        return result;
    }

    private static long[] requiredBytes(List<ValueId> ids, Map<ValueId, GraphValue> values) {
        long[] result = new long[ids.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = byteSize(values.get(ids.get(index)).descriptor());
        }
        return result;
    }

    private static MetalMpsGraphProgram.Node lower(
            Operation operation, int[] inputs, int output) {
        OperationKind kind = operation.kind();
        if (kind == CastKind.CAST) {
            CastAttrs attrs = (CastAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.CAST, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                    new long[] {MetalMpsGraphProgram.dataTypeWire(attrs.targetDataType())});
        }
        if (kind == AxisGatherKind.GATHER_ELEMENTS) {
            IndexAxisAttrs attrs = (IndexAxisAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.AXIS,
                    new long[] {attrs.axis()});
        }
        if (kind == GatherNdKind.GATHER_ND) {
            GatherNdAttrs attrs = (GatherNdAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.GATHER_ND,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.GATHER_ND,
                    new long[] {attrs.batchDimensions()});
        }
        if (kind == SelectKind.SELECT) {
            SelectAttrs attrs = (SelectAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.SELECT,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.SELECT,
                    new long[] {attrs.axis(), attrs.index()});
        }
        if (kind == PadKind.PAD) {
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.PAD,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.PAD,
                    padWords((PadAttrs) operation.attrs()));
        }
        if (kind == SliceKind.SLICE) {
            long[] words = operation.attrs() instanceof SliceAttrs attrs
                    ? sliceWords(attrs)
                    : cropWords((CropToShapeAttrs) operation.attrs());
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.SLICE,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.SLICE, words);
        }
        if (kind instanceof TensorCompositionKind composition) {
            CompositionAxisAttrs attrs = (CompositionAxisAttrs) operation.attrs();
            MetalMpsGraphProgram.NodeKind nodeKind =
                    composition == TensorCompositionKind.CONCAT
                            ? MetalMpsGraphProgram.NodeKind.CONCAT
                            : MetalMpsGraphProgram.NodeKind.STACK;
            return MetalMpsGraphProgram.Node.generic(
                    nodeKind, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {attrs.axis()});
        }
        if (kind == TileKind.TILE) {
            TileAttrs attrs = (TileAttrs) operation.attrs();
            long[] words = new long[attrs.repeats().size() + 1];
            words[0] = attrs.repeats().size();
            for (int index = 0; index < attrs.repeats().size(); index++) {
                words[index + 1] = attrs.repeats().get(index);
            }
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.TILE,
                    inputs, new int[] {output}, MetalMpsGraphProgram.AttributeKind.TILE, words);
        }
        if (kind == WindowTransformKind.UNFOLD2D) {
            Object raw = operation.attrs();
            Window2dAttrs window = raw instanceof Unfold2dAttrs explicit
                    ? explicit.window() : (Window2dAttrs) raw;
            boolean padded = raw instanceof Unfold2dAttrs;
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                    inputs, new int[] {output},
                    padded ? MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_2D
                            : MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    windowWords(window, padded ? ((Unfold2dAttrs) raw).paddingValue() : null));
        }
        if (kind == WindowTransformKind.UNFOLD3D) {
            Object raw = operation.attrs();
            Window3dAttrs window = raw instanceof Unfold3dAttrs explicit
                    ? explicit.window() : (Window3dAttrs) raw;
            boolean padded = raw instanceof Unfold3dAttrs;
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.UNFOLD3D,
                    inputs, new int[] {output},
                    padded ? MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_3D
                            : MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                    windowWords(window, padded ? ((Unfold3dAttrs) raw).paddingValue() : null));
        }
        if (kind == AxisGatherKind.GATHER) {
            IndexAxisAttrs attrs = (IndexAxisAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.gather(
                    inputs[0], inputs[1], output, attrs.axis());
        }
        if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
            ScatterElementsAttrs attrs = (ScatterElementsAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.scatterElements(
                    inputs[0], inputs[1], inputs[2], output, attrs.axis());
        }
        if (kind == OneHotKind.ONE_HOT) {
            OneHotAttrs attrs = (OneHotAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.oneHot(inputs[0], output, attrs.depth());
        }
        if (kind == UnaryElementwiseKind.NEG) {
            return MetalMpsGraphProgram.Node.neg(inputs[0], output);
        }
        if (kind == UnaryElementwiseKind.ABS) {
            return MetalMpsGraphProgram.Node.abs(inputs[0], output);
        }
        if (kind == UnaryElementwiseKind.RECIPROCAL) {
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.RECIPROCAL,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.NONE,
                    new long[0]);
        }
        if (kind == UnaryElementwiseKind.FLOOR
                || kind == UnaryElementwiseKind.CEIL
                || kind == UnaryElementwiseKind.SIGN
                || kind == UnaryElementwiseKind.RELU) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch ((UnaryElementwiseKind) kind) {
                case FLOOR -> MetalMpsGraphProgram.NodeKind.FLOOR;
                case CEIL -> MetalMpsGraphProgram.NodeKind.CEIL;
                case SIGN -> MetalMpsGraphProgram.NodeKind.SIGN;
                case RELU -> MetalMpsGraphProgram.NodeKind.RELU;
                default -> throw new IllegalStateException("unreachable exact unary kind");
            };
            return MetalMpsGraphProgram.Node.generic(
                    nodeKind, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        }
        if (kind instanceof FloatingClassificationKind classification) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (classification) {
                case IS_FINITE -> MetalMpsGraphProgram.NodeKind.IS_FINITE;
                case IS_NAN -> MetalMpsGraphProgram.NodeKind.IS_NAN;
                case IS_INF -> MetalMpsGraphProgram.NodeKind.IS_INF;
            };
            return MetalMpsGraphProgram.Node.generic(
                    nodeKind, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        }
        if (kind instanceof BooleanLogicalKind logical) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (logical) {
                case AND -> MetalMpsGraphProgram.NodeKind.LOGICAL_AND;
                case OR -> MetalMpsGraphProgram.NodeKind.LOGICAL_OR;
                case NOT -> MetalMpsGraphProgram.NodeKind.LOGICAL_NOT;
            };
            return MetalMpsGraphProgram.Node.generic(
                    nodeKind, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        }
        if (kind == WhereSelectionKind.WHERE) {
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.WHERE, inputs, new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        }
        if (kind instanceof BinaryComparisonKind comparison) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (comparison) {
                case GREATER_THAN -> MetalMpsGraphProgram.NodeKind.GT;
                case GREATER_OR_EQUAL -> MetalMpsGraphProgram.NodeKind.GE;
                case LESS_THAN -> MetalMpsGraphProgram.NodeKind.LT;
                case LESS_OR_EQUAL -> MetalMpsGraphProgram.NodeKind.LE;
                case EQUAL -> MetalMpsGraphProgram.NodeKind.EQ;
                case NOT_EQUAL -> MetalMpsGraphProgram.NodeKind.NE;
            };
            return MetalMpsGraphProgram.Node.binary(
                    nodeKind, inputs[0], inputs[1], output);
        }
        if (kind instanceof BinaryArithmeticKind binary) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (binary) {
                case ADD -> MetalMpsGraphProgram.NodeKind.ADD;
                case SUB -> MetalMpsGraphProgram.NodeKind.SUB;
                case MUL -> MetalMpsGraphProgram.NodeKind.MUL;
                case DIV -> MetalMpsGraphProgram.NodeKind.DIV;
                case MIN -> MetalMpsGraphProgram.NodeKind.TENSOR_MIN;
                case MAX -> MetalMpsGraphProgram.NodeKind.TENSOR_MAX;
                default -> throw new IllegalArgumentException(
                        "unsupported Metal binary operation: " + binary);
            };
            return MetalMpsGraphProgram.Node.binary(
                    nodeKind, inputs[0], inputs[1], output);
        }
        if (kind instanceof ScalarElementwiseKind scalar) {
            if (scalar == ScalarElementwiseKind.CLAMP) {
                ClampRangeAttrs attrs = (ClampRangeAttrs) operation.attrs();
                return MetalMpsGraphProgram.Node.clamp(
                        inputs[0],
                        output,
                        Float.floatToRawIntBits(attrs.minValue().float32Value()),
                        Float.floatToRawIntBits(attrs.maxValue().float32Value()));
            }
            ScalarValueAttrs attrs = (ScalarValueAttrs) operation.attrs();
            MetalMpsGraphProgram.NodeKind nodeKind = switch (scalar) {
                case ADD -> MetalMpsGraphProgram.NodeKind.SCALAR_ADD;
                case SUB -> MetalMpsGraphProgram.NodeKind.SCALAR_SUB;
                case MUL -> MetalMpsGraphProgram.NodeKind.SCALAR_MUL;
                case DIV -> MetalMpsGraphProgram.NodeKind.SCALAR_DIV;
                case MIN -> MetalMpsGraphProgram.NodeKind.SCALAR_MIN;
                case MAX -> MetalMpsGraphProgram.NodeKind.SCALAR_MAX;
                case POW -> throw new IllegalArgumentException(
                        "unsupported Metal scalar operation: " + scalar);
                case CLAMP -> throw new IllegalStateException("CLAMP handled above");
            };
            return MetalMpsGraphProgram.Node.scalarValue(
                    nodeKind,
                    inputs[0],
                    output,
                    Float.floatToRawIntBits(attrs.value().float32Value()));
        }
        if (kind instanceof CumulativeScanKind scan) {
            CumulativeScanAttrs attrs = (CumulativeScanAttrs) operation.attrs();
            MetalMpsGraphProgram.NodeKind nodeKind = scan == CumulativeScanKind.CUM_SUM
                    ? MetalMpsGraphProgram.NodeKind.CUM_SUM
                    : MetalMpsGraphProgram.NodeKind.CUM_PROD;
            return MetalMpsGraphProgram.Node.scan(
                    nodeKind,
                    inputs[0],
                    output,
                    attrs.axis(),
                    attrs.exclusive(),
                    attrs.reverse());
        }
        if (kind == MatmulKind.MATMUL) {
            return MetalMpsGraphProgram.Node.matmul(inputs[0], inputs[1], output);
        }
        if (kind instanceof AggregateReductionKind reduction) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (reduction) {
                case SUM -> MetalMpsGraphProgram.NodeKind.SUM;
                case MEAN -> MetalMpsGraphProgram.NodeKind.MEAN;
                case MIN -> MetalMpsGraphProgram.NodeKind.REDUCTION_MIN;
                case MAX -> MetalMpsGraphProgram.NodeKind.REDUCTION_MAX;
                default -> throw new IllegalArgumentException(
                        "unsupported Metal reduction: " + reduction);
            };
            if (operation.attrs() == io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE) {
                return MetalMpsGraphProgram.Node.reduction(
                        nodeKind, inputs[0], output,
                        MetalMpsGraphProgram.ReductionForm.FULL, List.of(), false);
            }
            if (operation.attrs() instanceof AxisReductionAttrs attrs) {
                return MetalMpsGraphProgram.Node.reduction(
                        nodeKind, inputs[0], output,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(attrs.axis()), attrs.keepDimensions());
            }
            if (operation.attrs() instanceof MultiAxisReductionAttrs attrs) {
                return MetalMpsGraphProgram.Node.reduction(
                        nodeKind, inputs[0], output,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        attrs.axes(), attrs.keepDimensions());
            }
            SumToShapeAttrs attrs = (SumToShapeAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.sumToShape(
                    inputs[0], output, attrs.targetShape().toLongArray());
        }
        if (kind == ContiguousKind.CONTIGUOUS) {
            return MetalMpsGraphProgram.Node.contiguous(inputs[0], output);
        }
        if (kind instanceof ShapeTransformKind transform) {
            TargetShapeAttrs attrs = (TargetShapeAttrs) operation.attrs();
            MetalMpsGraphProgram.NodeKind nodeKind =
                    transform == ShapeTransformKind.RESHAPE
                            ? MetalMpsGraphProgram.NodeKind.RESHAPE
                            : MetalMpsGraphProgram.NodeKind.EXPAND;
            return MetalMpsGraphProgram.Node.targetShape(
                    nodeKind, inputs[0], output, attrs.targetShape().toLongArray());
        }
        if (kind == AxisTransformKind.PERMUTE) {
            PermutationAttrs attrs = (PermutationAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.permutation(
                    inputs[0], output, attrs.axes());
        }
        if (kind == WindowTransformKind.UNFOLD_AXIS) {
            UnfoldAxisAttrs windowAttrs = (UnfoldAxisAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.unfoldAxis(
                    inputs[0], output,
                    windowAttrs.axis(), windowAttrs.size(), windowAttrs.step());
        }
        AxisTransformAttrs attrs = (AxisTransformAttrs) operation.attrs();
        MetalMpsGraphProgram.NodeKind nodeKind = kind == AxisTransformKind.EXPAND_DIMS
                ? MetalMpsGraphProgram.NodeKind.EXPAND_DIMS
                : MetalMpsGraphProgram.NodeKind.SQUEEZE;
        return MetalMpsGraphProgram.Node.axis(
                nodeKind, inputs[0], output, attrs.axis());
    }

    private static long[] padWords(PadAttrs attrs) {
        int rank = attrs.before().size();
        long[] scalar = scalarWords(attrs.constantValue());
        long[] words = new long[1 + rank * 2 + scalar.length];
        words[0] = rank;
        for (int axis = 0; axis < rank; axis++) {
            words[1 + axis] = attrs.before().get(axis);
            words[1 + rank + axis] = attrs.after().get(axis);
        }
        System.arraycopy(scalar, 0, words, 1 + rank * 2, scalar.length);
        return words;
    }

    private static long[] sliceWords(SliceAttrs attrs) {
        int count = attrs.axes().size();
        long[] words = new long[1 + count * 4];
        words[0] = count;
        for (int index = 0; index < count; index++) {
            words[1 + index] = attrs.starts().get(index);
            words[1 + count + index] = attrs.lengths().get(index);
            words[1 + count * 2 + index] = attrs.axes().get(index);
            words[1 + count * 3 + index] = attrs.steps().get(index);
        }
        return words;
    }

    private static long[] cropWords(CropToShapeAttrs attrs) {
        long[] target = attrs.targetShape().toLongArray();
        long[] prefix = attrs.prefixShape().toLongArray();
        if (target.length != prefix.length) {
            throw new IllegalArgumentException("Metal crop ranks must agree");
        }
        long[] words = new long[1 + target.length * 4];
        words[0] = target.length;
        for (int axis = 0; axis < target.length; axis++) {
            words[1 + axis] = prefix[axis];
            words[1 + target.length + axis] = target[axis];
            words[1 + target.length * 2 + axis] = axis;
            words[1 + target.length * 3 + axis] = 1L;
        }
        return words;
    }

    private static long[] windowWords(Window2dAttrs window, ScalarValue padding) {
        long[] words = new long[padding == null ? 9 : 11];
        words[0] = window.kernelHeight();
        words[1] = window.kernelWidth();
        words[2] = window.strideHeight();
        words[3] = window.strideWidth();
        words[4] = window.paddingHeight();
        words[5] = window.paddingWidth();
        words[6] = window.dilationHeight();
        words[7] = window.dilationWidth();
        words[8] = window.ceilMode() ? 1L : 0L;
        if (padding != null) {
            long[] scalar = scalarWords(padding);
            words[9] = scalar[0];
            words[10] = scalar[1];
        }
        return words;
    }

    private static long[] windowWords(Window3dAttrs window, ScalarValue padding) {
        long[] words = new long[padding == null ? 13 : 15];
        words[0] = window.kernelDepth();
        words[1] = window.kernelHeight();
        words[2] = window.kernelWidth();
        words[3] = window.strideDepth();
        words[4] = window.strideHeight();
        words[5] = window.strideWidth();
        words[6] = window.paddingDepth();
        words[7] = window.paddingHeight();
        words[8] = window.paddingWidth();
        words[9] = window.dilationDepth();
        words[10] = window.dilationHeight();
        words[11] = window.dilationWidth();
        words[12] = window.ceilMode() ? 1L : 0L;
        if (padding != null) {
            long[] scalar = scalarWords(padding);
            words[13] = scalar[0];
            words[14] = scalar[1];
        }
        return words;
    }

    private static long[] scalarWords(ScalarValue value) {
        long bits = switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case FLOAT32 -> Integer.toUnsignedLong(
                    Float.floatToRawIntBits(value.float32Value()));
            case BFLOAT16 -> Short.toUnsignedLong(value.bfloat16Bits());
            case INT32 -> Integer.toUnsignedLong(value.int32Value());
            case INT64 -> value.int64Value();
            case BOOL -> value.booleanValue() ? 1L : 0L;
        };
        return new long[] {MetalMpsGraphProgram.dataTypeWire(value.dataType()), bits};
    }

    private static long byteSize(TensorDescriptor descriptor) {
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) throw new IllegalArgumentException(
                    "Metal dimensions must be positive");
            elements = Math.multiplyExact(elements, dimension);
        }
        return Math.multiplyExact(elements, descriptor.dataType().byteWidth());
    }
}
