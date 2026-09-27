package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
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
 * For both profiles, it walks explicit unavailable/canonical/affine-view/materialized-layout states
 * in node order for the retained exact unary, affine, canonicalization, movement/indexing,
 * replacement/fold/aggregate, BOOL, ordering/arg-extrema, maximum-pooling, promoted integral
 * MATMUL, and zero-input {@code INITIAL_STATE} domains. Task 0066 admits all six carriers through
 * exact affine movement, INT32/INT64 index roles, all 36 cast pairs, floating classification and
 * promotion, and the precise legal first-order saved-role relations. Authenticated local
 * zero-stride descendants retain their logical layouts while preparation assigns independently safe
 * physical buffers. Under {@code ACCELERATOR}, analysis additionally accepts the documented FLOAT32
 * arithmetic/reduction/MSE/MATMUL/convolution/average-pooling rows, no-gradient BFLOAT16/FLOAT32
 * mixed MATMUL, canonical FLOAT32 {@code DROPOUT}, and the exact rank-one no-gradient FLOAT32
 * {@code L1_NORM} and {@code SCATTER_ADD} occurrences. Scatter requires canonical
 * base/index/update/output representations; the compiler inserts explicit {@code CONTIGUOUS}
 * between its generated zero-base {@code EXPAND} and Scatter. Its INT32/INT64 indices must remain
 * a materialized partition feed. Random lowering preserves initializer key/counter words, dropout's
 * raw binary64 probability, and all ordered value, mask, and state edges; recurrent nodes remain
 * rejected. An affine MATMUL operand is authenticated to the exact earlier local identity-prefix,
 * last-two-axis {@code PERMUTE} on that consuming edge. Schema-sixteen lowering emits one bounded
 * self-describing image over stable type wires 1..6, complete operation registry 1..115, attribute
 * registry 0..41, and the explicit prepared route. Production capability is exactly 85 operation
 * kinds; additional structural recipes remain inaccessible to this analysis. Every selected
 * Task-0066 or Task-0069 occurrence fixes the whole partition to {@code CUSTOM_PROGRAM}, with no
 * MPSGraph candidate, retry, fallback, timing, or autotuning.
 * Rank-zero values participate only where exact capability permits them. Analysis freshly
 * regenerates the complete candidate batch. Every supplied handoff authenticates its exact
 * partition, schema, workload, profile, and session target; an absent decision preserves the
 * singleton-NEG heuristic, while a present decision must additionally authenticate its candidate
 * identity. Any shared custom-program node fixes the whole partition to its custom program route
 * before exact declarations, including a declared run-owned buffer for every internal logical
 * value. Package-private tests may force only another candidate already approved by that freshly
 * validated batch; production has no corresponding input or switch. Published SELECT/SLICE values
 * retain logical storage layouts and declarations cover their full physical referenced spans; other
 * affine views retain the existing dense represented-order geometry. Analysis allocates no physical
 * resource and never changes partition ownership or capability.
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
     * <p>The ordinary analysis, capability checks, candidate regeneration, and any supplied handoff
     * authentication complete first. The force succeeds only when the exact route is a member of that
   * freshly generated batch; otherwise analysis fails before native creation.
     *
     * @param context non-null complete partition-local facts and borrowed Metal context
     * @param forcedRoute non-null exact route to require
     * @return non-null analysis retaining the exact planned partition and forced route
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if normal analysis rejects the partition or the route is not
     * currently approved for it
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
        var localSingletonHeight = new LinkedHashMap<ValueId, Boolean>();
    var physicalLayouts = new LinkedHashMap<ValueId, LayoutDescriptor>();
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
                    state = feedState(inputValue.descriptor());
                    if (state == MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                        throw new IllegalArgumentException(
                                "Metal graph feeds require resolved positive non-overlapping"
                                        + " storage geometry: " + inputId);
                    }
                    states.put(inputId, state);
                    feeds.add(inputId);
                    localTranspose.put(inputId, false);
                    localSingletonHeight.put(inputId, false);
          physicalLayouts.put(inputId, inputValue.descriptor().layout().orElseThrow());
                }
                inputStates.add(state);
            }
            if (node.outputs().isEmpty()) {
                throw new IllegalArgumentException(
                        "Metal supported-operation partition requires node outputs");
            }
            var outputDescriptors = new ArrayList<TensorDescriptor>(node.outputs().size());
            for (ValueId outputId : node.outputs()) {
                GraphValue outputValue = graphValues.get(outputId);
                if (outputValue == null) {
                    throw new IllegalArgumentException(
                            "partition value is missing: " + outputId);
                }
                if (states.containsKey(outputId) || valueIndexes.containsKey(outputId)) {
                    throw new IllegalArgumentException(
                            "Metal output must be produced exactly once in topological order");
                }
                outputDescriptors.add(outputValue.descriptor());
            }
            if (!MetalCapabilityProvider.supportsOccurrence(
                    numericalProfile,
                    node.operation(),
                    inputDescriptors,
                    outputDescriptors)) {
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
            int[] outputIndices = new int[node.outputs().size()];
            for (int outputPosition = 0;
                    outputPosition < outputIndices.length;
                    outputPosition++) {
                outputIndices[outputPosition] = index(
                        node.outputs().get(outputPosition),
                        graphValues,
                        valueIndexes,
                        valueIds,
                        descriptors);
            }
            MetalMpsGraphProgram.Node lowered = lower(
                    node.operation(),
                    inputIndices,
                    outputIndices,
                    inputDescriptors,
                    outputDescriptors);
            if (lowered.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD) {
                ValueId indexFeed = node.inputs().get(1);
                if (!feeds.contains(indexFeed) || context.constants().containsKey(indexFeed)) {
                    throw new IllegalArgumentException(
                            "Metal SCATTER_ADD indices must be a materialized partition feed");
                }
            }
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
            if (isTask0064TwoDimensional(lowered.kind())) {
                for (int inputIndex = 0; inputIndex < inputStates.size(); inputIndex++) {
                    if (inputStates.get(inputIndex)
                                    == MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                            && !Boolean.TRUE.equals(localSingletonHeight.get(
                                    node.inputs().get(inputIndex)))) {
                        throw new IllegalArgumentException(
                                "Metal Conv2d/Pool2d affine input is not an authenticated local"
                                        + " singleton-height expansion");
                    }
                }
            }
            if (!lowered.kind().acceptsCardinality(
                            inputStates.size(), outputIndices.length)
                    || inputStates.stream().anyMatch(state -> !lowered.kind().accepts(state))) {
                throw new IllegalArgumentException(
                        "Metal node input value state is unavailable or incompatible");
            }
            programNodes.add(lowered);
            boolean exactLocalTranspose = outputIndices.length == 1
                    && isExactLocalTranspose(
                            lowered,
                            inputStates,
                            inputDescriptors,
                            outputDescriptors.getFirst(),
                            node.operation());
            boolean exactLocalSingletonHeight = outputIndices.length == 1
                    && isExactLocalSingletonHeight(
                            lowered,
                            inputStates,
                            inputDescriptors,
                            outputDescriptors.getFirst(),
                            node.operation());
            for (int outputPosition = 0; outputPosition < node.outputs().size(); outputPosition++) {
        ValueId outputId = node.outputs().get(outputPosition);
                localTranspose.put(outputId, exactLocalTranspose);
                localSingletonHeight.put(outputId, exactLocalSingletonHeight);
        TensorDescriptor outputDescriptor = outputDescriptors.get(outputPosition);
        physicalLayouts.put(
            outputId,
            physicalOutputLayout(
                node.operation(), node.inputs(), outputDescriptor, physicalLayouts));
        boolean denseLogicalView =
            (lowered.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                    || lowered.kind() == MetalMpsGraphProgram.NodeKind.SLICE)
                && outputDescriptor.layout().orElseThrow().hasZeroStride();
                states.put(outputId,
            denseLogicalView
                ? MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                : lowered.kind().outputState());
            }
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
        long[] feedBytes = requiredBytes(feeds, graphValues, physicalLayouts);
        long[] targetBytes = requiredBytes(targets, graphValues, physicalLayouts);
        boolean containsCustomProgram = graphProgram.nodes().stream()
                .anyMatch(node -> usesCustomProgram(node, descriptors));
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
        long[] internalBytes = requiredBytes(internalValues, graphValues, physicalLayouts);
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
    List<LayoutDescriptor> physicalValueLayouts =
        valueIds.stream()
            .map(
                value ->
                    Objects.requireNonNull(physicalLayouts.get(value), "physical value layout"))
            .toList();
        var heuristicPlan = new MetalNegPreparationPlan(
                context.numericalProfile(),
                context.partition(), context.partitionDag(), deviceContext,
                route,
                valueIds, descriptors,
            physicalValueLayouts, valueStates, graphProgram,
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
                        valueIds, descriptors,
                physicalValueLayouts, valueStates, graphProgram,
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

    private static MetalMpsGraphProgram.ValueState feedState(TensorDescriptor descriptor) {
        if (!MetalCapabilityProvider.supportedStorageLayout(descriptor, 0)) {
            return MetalMpsGraphProgram.ValueState.UNAVAILABLE;
        }
        return descriptor.layout().orElseThrow().equals(
                        LayoutDescriptor.contiguous(descriptor.shape()))
                ? MetalMpsGraphProgram.ValueState.CANONICAL
                : MetalMpsGraphProgram.ValueState.MATERIALIZED_LAYOUT;
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

    private static LayoutDescriptor physicalOutputLayout(
      Operation operation,
      List<ValueId> inputIds,
      TensorDescriptor output,
      Map<ValueId, LayoutDescriptor> physicalLayouts) {
    if (operation.kind() != SelectKind.SELECT && operation.kind() != SliceKind.SLICE) {
      return LayoutDescriptor.contiguous(output.shape());
    }
    if (inputIds.size() != 1) {
      throw new IllegalArgumentException("Metal physical view derivation requires one input");
    }
    LayoutDescriptor input =
        Objects.requireNonNull(physicalLayouts.get(inputIds.getFirst()), "physical input layout");
    long[] strides = input.strides();
    long offset = input.storageOffset();
    if (operation.kind() == SelectKind.SELECT) {
      SelectAttrs attrs = (SelectAttrs) operation.attrs();
      int axis = attrs.axis();
      offset = Math.addExact(offset, Math.multiplyExact(attrs.index(), strides[axis]));
      long[] selected = new long[strides.length - 1];
      System.arraycopy(strides, 0, selected, 0, axis);
      System.arraycopy(strides, axis + 1, selected, axis, strides.length - axis - 1);
      strides = selected;
    } else if (operation.attrs() instanceof CropToShapeAttrs crop) {
      long[] prefix = crop.prefixShape().toLongArray();
      if (prefix.length != strides.length) {
        throw new IllegalArgumentException("Metal CropToShape physical rank is incompatible");
      }
      for (int axis = 0; axis < strides.length; axis++) {
        offset = Math.addExact(offset, Math.multiplyExact(prefix[axis], strides[axis]));
      }
    } else {
      SliceAttrs attrs = (SliceAttrs) operation.attrs();
      for (int index = 0; index < attrs.axes().size(); index++) {
        int axis = attrs.axes().get(index);
        long step = attrs.steps().get(index);
        if (step <= 0L) {
          throw new IllegalArgumentException(
              "Metal SLICE physical derivation requires positive steps");
        }
        offset =
            Math.addExact(offset, Math.multiplyExact(attrs.starts().get(index), strides[axis]));
        strides[axis] = Math.multiplyExact(strides[axis], step);
      }
    }
    LayoutDescriptor physical = LayoutDescriptor.of(output.shape(), strides, offset, true);
    TensorDescriptor physicalDescriptor =
        new TensorDescriptor(
            output.dataType(),
            output.shape(),
            java.util.Optional.of(physical),
            output.requiresGrad());
    if (!MetalCapabilityProvider.supportedStorageLayout(physicalDescriptor, 0)) {
      throw new IllegalArgumentException(
          "Metal derived physical view is not positive and non-overlapping");
    }
    return physical;
  }

  private static long[] requiredBytes(
            List<ValueId> ids,
            Map<ValueId, GraphValue> values,
            Map<ValueId, LayoutDescriptor> physicalLayouts) {
        long[] result = new long[ids.size()];
        for (int index = 0; index < result.length; index++) {
            ValueId id = ids.get(index);
      LayoutDescriptor physical =
          Objects.requireNonNull(physicalLayouts.get(id), "physicalLayouts[" + id + "]");
            result[index] =
          Math.multiplyExact(
              physical.referencedElementSpan(),values.get(id).descriptor().dataType().byteWidth());
        }
        return result;
    }

    private static boolean usesCustomProgram(
            MetalMpsGraphProgram.Node node, List<TensorDescriptor> descriptors) {
        if (node.kind().isCustomProgramOperation()) return true;
        if (node.kind() != MetalMpsGraphProgram.NodeKind.MATMUL) return false;
        TensorDescriptor left = descriptors.get(node.firstInputIndex());
        TensorDescriptor right = descriptors.get(node.secondInputIndex());
        TensorDescriptor output = descriptors.get(node.outputIndex());
        return left.dataType() != DataType.FLOAT32
                || right.dataType() != DataType.FLOAT32
                || output.dataType() != DataType.FLOAT32
                || left.shape().rank() != 2
                || right.shape().rank() != 2
                || output.shape().rank() != 2;
    }
    private static boolean isTask0064TwoDimensional(
            MetalMpsGraphProgram.NodeKind kind) {
        return kind == MetalMpsGraphProgram.NodeKind.CONV2D
                || kind == MetalMpsGraphProgram.NodeKind.MAX_POOL2D
                || kind == MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D;
    }

    private static boolean isExactLocalSingletonHeight(
            MetalMpsGraphProgram.Node lowered,
            List<MetalMpsGraphProgram.ValueState> inputStates,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            Operation operation) {
        if (lowered.kind() != MetalMpsGraphProgram.NodeKind.EXPAND_DIMS
                || inputStates.getFirst() != MetalMpsGraphProgram.ValueState.CANONICAL
                || !(operation.attrs() instanceof AxisTransformAttrs attrs)
                || attrs.axis() != 2
                || inputs.getFirst().shape().rank() != 3
                || output.shape().rank() != 4
                || output.shape().toLongArray()[2] != 1L) {
            return false;
        }
        long[] shape = output.shape().toLongArray();
        LayoutDescriptor expected = LayoutDescriptor.of(
                output.shape(),
                new long[] {
                    Math.multiplyExact(shape[1], shape[3]), shape[3], shape[3], 1L
                },
                0L,
                true);
        return output.layout().filter(expected::equals).isPresent();
    }



    private static boolean isExactLocalTranspose(
            MetalMpsGraphProgram.Node lowered,
            List<MetalMpsGraphProgram.ValueState> inputStates,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            Operation operation) {
        if (lowered.kind() != MetalMpsGraphProgram.NodeKind.PERMUTE
                || inputStates.getFirst() != MetalMpsGraphProgram.ValueState.CANONICAL
                || !(operation.attrs() instanceof PermutationAttrs permutation)) {
            return false;
        }
        int rank = inputs.getFirst().shape().rank();
        if (rank < 2 || output.shape().rank() != rank || permutation.axes().size() != rank) {
            return false;
        }
        for (int axis = 0; axis < rank - 2; axis++) {
            if (permutation.axes().get(axis) != axis) return false;
        }
        if (permutation.axes().get(rank - 2) != rank - 1
                || permutation.axes().get(rank - 1) != rank - 2) {
            return false;
        }
        return output.layout().orElseThrow().equals(lastTwoTransposeLayout(output));
    }

    private static LayoutDescriptor lastTwoTransposeLayout(TensorDescriptor output) {
        int rank = output.shape().rank();
        long[] sourceShape = output.shape().toLongArray();
        long swap = sourceShape[rank - 2];
        sourceShape[rank - 2] = sourceShape[rank - 1];
        sourceShape[rank - 1] = swap;
        long[] sourceStrides =
                LayoutDescriptor.contiguous(io.github.pho001.synaptik.model.shape.Shape.of(sourceShape))
                        .strides();
        long[] outputStrides = sourceStrides.clone();
        outputStrides[rank - 2] = sourceStrides[rank - 1];
        outputStrides[rank - 1] = sourceStrides[rank - 2];
        return LayoutDescriptor.of(output.shape(), outputStrides, 0L, true);
    }

    private static MetalMpsGraphProgram.Node lower(
            Operation operation,
            int[] inputs,
            int[] outputs,
            List<TensorDescriptor> inputDescriptors,
            List<TensorDescriptor> outputDescriptors) {
        OperationKind kind = operation.kind();
        if (kind == GraphRngKind.INITIAL_STATE) {
            GraphRngStateAttrs attrs = (GraphRngStateAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.INITIAL_STATE,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.GRAPH_RNG_STATE,
                    new long[] {attrs.key(), attrs.counter()});
        }
        if (kind == DropoutKind.DROPOUT) {
            DropoutAttrs attrs = (DropoutAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.DROPOUT,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.DROPOUT,
                    new long[] {Double.doubleToRawLongBits(attrs.probability())});
        }
        if (kind == Conv2dKind.CONV2D) {
            Conv2dAttrs attrs = (Conv2dAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.CONV2D,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.CONV_2D,
                    new long[] {
                        attrs.strideHeight(), attrs.strideWidth(),
                        attrs.paddingHeight(), attrs.paddingWidth(),
                        attrs.dilationHeight(), attrs.dilationWidth(), attrs.groups()
                    });
        }
        if (kind == Conv3dKind.CONV3D) {
            Conv3dAttrs attrs = (Conv3dAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.CONV3D,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.CONV_3D,
                    new long[] {
                        attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                        attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                        attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                        attrs.groups()
                    });
        }
        if (kind instanceof Pool2dKind pool) {
            long[] words;
            if (pool == Pool2dKind.MAX_POOL2D) {
                MaxPool2dAttrs attrs = (MaxPool2dAttrs) operation.attrs();
                words = new long[] {
                    attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode() ? 1L : 0L
                };
            } else {
                AveragePool2dAttrs attrs = (AveragePool2dAttrs) operation.attrs();
                words = new long[] {
                    attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode() ? 1L : 0L
                };
            }
            return MetalMpsGraphProgram.Node.generic(
                    pool == Pool2dKind.MAX_POOL2D
                            ? MetalMpsGraphProgram.NodeKind.MAX_POOL2D
                            : MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    words);
        }
        if (kind instanceof Pool3dKind pool) {
            long[] words;
            if (pool == Pool3dKind.MAX_POOL3D) {
                MaxPool3dAttrs attrs = (MaxPool3dAttrs) operation.attrs();
                words = new long[] {
                    attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode() ? 1L : 0L
                };
            } else {
                AveragePool3dAttrs attrs = (AveragePool3dAttrs) operation.attrs();
                words = new long[] {
                    attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode() ? 1L : 0L
                };
            }
            return MetalMpsGraphProgram.Node.generic(
                    pool == Pool3dKind.MAX_POOL3D
                            ? MetalMpsGraphProgram.NodeKind.MAX_POOL3D
                            : MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                    words);
        }
        int output = outputs[0];
        TensorDescriptor outputDescriptor = outputDescriptors.getFirst();
        if (kind instanceof OrderingKind ordering) {
            SortAttrs attrs = (SortAttrs) operation.attrs();
            MetalMpsGraphProgram.NodeKind nodeKind = ordering == OrderingKind.SORT
                    ? MetalMpsGraphProgram.NodeKind.SORT
                    : MetalMpsGraphProgram.NodeKind.ARGSORT;
            return MetalMpsGraphProgram.Node.generic(
                    nodeKind,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.SORT,
                    new long[] {attrs.axis(), attrs.descending() ? 1L : 0L});
        }
        if (kind == TopKKind.TOP_K) {
            TopKAttrs attrs = (TopKAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.TOP_K,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.TOP_K,
                    new long[] {
                        attrs.axis(),
                        attrs.k(),
                        attrs.largest() ? 1L : 0L,
                        attrs.sorted() ? 1L : 0L
                    });
        }
        if (kind == AggregateReductionKind.ARG_MAX
                || kind == AggregateReductionKind.ARG_MIN) {
            ArgExtremaAttrs attrs = (ArgExtremaAttrs) operation.attrs();
            long tie = attrs.tiePolicy() == ArgExtremaTiePolicy.FIRST_INDEX ? 1L : 2L;
            return MetalMpsGraphProgram.Node.generic(
                    kind == AggregateReductionKind.ARG_MAX
                            ? MetalMpsGraphProgram.NodeKind.ARG_MAX
                            : MetalMpsGraphProgram.NodeKind.ARG_MIN,
                    inputs,
                    outputs,
                    MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA,
                    new long[] {
                        attrs.axis(), attrs.keepDimensions() ? 1L : 0L, tie
                    });
        }
        if (kind == LossKind.MEAN_SQUARED_ERROR) {
            MeanSquaredErrorAttrs attrs = (MeanSquaredErrorAttrs) operation.attrs();
            long reduction = switch (attrs.reduction()) {
                case NONE -> 1L;
                case SUM -> 2L;
                case MEAN -> 3L;
            };
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.MSE,
                    new long[] {reduction});
        }
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
        if (kind == ScatterNdKind.SCATTER_ND) {
            ScatterNdAttrs attrs = (ScatterNdAttrs) operation.attrs();
            long reduction = switch (attrs.reduction()) {
                case NONE -> 1L;
                case ADD -> 2L;
                case MUL -> 3L;
                case MAX -> 4L;
                case MIN -> 5L;
            };
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.SCATTER_ND,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.SCATTER_ND,
                    new long[] {attrs.batchDimensions(), reduction});
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
        if (kind == SliceKind.SLICE || kind == SliceKind.SLICE_UPDATE) {
            boolean crop = operation.attrs() instanceof CropToShapeAttrs;
            long[] words = operation.attrs() instanceof SliceAttrs attrs
                    ? sliceWords(attrs)
                    : cropWords((CropToShapeAttrs) operation.attrs());
            return MetalMpsGraphProgram.Node.generic(
                    kind == SliceKind.SLICE
                            ? MetalMpsGraphProgram.NodeKind.SLICE
                            : MetalMpsGraphProgram.NodeKind.SLICE_UPDATE,
                    inputs,
                    new int[] {output},
                    crop
                            ? MetalMpsGraphProgram.AttributeKind.CROP_TO_SHAPE
                            : MetalMpsGraphProgram.AttributeKind.SLICE,
                    words);
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
        if (kind == WindowTransformKind.FOLD_AXIS) {
            FoldAxisAttrs attrs = (FoldAxisAttrs) operation.attrs();
            long size = inputDescriptors.getFirst().shape().toLongArray()[
                    inputDescriptors.getFirst().shape().rank() - 1];
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.FOLD_AXIS,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.WINDOW_AXIS,
                    new long[] {attrs.axis(), size, attrs.step()});
        }
        if (kind == WindowTransformKind.FOLD2D) {
            Fold2dAttrs attrs = (Fold2dAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.FOLD2D,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                    foldWindowWords(outputDescriptor, windowWords(attrs.window(), null)));
        }
        if (kind == WindowTransformKind.FOLD3D) {
            Fold3dAttrs attrs = (Fold3dAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.FOLD3D,
                    inputs,
                    new int[] {output},
                    MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                    foldWindowWords(outputDescriptor, windowWords(attrs.window(), null)));
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
        if (kind == AxisScatterKind.SCATTER_ADD) {
            IndexAxisAttrs attrs = (IndexAxisAttrs) operation.attrs();
            return MetalMpsGraphProgram.Node.scatterAdd(
                    inputs[0], inputs[1], inputs[2], output, attrs.axis());
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
                case PROD -> MetalMpsGraphProgram.NodeKind.PROD;
                case ALL -> MetalMpsGraphProgram.NodeKind.ALL;
                case ANY -> MetalMpsGraphProgram.NodeKind.ANY;
                case LOG_SUM_EXP -> MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP;
                case VARIANCE -> MetalMpsGraphProgram.NodeKind.VARIANCE;
                case STANDARD_DEVIATION -> MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION;
                case L1_NORM -> MetalMpsGraphProgram.NodeKind.L1_NORM;
                case L2_NORM -> MetalMpsGraphProgram.NodeKind.L2_NORM;
                default -> throw new IllegalArgumentException(
                        "unsupported Metal reduction: " + reduction);
            };
            if (operation.attrs() instanceof StatisticalReductionAttrs attrs) {
                return MetalMpsGraphProgram.Node.statisticalReduction(
                        nodeKind,
                        inputs[0],
                        output,
                        attrs.axes(),
                        attrs.keepDimensions(),
                        attrs.correction());
            }
            if (operation.attrs()
                    == io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE) {
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
        long[] words = new long[2 + target.length * 2];
        words[0] = target.length;
        System.arraycopy(target, 0, words, 1, target.length);
        words[1 + target.length] = prefix.length;
        System.arraycopy(prefix, 0, words, 2 + target.length, prefix.length);
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

    private static long[] foldWindowWords(
            TensorDescriptor outputDescriptor, long[] windowWords) {
        long[] shape = outputDescriptor.shape().toLongArray();
        long[] words = new long[1 + shape.length + windowWords.length];
        words[0] = shape.length;
        System.arraycopy(shape, 0, words, 1, shape.length);
        System.arraycopy(windowWords, 0, words, 1 + shape.length, windowWords.length);
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

    private static long byteSize(
            TensorDescriptor descriptor, MetalMpsGraphProgram.ValueState state) {
        Objects.requireNonNull(state, "state");
        long elements;
        if (state == MetalMpsGraphProgram.ValueState.MATERIALIZED_LAYOUT) {
            if (!MetalCapabilityProvider.supportedStorageLayout(descriptor, 0)) {
                throw new IllegalArgumentException(
                        "Metal storage-layout value has unsupported geometry");
            }
            elements = descriptor.layout().orElseThrow().referencedElementSpan();
        } else {
            elements = 1L;
            for (long dimension : descriptor.shape().toLongArray()) {
                if (dimension <= 0L) {
                    throw new IllegalArgumentException("Metal dimensions must be positive");
                }
                elements = Math.multiplyExact(elements, dimension);
            }
        }
        return Math.multiplyExact(elements, descriptor.dataType().byteWidth());
    }
}
