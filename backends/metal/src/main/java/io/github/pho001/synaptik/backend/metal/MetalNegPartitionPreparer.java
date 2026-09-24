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
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
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
 * Under {@code STRICT_IEEE}, it walks explicit unavailable/canonical/affine-view states in node
 * order for the retained NEG/ABS, affine, and CONTIGUOUS domain. Under {@code ACCELERATOR}, it
 * accepts canonical ABS, binary arithmetic, the exact SUM/MEAN/SUM_TO_SHAPE reduction forms, and
 * positive static rank-two FLOAT32 MATMUL. An affine MATMUL operand is authenticated to the exact
 * earlier local rank-two {@code PERMUTE [1,0]} of a canonical source; that view may be consumed
 * only by local MATMUL and may not cross or become a partition boundary. MATMUL lowering retains
 * both ordered operands in a schema-eight wire-15 record. Reduction lowering retains the typed
 * form, ordered normalized axes (including empty), exact keep-dimensions flag, sum-to-Shape
 * target, and shape-derived term geometry. Every graph feed is canonical positive-rank FLOAT32.
 * Analysis freshly regenerates the complete candidate batch; an absent decision preserves the
 * singleton-NEG heuristic, while a present decision must authenticate against current schema,
 * workload, profile, session target, and candidate identity. The selected route is fixed before
 * exact declarations. Published affine views retain logical descriptors while declarations use
 * full dense represented-order byte geometry. Analysis allocates no physical resource and never
 * changes partition ownership or capability.</p>
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
            if (inputStates.size() != lowered.kind().inputCount()
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
            if (numericalProfile == NumericalProfile.ACCELERATOR
                    && lowered.kind() == MetalMpsGraphProgram.NodeKind.PERMUTE
                    && !exactLocalTranspose) {
                throw new IllegalArgumentException(
                        "Metal accelerator PERMUTE must be an exact local rank-two transpose");
            }
            localTranspose.put(outputId, exactLocalTranspose);
            states.put(outputId, lowered.kind().outputState());
        }
        if (numericalProfile == NumericalProfile.ACCELERATOR) {
            for (Map.Entry<ValueId, Boolean> entry : localTranspose.entrySet()) {
                if (!entry.getValue()) continue;
                var consumers = context.partitionDag().consumers(entry.getKey());
                if (consumers.isEmpty()
                        || consumers.stream().anyMatch(consumer ->
                                consumer.node().operation().kind() != MatmulKind.MATMUL)) {
                    throw new IllegalArgumentException(
                            "Metal local transpose must be consumed only by local MATMUL");
                }
            }
        }
        var graphProgram = new MetalMpsGraphProgram(programNodes);

        int[] ranks = new int[valueIds.size()];
        long[] dimensions = new long[Math.multiplyExact(valueIds.size(), 16)];
        for (int valueIndex = 0; valueIndex < valueIds.size(); valueIndex++) {
            long[] shape = descriptors.get(valueIndex).shape().toLongArray();
            ranks[valueIndex] = shape.length;
            System.arraycopy(shape, 0, dimensions, valueIndex * 16, shape.length);
        }
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
        if (numericalProfile == NumericalProfile.ACCELERATOR
                && targets.stream().anyMatch(
                        target -> Boolean.TRUE.equals(localTranspose.get(target)))) {
            throw new IllegalArgumentException(
                    "Metal local MATMUL transpose cannot be a partition boundary");
        }
        if (feeds.isEmpty() || targets.isEmpty()) {
            throw new IllegalArgumentException(
                    "Metal supported-operation partition requires at least one feed and one target");
        }
        var feedSplats = new ArrayList<Optional<ScalarValue>>(feeds.size());
        for (ValueId feed : feeds) {
            ScalarValue scalar = context.constants().get(feed);
            if (scalar != null && scalar.dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException("Metal splat feed must be FLOAT32");
            }
            feedSplats.add(Optional.ofNullable(scalar));
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
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(
                feeds.size() + targets.size());
        for (int index = 0; index < feeds.size(); index++) {
            declarations.add(new PreparationResourceRequirement.Buffer(
                    feeds.get(index), feedBytes[index], Float.BYTES));
        }
        for (int index = 0; index < targets.size(); index++) {
            declarations.add(new PreparationResourceRequirement.Buffer(
                    targets.get(index), targetBytes[index], Float.BYTES));
        }
        long singletonElements = feedBytes.length == 1 ? feedBytes[0] / Float.BYTES : 0L;
        MetalNegPreparationPlan.Route route = nodeCount == 1
                        && graphProgram.nodes().getFirst().kind()
                                == MetalMpsGraphProgram.NodeKind.NEG
                        && feeds.size() == 1
                        && targets.size() == 1
                        && singletonElements >= 1L
                        && singletonElements <= UINT32_MAX
                ? MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG
                : MetalNegPreparationPlan.Route.MPSGRAPH;
        Optional<PreparationResourceRequirement.Workspace> heuristicWorkspace = workspace(
                route, feeds.size(), targets.size());
        var heuristicPlan = new MetalNegPreparationPlan(
                context.numericalProfile(),
                context.partition(), context.partitionDag(), deviceContext,
                route,
                valueIds, descriptors, valueStates, ranks, dimensions, graphProgram,
                feeds, feedIndices, targets, targetIndices, declarations, feedSplats,
                heuristicWorkspace, feedBytes, targetBytes);
        MetalNegTuningBatch freshBatch = new MetalNegRouteCandidateGenerator()
                .generate(context, heuristicPlan, MetalNegTuningBatch.Candidate.values().length);
        var suppliedHandoff = context.backendInputs().tuningHandoff();
        if (suppliedHandoff.isPresent()
                && suppliedHandoff.orElseThrow().selectedDecision().isPresent()) {
            var handoff = suppliedHandoff.orElseThrow();
            var decision = handoff.selectedDecision().orElseThrow();
            if (handoff.partition() != context.partition()
                    || !handoff.candidateBatch().compatibility()
                            .equals(freshBatch.compatibility())
                    || handoff.candidateBatch().find(decision.selectedCandidate()).isEmpty()) {
                throw new IllegalArgumentException(
                        "Metal NEG tuning handoff is stale or foreign");
            }
            route = decision.match(freshBatch)
                    .orElseThrow(() -> new IllegalArgumentException(
                            "Metal NEG tuning decision is incompatible"))
                    .route();
        }

        Optional<PreparationResourceRequirement.Workspace> selectedWorkspace = workspace(
                route, feeds.size(), targets.size());
        var plan = route == heuristicPlan.route()
                ? heuristicPlan
                : new MetalNegPreparationPlan(
                        context.numericalProfile(),
                        context.partition(), context.partitionDag(), deviceContext,
                        route,
                        valueIds, descriptors, valueStates, ranks, dimensions, graphProgram,
                        feeds, feedIndices, targets, targetIndices, declarations, feedSplats,
                        selectedWorkspace, feedBytes, targetBytes);
        var allDeclarations = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.addressWorkspace().ifPresent(allDeclarations::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, allDeclarations);
    }

    private static Optional<PreparationResourceRequirement.Workspace> workspace(
            MetalNegPreparationPlan.Route route, int feedCount, int targetCount) {
        if (route != MetalNegPreparationPlan.Route.MPSGRAPH) return Optional.empty();
        long workspaceBytes = Math.multiplyExact(
                Math.addExact((long) feedCount, targetCount), Long.BYTES);
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
        return descriptor.dataType() == DataType.FLOAT32
                && descriptor.layout().isPresent()
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
        if (kind == UnaryElementwiseKind.NEG) {
            return MetalMpsGraphProgram.Node.neg(inputs[0], output);
        }
        if (kind == UnaryElementwiseKind.ABS) {
            return MetalMpsGraphProgram.Node.abs(inputs[0], output);
        }
        if (kind instanceof BinaryArithmeticKind binary) {
            MetalMpsGraphProgram.NodeKind nodeKind = switch (binary) {
                case ADD -> MetalMpsGraphProgram.NodeKind.ADD;
                case SUB -> MetalMpsGraphProgram.NodeKind.SUB;
                case MUL -> MetalMpsGraphProgram.NodeKind.MUL;
                case DIV -> MetalMpsGraphProgram.NodeKind.DIV;
                default -> throw new IllegalArgumentException(
                        "unsupported Metal binary operation: " + binary);
            };
            return MetalMpsGraphProgram.Node.binary(
                    nodeKind, inputs[0], inputs[1], output);
        }
        if (kind == MatmulKind.MATMUL) {
            return MetalMpsGraphProgram.Node.matmul(inputs[0], inputs[1], output);
        }
        if (kind instanceof AggregateReductionKind reduction) {
            MetalMpsGraphProgram.NodeKind nodeKind = reduction == AggregateReductionKind.SUM
                    ? MetalMpsGraphProgram.NodeKind.SUM
                    : MetalMpsGraphProgram.NodeKind.MEAN;
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
        AxisTransformAttrs attrs = (AxisTransformAttrs) operation.attrs();
        MetalMpsGraphProgram.NodeKind nodeKind = kind == AxisTransformKind.EXPAND_DIMS
                ? MetalMpsGraphProgram.NodeKind.EXPAND_DIMS
                : MetalMpsGraphProgram.NodeKind.SQUEEZE;
        return MetalMpsGraphProgram.Node.axis(
                nodeKind, inputs[0], output, attrs.axis());
    }

    private static long byteSize(TensorDescriptor descriptor) {
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) throw new IllegalArgumentException(
                    "Metal dimensions must be positive");
            elements = Math.multiplyExact(elements, dimension);
        }
        return Math.multiplyExact(elements, Float.BYTES);
    }
}
