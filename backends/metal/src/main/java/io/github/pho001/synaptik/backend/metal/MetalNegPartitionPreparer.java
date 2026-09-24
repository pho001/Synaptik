package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
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
 * Analyzes and lowers one complete maximal Metal-owned elementwise partition.
 *
 * <p>The deterministic analysis assigns stable native value indices, retains every node kind and
 * ordered operand, and derives unique feeds and targets before selecting a closed private route.
 * Analysis freshly regenerates the complete candidate batch; an absent decision preserves the
 * singleton-NEG heuristic, while a present decision must authenticate against current schema,
 * workload, session target, and candidate identity. The selected route is then fixed before exact
 * declarations. Analysis allocates no physical resource and never changes partition ownership or
 * capability.</p>
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
        int nodeCount = context.nodes().size();
        var programNodes = new ArrayList<MetalMpsGraphProgram.Node>(nodeCount);
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            var node = context.nodes().get(nodeIndex);
            var inputDescriptors = new ArrayList<TensorDescriptor>(node.inputs().size());
            for (ValueId inputId : node.inputs()) {
                GraphValue inputValue = graphValues.get(inputId);
                if (inputValue == null) {
                    throw new IllegalArgumentException(
                            "partition value is missing: " + inputId);
                }
                inputDescriptors.add(inputValue.descriptor());
            }
            if (node.outputs().size() != 1) {
                throw new IllegalArgumentException(
                        "Metal elementwise partition requires one output per node");
            }
            ValueId outputId = node.outputs().getFirst();
            GraphValue outputValue = graphValues.get(outputId);
            if (outputValue == null) {
                throw new IllegalArgumentException(
                        "partition value is missing: " + outputId);
            }
            if (!MetalCapabilityProvider.supportsOccurrence(
                    node.operation(), inputDescriptors, List.of(outputValue.descriptor()))) {
                throw new IllegalArgumentException(
                        "Metal elementwise occurrence is outside the capability domain");
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
            if (valueIndexes.containsKey(outputId)) {
                throw new IllegalArgumentException(
                        "Metal elementwise output must be produced exactly once in topological order");
            }
            int outputIndex = index(
                    outputId, graphValues, valueIndexes, valueIds, descriptors);
            programNodes.add(lower(
                    node.operation(), inputIndices, outputIndex));
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

        var feeds = new ArrayList<ValueId>();
        for (var node : context.partitionDag().nodes()) {
            for (ValueId input : node.inputs()) {
                if (context.partitionDag().producer(input).isEmpty() && !feeds.contains(input)) {
                    feeds.add(input);
                }
            }
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
        if (feeds.isEmpty() || targets.isEmpty()) {
            throw new IllegalArgumentException(
                    "Metal elementwise partition requires at least one feed and one target");
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
                context.partition(), context.partitionDag(), deviceContext,
                route,
                valueIds, descriptors, ranks, dimensions, graphProgram,
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
                        context.partition(), context.partitionDag(), deviceContext,
                        route,
                        valueIds, descriptors, ranks, dimensions, graphProgram,
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
                    "Metal elementwise dimensions must be positive");
            elements = Math.multiplyExact(elements, dimension);
        }
        return Math.multiplyExact(elements, Float.BYTES);
    }
}
