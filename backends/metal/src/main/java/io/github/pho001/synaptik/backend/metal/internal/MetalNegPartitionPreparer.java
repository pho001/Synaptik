package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
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
 * Analyzes and lowers one complete maximal Metal-owned unary-NEG partition.
 *
 * <p>The deterministic analysis assigns stable native value indices, derives unique feeds and
 * targets from the partition DAG and logical requirements, and declares feed buffers followed by
 * target buffers and one address workspace. It allocates no physical resource.</p>
 */
final class MetalNegPartitionPreparer implements BackendPartitionPreparer<
        MetalNegAnalysisInputs, MetalNegPreparationPlan> {
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
        int[] nodeInputs = new int[nodeCount];
        int[] nodeOutputs = new int[nodeCount];
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            var node = context.nodes().get(nodeIndex);
            if (node.operation().kind() != UnaryElementwiseKind.NEG
                    || node.operation().attrs() != NoOperationAttrs.INSTANCE
                    || node.inputs().size() != 1 || node.outputs().size() != 1) {
                throw new IllegalArgumentException(
                        "Metal NEG partition contains an unsupported operation occurrence");
            }
            nodeInputs[nodeIndex] = index(node.inputs().getFirst(), graphValues,
                    valueIndexes, valueIds, descriptors);
            nodeOutputs[nodeIndex] = index(node.outputs().getFirst(), graphValues,
                    valueIndexes, valueIds, descriptors);
            TensorDescriptor input = descriptors.get(nodeInputs[nodeIndex]);
            TensorDescriptor output = descriptors.get(nodeOutputs[nodeIndex]);
            if (!eligible(input) || !eligible(output)
                    || !input.shape().equals(output.shape())
                    || input.requiresGrad() != output.requiresGrad()) {
                throw new IllegalArgumentException(
                        "Metal NEG occurrence descriptors are outside the capability domain");
            }
        }

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
                    "Metal NEG partition requires at least one feed and one target");
        }
        var feedSplats = new ArrayList<Optional<ScalarValue>>(feeds.size());
        for (ValueId feed : feeds) {
            ScalarValue scalar = context.constants().get(feed);
            if (scalar != null && scalar.dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException("Metal NEG splat feed must be FLOAT32");
            }
            feedSplats.add(Optional.ofNullable(scalar));
        }
        for (ValueId constant : context.constants().keySet()) {
            if (!feeds.contains(constant)) {
                throw new IllegalArgumentException("Metal NEG constant must be a boundary feed");
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
        long workspaceBytes = Math.multiplyExact(
                Math.addExact((long) feeds.size(), targets.size()), Long.BYTES);
        var workspace = new PreparationResourceRequirement.Workspace(
                0L, workspaceBytes, Long.BYTES);

        var plan = new MetalNegPreparationPlan(
                context.partition(), context.partitionDag(), deviceContext,
                valueIds, descriptors, ranks, dimensions,
                nodeInputs, nodeOutputs, feeds, feedIndices, targets, targetIndices,
                declarations, feedSplats, workspace, feedBytes, targetBytes);
        var allDeclarations = new ArrayList<PreparationResourceRequirement>(declarations);
        allDeclarations.add(workspace);
        return new BackendPartitionAnalysis<>(context.partition(), plan, allDeclarations);
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

    private static boolean eligible(TensorDescriptor descriptor) {
        if (descriptor.dataType() != DataType.FLOAT32
                || !descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() < 1 || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) return false;
        try {
            if (byteSize(descriptor) <= 0L) return false;
        } catch (ArithmeticException overflow) {
            return false;
        }
        var layout = descriptor.layout().orElseThrow();
        return layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                && !layout.isView() && layout.storageOffset() == 0L;
    }

    private static long byteSize(TensorDescriptor descriptor) {
        long elements = 1L;
        for (long dimension : descriptor.shape().toLongArray()) {
            if (dimension <= 0L) throw new IllegalArgumentException(
                    "Metal NEG dimensions must be positive");
            elements = Math.multiplyExact(elements, dimension);
        }
        return Math.multiplyExact(elements, Float.BYTES);
    }
}
