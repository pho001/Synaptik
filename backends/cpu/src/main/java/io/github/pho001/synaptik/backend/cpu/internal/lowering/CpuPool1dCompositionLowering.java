package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Recognizes only the visible NCW expand-Pool2d-squeeze Pool1d composition. */
public final class CpuPool1dCompositionLowering {
    private final CpuPool2dLowering pool2d = new CpuPool2dLowering();

    /** Creates a stateless exact-topology recognizer. */
    public CpuPool1dCompositionLowering() {}

    /**
     * Validates and lowers the exact three-node Pool1d composition.
     *
     * @param context complete non-null three-node CPU projection
     * @return one schema-55 Pool2d unit whose rank edits remain virtual
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if topology, descriptors, attributes, or memory facts differ
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 3)
            throw new IllegalArgumentException("CPU Pool1d composition requires exactly three nodes");
        CompiledNode expand = context.nodes().get(0), pool = context.nodes().get(1),
                squeeze = context.nodes().get(2);
        if (expand.operation().kind() != AxisTransformKind.EXPAND_DIMS
                || !(expand.operation().attrs() instanceof AxisTransformAttrs ea) || ea.axis() != 2
                || expand.inputs().size() != 1 || expand.outputs().size() != 1
                || pool.inputs().size() != 1 || pool.outputs().size() != 1
                || !pool.inputs().getFirst().equals(expand.outputs().getFirst())
                || squeeze.operation().kind() != AxisTransformKind.SQUEEZE
                || !(squeeze.operation().attrs() instanceof AxisTransformAttrs sa) || sa.axis() != 2
                || squeeze.inputs().size() != 1 || squeeze.outputs().size() != 1
                || !squeeze.inputs().getFirst().equals(pool.outputs().getFirst()))
            throw new IllegalArgumentException("partition is not the exact visible Pool1d composition");
        Object attrs = pool.operation().attrs();
        boolean exactHeight = pool.operation().kind() == Pool2dKind.MAX_POOL2D
                        && attrs instanceof MaxPool2dAttrs a && height(a.kernelHeight(),
                                a.strideHeight(), a.paddingHeight(), a.dilationHeight())
                || pool.operation().kind() == Pool2dKind.AVERAGE_POOL2D
                        && attrs instanceof AveragePool2dAttrs average && height(average.kernelHeight(),
                                average.strideHeight(), average.paddingHeight(), average.dilationHeight());
        if (!exactHeight) throw new IllegalArgumentException("Pool1d height geometry disagrees");

        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(value -> memory.put(value.valueId(), value));
        ValueId inputId = expand.inputs().getFirst();
        ValueId expandedId = expand.outputs().getFirst();
        ValueId pooledId = pool.outputs().getFirst();
        ValueId outputId = squeeze.outputs().getFirst();
        requirePrivate(memory.get(expandedId), context);
        requirePrivate(memory.get(pooledId), context);
        GraphValue input = require(values, inputId), expanded = require(values, expandedId),
                pooled = require(values, pooledId), output = require(values, outputId);
        var capabilities = new CpuCapabilityProvider();
        if (!capabilities.supports(new OperationCapabilityQuery(expand.operation(),
                        List.of(input.descriptor()), List.of(expanded.descriptor())))
                || !capabilities.supports(new OperationCapabilityQuery(squeeze.operation(),
                        List.of(pooled.descriptor()), List.of(output.descriptor()))))
            throw new IllegalArgumentException("Pool1d affine components are not independently supported");
        requireExternal(input); requireExternal(output);
        requireExpanded(input, expanded); requireSqueezed(output, pooled);

        CompiledNode syntheticNode = new CompiledNode(pool.id(),
                new Operation(pool.operation().kind(), pool.operation().attrs()),
                List.of(inputId), List.of(outputId));
        PlannedPartition syntheticPartition = new PlannedPartition(
                CpuCapabilityProvider.CPU_BACKEND_ID, List.of(pool.id()));
        GraphValue syntheticInput = expanded(inputId, input);
        GraphValue syntheticOutput = expanded(outputId, output);
        List<GraphValue> syntheticValues = List.of(syntheticInput, syntheticOutput);
        List<LogicalMemoryRequirement> syntheticMemory = new ArrayList<>();
        syntheticMemory.add(new LogicalMemoryRequirement(inputId, syntheticInput.descriptor(),
                Optional.empty(), List.of(syntheticPartition), false));
        syntheticMemory.add(new LogicalMemoryRequirement(outputId, syntheticOutput.descriptor(),
                Optional.of(syntheticPartition), List.of(), true));
        var synthetic = new PrepareContext<>(syntheticPartition, List.of(syntheticNode),
                syntheticValues, syntheticMemory, Map.of(), context.backendInputs());
        var lowered = pool2d.lower(synthetic);
        return new CpuPartitionLowering.LoweredPartition(lowered.portableKernelIr(),
                lowered.boundaryValues(), lowered.accessBindings(), lowered.referencedElementSpans(),
                lowered.boundaryDataTypes(), List.of(expandedId, pooledId), lowered.extents(),
                lowered.elementCount(),
                "legal: exact visible NCW Pool1d composition with virtual singleton views",
                lowered.affineAddressPairs(), lowered.movementGeometry(),
                lowered.indexingGeometry(), lowered.scatterGeometry(), lowered.foldGeometry(),
                lowered.orderingGeometry(), lowered.randomGeometry(), lowered.scanGeometry(),
                lowered.aggregateGeometry(), lowered.argExtremaGeometry(),
                lowered.maskedReductionGeometry(), lowered.advancedReductionGeometry(),
                lowered.softmaxGeometry(), lowered.trailingNormalizationGeometry(),
                lowered.batchNormInferenceGeometry(), lowered.batchNormTrainingGeometry(),
                lowered.conv2dGeometry(), lowered.conv3dGeometry(), lowered.matmulFacts(),
                lowered.pool2dGeometry(), lowered.pool3dGeometry());
    }

    private static boolean height(long kernel, long stride, long padding, long dilation) {
        return kernel == 1 && stride == 1 && padding == 0 && dilation == 1;
    }

    private static void requirePrivate(LogicalMemoryRequirement requirement,
            PrepareContext<?> context) {
        if (requirement == null || requirement.graphOutput()
                || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())
                || !requirement.consumerPartitions().equals(List.of(context.partition())))
            throw new IllegalArgumentException("Pool1d intermediate must be private and single-use");
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("Pool1d value is not projected: " + id);
        return value;
    }

    private static void requireExternal(GraphValue value) {
        var descriptor = value.descriptor();
        if (descriptor.shape().rank() != 3 || !descriptor.shape().isFullyStatic()
                || descriptor.layout().isEmpty() || descriptor.layout().orElseThrow().storageOffset() < 0
                || Arrays.stream(descriptor.layout().orElseThrow().strides()).anyMatch(x -> x < 0))
            throw new IllegalArgumentException("Pool1d external boundary must be static resolved NCW");
    }

    private static void requireExpanded(GraphValue sourceValue, GraphValue viewValue) {
        var source = sourceValue.descriptor(); var view = viewValue.descriptor();
        if (!Arrays.equals(insert(source.shape().toLongArray()), view.shape().toLongArray())
                || view.dataType() != source.dataType()
                || view.requiresGrad() != source.requiresGrad() || view.layout().isEmpty())
            throw new IllegalArgumentException("Pool1d expanded descriptor disagrees");
        requireViewLayout(source.layout().orElseThrow(), view.layout().orElseThrow(),
                source.shape().toLongArray()[2]);
    }

    private static void requireSqueezed(GraphValue resultValue, GraphValue rankFourValue) {
        var result = resultValue.descriptor(); var rankFour = rankFourValue.descriptor();
        if (!Arrays.equals(insert(result.shape().toLongArray()), rankFour.shape().toLongArray())
                || rankFour.dataType() != result.dataType()
                || rankFour.requiresGrad() != result.requiresGrad() || rankFour.layout().isEmpty())
            throw new IllegalArgumentException("Pool1d squeezed descriptor disagrees");
        requireViewLayout(result.layout().orElseThrow(), rankFour.layout().orElseThrow(),
                result.shape().toLongArray()[2]);
    }

    private static void requireViewLayout(LayoutDescriptor source, LayoutDescriptor view,
            long width) {
        long[] s = source.strides(), v = view.strides();
        if (source.storageOffset() != view.storageOffset()
                || source.referencedElementSpan() != view.referencedElementSpan()
                || s.length != 3 || v.length != 4 || v[0] != s[0] || v[1] != s[1]
                || v[2] != Math.multiplyExact(s[2], width)
                || v[3] != s[2])
            throw new IllegalArgumentException("Pool1d singleton layout is not the exact affine view");
    }

    private static long[] insert(long[] source) {
        return new long[] {source[0], source[1], 1, source[2]};
    }

    private static GraphValue expanded(ValueId id, GraphValue source) {
        long[] shape = insert(source.descriptor().shape().toLongArray());
        LayoutDescriptor layout = source.descriptor().layout().orElseThrow();
        long[] old = layout.strides(); long[] sourceShape = source.descriptor().shape().toLongArray();
        long[] strides = {old[0], old[1], Math.multiplyExact(old[2], sourceShape[2]), old[2]};
        Shape expanded = Shape.of(shape);
        return new GraphValue(id, new TensorDescriptor(source.descriptor().dataType(), expanded,
                Optional.of(LayoutDescriptor.of(expanded, strides, layout.storageOffset(), true)),
                source.descriptor().requiresGrad()));
    }
}
