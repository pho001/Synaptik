package runtime.state;

import runtime.residency.TensorResidencyState;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.publication.PublicationPlan;
import graph.model.CompiledNode;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Run-scoped runtime tensor identity and node-id lookup.
 */
public final class RuntimeTensorStore {
    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;

    public static RuntimeTensorStore create(
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            int forwardBoundaryNodeId,
            Map<Integer, TensorResidencyState> residencyByNodeId,
            PublicationPlan publicationPlan
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        Objects.requireNonNull(residencyByNodeId, "residencyByNodeId cannot be null");
        Objects.requireNonNull(publicationPlan, "publicationPlan cannot be null");
        Map<Integer, PublicationPlan.RuntimeInputBinding> runtimeInputs = publicationPlan.runtimeInputsByNodeId();
        Map<Integer, Tensor> runtimeTensors = new HashMap<>(compiledNodes.size());
        Map<Tensor, Integer> runtimeNodeIds = new IdentityHashMap<>(compiledNodes.size());
        for (CompiledNode node : compiledNodes) {
            Tensor runtimeTensor = new Tensor(
                    node.shape(),
                    node.strides(),
                    node.storageOffset(),
                    null,
                    node.operation(),
                    node.label(),
                    node.dataType()
            );
            CompiledTensorDescriptor descriptor = descriptorIndex.byNodeId(node.id());
            runtimeTensor.setRequiresGrad(descriptor.requiresGrad());
            runtimeTensor.setTrainableParameter(descriptor.trainableParameter());
            if (node.leaf()) {
                PublicationPlan.RuntimeInputBinding binding = runtimeInputs.get(node.id());
                if (binding == null) {
                    throw new IllegalStateException("Missing runtime input binding for leaf nodeId=" + node.id());
                }
                if (binding.kind() == PublicationPlan.RuntimeInputBindingKind.FORWARD_LEAF_ALIAS) {
                    TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, binding.sourceTensor());
                } else {
                    runtimeTensor.copyDataFrom(binding.sourceTensor());
                }
                residencyByNodeId.put(node.id(), TensorResidencyState.cpuArrayCurrent("leaf runtime binding"));
            } else {
                residencyByNodeId.put(node.id(), TensorResidencyState.cpuArrayStale("runtime tensor allocated"));
            }
            runtimeTensors.put(node.id(), runtimeTensor);
            runtimeNodeIds.put(runtimeTensor, node.id());
        }
        bindRuntimeInputs(compiledNodes, runtimeTensors);
        bindCpuAliasViews(compiledNodes, runtimeTensors, residencyByNodeId);
        return new RuntimeTensorStore(runtimeTensors, runtimeNodeIds);
    }

    public RuntimeTensorStore(
            Map<Integer, Tensor> runtimeTensorByNodeId,
            Map<Tensor, Integer> runtimeNodeIdByTensor
    ) {
        this.runtimeTensorByNodeId = Map.copyOf(runtimeTensorByNodeId);
        this.runtimeNodeIdByTensor = new IdentityHashMap<>(runtimeNodeIdByTensor);
    }

    private static void bindRuntimeInputs(List<CompiledNode> compiledNodes, Map<Integer, Tensor> runtimeTensors) {
        for (CompiledNode node : compiledNodes) {
            if (node.inputIds().isEmpty()) {
                continue;
            }
            java.util.ArrayList<Tensor> runtimeInputs = new java.util.ArrayList<>(node.inputIds().size());
            for (int inputId : node.inputIds()) {
                Tensor input = runtimeTensors.get(inputId);
                if (input == null) {
                    throw new IllegalStateException("Missing runtime input tensor for nodeId=" + node.id() + ", inputId=" + inputId);
                }
                runtimeInputs.add(input);
            }
            TensorInternalAccess.setPrevTensors(runtimeTensors.get(node.id()), runtimeInputs);
        }
    }

    private static void bindCpuAliasViews(
            List<CompiledNode> compiledNodes,
            Map<Integer, Tensor> runtimeTensors,
            Map<Integer, TensorResidencyState> residencyByNodeId
    ) {
        for (CompiledNode node : compiledNodes) {
            if (node.storageOwnerId() == node.id()) {
                continue;
            }
            int sourceNodeId = node.storageOwnerId();
            TensorResidencyState sourceResidency = residencyByNodeId.get(sourceNodeId);
            if (sourceResidency != null && sourceResidency.cpuCurrent()) {
                TensorInternalAccess.aliasRuntimeFrom(runtimeTensors.get(node.id()), runtimeTensors.get(sourceNodeId));
                residencyByNodeId.get(node.id()).markCpuCurrent("alias view runtime binding");
            }
        }
    }

    public Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }

    public long logicalByteLength(int nodeId) {
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        return (long) tensor.getFlatDataSize() * elementByteSize(tensor.getDataType());
    }

    private static int elementByteSize(DataType dataType) {
        if (dataType == null) {
            return 0;
        }
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
        };
    }
}
