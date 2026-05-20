package graph.execution.state;

import backend.memory.TensorResidencyState;
import graph.AliasViewPolicy;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.IdentityHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Run-scoped runtime tensor identity and node-id lookup.
 */
final class RuntimeTensorStore {
    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;

    static RuntimeTensorStore create(
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            int forwardBoundaryNodeId,
            Map<Integer, TensorResidencyState> residencyByNodeId
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        Objects.requireNonNull(residencyByNodeId, "residencyByNodeId cannot be null");
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
                if (node.id() <= forwardBoundaryNodeId) {
                    TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, node.publicationTensor());
                } else {
                    runtimeTensor.copyDataFrom(node.publicationTensor());
                }
                residencyByNodeId.put(node.id(), TensorResidencyState.cpuArrayCurrent("leaf runtime binding"));
            } else {
                residencyByNodeId.put(node.id(), TensorResidencyState.cpuArrayStale("runtime tensor allocated"));
            }
            runtimeTensors.put(node.id(), runtimeTensor);
            runtimeNodeIds.put(runtimeTensor, node.id());
        }
        bindRuntimeInputs(compiledNodes, runtimeTensors);
        bindCpuAliasViews(compiledNodes, descriptorIndex, runtimeTensors, residencyByNodeId);
        return new RuntimeTensorStore(runtimeTensors, runtimeNodeIds);
    }

    RuntimeTensorStore(
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
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Integer, Tensor> runtimeTensors,
            Map<Integer, TensorResidencyState> residencyByNodeId
    ) {
        for (CompiledNode node : compiledNodes) {
            if (node.inputIds().isEmpty() || !isCpuAliasView(node, descriptorIndex)) {
                continue;
            }
            int sourceNodeId = node.inputIds().getFirst();
            TensorResidencyState sourceResidency = residencyByNodeId.get(sourceNodeId);
            if (sourceResidency != null && sourceResidency.cpuCurrent()) {
                TensorInternalAccess.aliasRuntimeFrom(runtimeTensors.get(node.id()), runtimeTensors.get(sourceNodeId));
                residencyByNodeId.get(node.id()).markCpuCurrent("alias view runtime binding");
            }
        }
    }

    private static boolean isCpuAliasView(CompiledNode node, CompiledTensorDescriptorIndex descriptorIndex) {
        if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
            return false;
        }
        return AliasViewPolicy.aliasesInput0AtRuntime(node, descriptorIndex);
    }

    Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }

    long logicalByteLength(int nodeId) {
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
