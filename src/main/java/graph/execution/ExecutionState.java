package graph.execution;

import backend.kernels.cpu.CpuNodeWorkspace;
import graph.CompiledNode;
import backend.CpuPreparedInput;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Per-run mutable execution state.
 *
 * Prepared programs keep immutable compile/prepare metadata. Every execute call materializes its own
 * runtime tensor bindings and workspaces here so runs do not share mutable graph state.
 */
public final class ExecutionState {
    private record PreparedInputKey(int nodeId, int inputIndex) {
    }

    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId;
    private final Map<PreparedInputKey, Tensor> preparedInputTensorByKey;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;

    private ExecutionState(
            Map<Integer, Tensor> runtimeTensorByNodeId,
            Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId,
            Map<PreparedInputKey, Tensor> preparedInputTensorByKey,
            Map<Tensor, Integer> runtimeNodeIdByTensor
    ) {
        this.runtimeTensorByNodeId = Map.copyOf(runtimeTensorByNodeId);
        this.cpuWorkspaceByNodeId = Map.copyOf(cpuWorkspaceByNodeId);
        this.preparedInputTensorByKey = Map.copyOf(preparedInputTensorByKey);
        this.runtimeNodeIdByTensor = Map.copyOf(runtimeNodeIdByTensor);
    }

    public static ExecutionState create(
            List<CompiledNode> compiledNodes,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            int forwardBoundaryNodeId
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(metadataIndex, "metadataIndex cannot be null");

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
            runtimeTensor.setRequiresGrad(node.semanticTensor().getRequiresGrad());
            if (node.leaf()) {
                if (node.id() <= forwardBoundaryNodeId) {
                    TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, node.sourceTensor());
                } else {
                    runtimeTensor.copyDataFrom(node.sourceTensor());
                }
            }
            runtimeTensors.put(node.id(), runtimeTensor);
            runtimeNodeIds.put(runtimeTensor, node.id());
        }
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

        Map<Integer, CpuNodeWorkspace> workspaces = new HashMap<>();
        Map<PreparedInputKey, Tensor> preparedInputs = new HashMap<>();
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            CpuNodeWorkspace workspace = entry.getValue().cpuWorkspace();
            if (workspace != null) {
                workspaces.put(entry.getKey(), workspace.fork());
            }
            if (entry.getValue().cpuPlan() != null) {
                for (CpuPreparedInput preparedInput : entry.getValue().cpuPlan().layoutPlan().preparedInputs()) {
                    Tensor template = preparedInput.runtimeTensor();
                    Tensor runtimePrepared = new Tensor(
                            template.getShapeUnsafe().clone(),
                            template.getStridesUnsafe().clone(),
                            template.getStorageOffsetUnsafe(),
                            null,
                            null,
                            template.getLabel(),
                            template.getDataType()
                    );
                    runtimePrepared.setRequiresGrad(template.getRequiresGrad());
                    preparedInputs.put(new PreparedInputKey(entry.getKey(), preparedInput.inputIndex()), runtimePrepared);
                }
            }
        }
        return new ExecutionState(runtimeTensors, workspaces, preparedInputs, runtimeNodeIds);
    }

    public Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    public CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        return cpuWorkspaceByNodeId.get(nodeId);
    }

    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(new PreparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }

    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }
}
