package graph.execution;

import backend.cpu.kernels.CpuNodeWorkspace;
import backend.memory.TensorResidencyState;
import graph.CompiledNode;
import backend.cpu.plan.CpuPreparedInput;
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
 * <p>Prepared programs keep immutable compile/prepare metadata. Every execute call materializes its own
 * runtime tensor bindings and workspaces here so runs do not share mutable graph state.
 */
public final class ExecutionState {
    private record PreparedInputKey(int nodeId, int inputIndex) {
    }

    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId;
    private final Map<PreparedInputKey, Tensor> preparedInputTensorByKey;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;
    private final Map<Integer, TensorResidencyState> residencyByNodeId;

    private ExecutionState(
            Map<Integer, Tensor> runtimeTensorByNodeId,
            Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId,
            Map<PreparedInputKey, Tensor> preparedInputTensorByKey,
            Map<Tensor, Integer> runtimeNodeIdByTensor,
            Map<Integer, TensorResidencyState> residencyByNodeId
    ) {
        this.runtimeTensorByNodeId = Map.copyOf(runtimeTensorByNodeId);
        this.cpuWorkspaceByNodeId = Map.copyOf(cpuWorkspaceByNodeId);
        this.preparedInputTensorByKey = Map.copyOf(preparedInputTensorByKey);
        this.runtimeNodeIdByTensor = Map.copyOf(runtimeNodeIdByTensor);
        this.residencyByNodeId = Map.copyOf(residencyByNodeId);
    }

    /**
     * Creates per-run runtime tensors, prepared input buffers, and CPU workspaces.
     *
     * @param compiledNodes compiled node snapshots in graph order
     * @param metadataIndex prepared execution metadata keyed by node id
     * @param forwardBoundaryNodeId last forward node id, used to decide leaf aliasing versus copying
     * @return mutable execution state for one run
     */
    public static ExecutionState create(
            List<CompiledNode> compiledNodes,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            int forwardBoundaryNodeId
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(metadataIndex, "metadataIndex cannot be null");

        Map<Integer, Tensor> runtimeTensors = new HashMap<>(compiledNodes.size());
        Map<Tensor, Integer> runtimeNodeIds = new IdentityHashMap<>(compiledNodes.size());
        Map<Integer, TensorResidencyState> residency = new HashMap<>(compiledNodes.size());
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
                residency.put(node.id(), TensorResidencyState.cpuArrayCurrent("leaf runtime binding"));
            } else {
                residency.put(node.id(), TensorResidencyState.cpuArrayStale("runtime tensor allocated"));
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
        Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate = new IdentityHashMap<>();
        Map<PreparedInputKey, Tensor> preparedInputs = new HashMap<>();
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            CpuNodeWorkspace workspace = entry.getValue().cpuWorkspace();
            if (workspace != null) {
                CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
                workspaces.put(entry.getKey(), runtimeWorkspace);
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
        return new ExecutionState(runtimeTensors, workspaces, preparedInputs, runtimeNodeIds, residency);
    }

    /**
     * Returns the runtime tensor for a compiled node.
     *
     * @param nodeId compiled node id
     * @return runtime tensor
     */
    public Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    /**
     * Returns the CPU workspace fork for a compiled node.
     *
     * @param nodeId compiled node id
     * @return CPU workspace, or {@code null} when the node does not use one
     */
    public CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        return cpuWorkspaceByNodeId.get(nodeId);
    }

    /**
     * Returns a prepared runtime input tensor for a node input.
     *
     * @param nodeId compiled node id
     * @param inputIndex input index
     * @return prepared runtime tensor
     */
    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(new PreparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }

    /**
     * Looks up the compiled node id for a runtime tensor.
     *
     * @param tensor runtime tensor
     * @return node id, or {@code null} when the tensor is unknown
     */
    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }

    /**
     * Returns runtime residency state for a compiled node.
     *
     * @param nodeId compiled node id
     * @return mutable residency state for the runtime tensor
     */
    public TensorResidencyState residencyForNodeId(int nodeId) {
        TensorResidencyState state = residencyByNodeId.get(nodeId);
        if (state == null) {
            throw new IllegalStateException("Missing runtime residency state for nodeId=" + nodeId);
        }
        return state;
    }

    /**
     * Marks a node output as current in CPU array storage.
     *
     * @param nodeId compiled node id
     * @param reason diagnostic transition reason
     */
    public void markCpuCurrent(int nodeId, String reason) {
        residencyForNodeId(nodeId).markCpuCurrent(reason);
    }
}
