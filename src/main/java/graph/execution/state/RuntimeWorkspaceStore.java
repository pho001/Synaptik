package graph.execution.state;

import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import tensor.Tensor;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Run-scoped backend workspaces and prepared input buffers.
 */
final class RuntimeWorkspaceStore implements PreparedRuntimeStateAllocator {
    private final Map<Integer, Object> workspaceByNodeId;
    private final Map<Long, Tensor> preparedInputTensorByKey;
    private final Map<Object, Object> runtimeWorkspaceByTemplate;

    static RuntimeWorkspaceStore create(Map<Integer, CompiledNodeExecutionMetadata> metadataIndex) {
        RuntimeWorkspaceStore store = new RuntimeWorkspaceStore(new HashMap<>(), new HashMap<>(), new IdentityHashMap<>());
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            if (entry.getValue() != null && entry.getValue().artifact() != null) {
                entry.getValue().artifact().allocateRuntimeState(entry.getKey(), store);
            }
        }
        return store.freeze();
    }

    RuntimeWorkspaceStore(
            Map<Integer, Object> workspaceByNodeId,
            Map<Long, Tensor> preparedInputTensorByKey,
            Map<Object, Object> runtimeWorkspaceByTemplate
    ) {
        this.workspaceByNodeId = workspaceByNodeId;
        this.preparedInputTensorByKey = preparedInputTensorByKey;
        this.runtimeWorkspaceByTemplate = runtimeWorkspaceByTemplate;
    }

    @Override
    public Object forkWorkspace(Object template, Supplier<?> forkFactory) {
        if (template == null || forkFactory == null) {
            return null;
        }
        return runtimeWorkspaceByTemplate.computeIfAbsent(template, ignored -> forkFactory.get());
    }

    @Override
    public void putWorkspace(int nodeId, Object workspace) {
        if (workspace != null) {
            workspaceByNodeId.put(nodeId, workspace);
        }
    }

    @Override
    public void putPreparedInputTensor(int nodeId, int inputIndex, Tensor tensor) {
        preparedInputTensorByKey.put(preparedInputKey(nodeId, inputIndex), tensor);
    }

    static long preparedInputKey(int nodeId, int inputIndex) {
        return ((long) nodeId << Integer.SIZE) ^ (inputIndex & 0xffffffffL);
    }

    private RuntimeWorkspaceStore freeze() {
        return new RuntimeWorkspaceStore(
                Map.copyOf(workspaceByNodeId),
                Map.copyOf(preparedInputTensorByKey),
                Map.of()
        );
    }

    Object workspaceForNodeId(int nodeId) {
        return workspaceByNodeId.get(nodeId);
    }

    Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(preparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }
}
