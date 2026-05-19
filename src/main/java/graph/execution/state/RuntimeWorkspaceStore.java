package graph.execution.state;

import backend.cpu.kernels.CpuNodeWorkspace;
import tensor.Tensor;

import java.util.Map;

/**
 * Run-scoped CPU workspaces and prepared input buffers.
 */
final class RuntimeWorkspaceStore {
    private final Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId;
    private final Map<Long, Tensor> preparedInputTensorByKey;

    RuntimeWorkspaceStore(
            Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId,
            Map<Long, Tensor> preparedInputTensorByKey
    ) {
        this.cpuWorkspaceByNodeId = Map.copyOf(cpuWorkspaceByNodeId);
        this.preparedInputTensorByKey = Map.copyOf(preparedInputTensorByKey);
    }

    static long preparedInputKey(int nodeId, int inputIndex) {
        return ((long) nodeId << Integer.SIZE) ^ (inputIndex & 0xffffffffL);
    }

    CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        return cpuWorkspaceByNodeId.get(nodeId);
    }

    Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(preparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }
}
