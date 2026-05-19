package graph.execution.residency;

import backend.memory.TensorResidencyState;

import java.util.Map;

/**
 * Run-scoped tensor residency state by compiled node id.
 */
public final class RuntimeResidencyStore {
    private final Map<Integer, TensorResidencyState> residencyByNodeId;

    public RuntimeResidencyStore(Map<Integer, TensorResidencyState> residencyByNodeId) {
        this.residencyByNodeId = Map.copyOf(residencyByNodeId);
    }

    public TensorResidencyState residencyForNodeId(int nodeId) {
        TensorResidencyState state = residencyByNodeId.get(nodeId);
        if (state == null) {
            throw new IllegalStateException("Missing runtime residency state for nodeId=" + nodeId);
        }
        return state;
    }
}
