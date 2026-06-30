package prepare.context;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class PartitionRoleIndex {
    private final Map<Integer, PartitionExecutionRole> rolesByNodeId;

    PartitionRoleIndex() {
        this.rolesByNodeId = new HashMap<>();
    }

    private PartitionRoleIndex(Map<Integer, PartitionExecutionRole> rolesByNodeId) {
        this.rolesByNodeId = new HashMap<>(rolesByNodeId);
    }

    void publishRoles(int anchorNodeId, List<Integer> nodeIds) {
        if (nodeIds == null) {
            return;
        }
        for (int nodeId : nodeIds) {
            rolesByNodeId.put(nodeId, nodeId == anchorNodeId
                    ? PartitionExecutionRole.ANCHOR
                    : PartitionExecutionRole.INTERIOR);
        }
    }

    PartitionExecutionRole roleFor(int nodeId) {
        return rolesByNodeId.getOrDefault(nodeId, PartitionExecutionRole.NONE);
    }

    PartitionRoleIndex fork() {
        return new PartitionRoleIndex(rolesByNodeId);
    }
}
