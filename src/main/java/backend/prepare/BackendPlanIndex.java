package backend.prepare;

import backend.contract.ComputeBackend;
import graph.compile.planning.partition.PartitionPlan;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class BackendPlanIndex {
    private final Map<Integer, PartitionPlan> plansByAnchor;

    BackendPlanIndex() {
        this.plansByAnchor = new HashMap<>();
    }

    private BackendPlanIndex(Map<Integer, PartitionPlan> plansByAnchor) {
        this.plansByAnchor = new HashMap<>(plansByAnchor);
    }

    void publish(List<PartitionPlan> plans, PartitionRoleIndex roleIndex) {
        plansByAnchor.clear();
        if (plans == null) {
            return;
        }
        for (PartitionPlan plan : plans) {
            if (plan == null) {
                continue;
            }
            plansByAnchor.put(plan.anchorNodeId(), plan);
            if (plan.backend() != ComputeBackend.CPU) {
                roleIndex.publishRoles(plan.anchorNodeId(), plan.nodeIds());
            }
        }
    }

    PartitionPlan planForAnchor(int nodeId) {
        return plansByAnchor.get(nodeId);
    }

    BackendPlanIndex fork() {
        return new BackendPlanIndex(plansByAnchor);
    }
}
