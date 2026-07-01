package prepare.context;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredPartition;
import backend.lowering.LoweringFamily;
import backend.lowering.partition.CpuSpecializedPrimitivePayload;
import backend.lowering.partition.BackendPartitionExecutionPlan;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class LoweredPartitionIndex {
    private final Map<Integer, LoweredExecutionUnit> cpuUnitsByAnchor;
    private final Map<Integer, LoweredExecutionUnit> cpuFusedUnitsByStart;
    private final Map<Integer, LoweredExecutionUnit> cpuSpecializedUnitsByStart;
    private final Map<Integer, LoweredPartition> metalPartitionsByAnchor;
    private final Map<Integer, LoweredPartition> metalPartitionsByStart;
    private final Map<Integer, LoweredPartition> cudaPartitionsByAnchor;
    private final Map<Integer, LoweredPartition> cudaPartitionsByStart;

    LoweredPartitionIndex() {
        this.cpuUnitsByAnchor = new HashMap<>();
        this.cpuFusedUnitsByStart = new HashMap<>();
        this.cpuSpecializedUnitsByStart = new HashMap<>();
        this.metalPartitionsByAnchor = new HashMap<>();
        this.metalPartitionsByStart = new HashMap<>();
        this.cudaPartitionsByAnchor = new HashMap<>();
        this.cudaPartitionsByStart = new HashMap<>();
    }

    private LoweredPartitionIndex(
            Map<Integer, LoweredExecutionUnit> cpuUnitsByAnchor,
            Map<Integer, LoweredExecutionUnit> cpuFusedUnitsByStart,
            Map<Integer, LoweredExecutionUnit> cpuSpecializedUnitsByStart,
            Map<Integer, LoweredPartition> metalPartitionsByAnchor,
            Map<Integer, LoweredPartition> metalPartitionsByStart,
            Map<Integer, LoweredPartition> cudaPartitionsByAnchor,
            Map<Integer, LoweredPartition> cudaPartitionsByStart
    ) {
        this.cpuUnitsByAnchor = new HashMap<>(cpuUnitsByAnchor);
        this.cpuFusedUnitsByStart = new HashMap<>(cpuFusedUnitsByStart);
        this.cpuSpecializedUnitsByStart = new HashMap<>(cpuSpecializedUnitsByStart);
        this.metalPartitionsByAnchor = new HashMap<>(metalPartitionsByAnchor);
        this.metalPartitionsByStart = new HashMap<>(metalPartitionsByStart);
        this.cudaPartitionsByAnchor = new HashMap<>(cudaPartitionsByAnchor);
        this.cudaPartitionsByStart = new HashMap<>(cudaPartitionsByStart);
    }

    void publish(List<LoweredPartition> loweredPartitions, PartitionRoleIndex roleIndex) {
        cpuUnitsByAnchor.clear();
        cpuFusedUnitsByStart.clear();
        cpuSpecializedUnitsByStart.clear();
        metalPartitionsByAnchor.clear();
        metalPartitionsByStart.clear();
        cudaPartitionsByAnchor.clear();
        cudaPartitionsByStart.clear();
        if (loweredPartitions == null) {
            return;
        }
        for (LoweredPartition partition : loweredPartitions) {
            if (partition == null) {
                continue;
            }
            if (partition.source().partition().target() == planning.partition.PartitionTarget.CPU) {
                publishCpuPartition(partition, roleIndex);
            }
        }
        for (LoweredPartition partition : loweredPartitions) {
            if (partition == null) {
                continue;
            }
            switch (partition.source().partition().target()) {
                case GPU_METAL -> publishGpuPartition(partition, metalPartitionsByAnchor, metalPartitionsByStart, roleIndex);
                case GPU_CUDA -> publishGpuPartition(partition, cudaPartitionsByAnchor, cudaPartitionsByStart, roleIndex);
                default -> {
                }
            }
        }
    }

    private void publishCpuPartition(LoweredPartition partition, PartitionRoleIndex roleIndex) {
        for (LoweredExecutionUnit unit : partition.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            BackendPartitionExecutionPlan plan = partitionPlan(unit);
            if (plan == null && unit.loweringFamily() != LoweringFamily.FUSED_NATIVE) {
                continue;
            }
            if (unit.loweringFamily() == LoweringFamily.FUSED_NATIVE) {
                cpuFusedUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
                continue;
            }
            if (isCpuSpecializedUnit(plan)) {
                cpuSpecializedUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
            }
            int anchorNodeId = plan == null ? unit.orderedNodeIds().getLast() : plan.anchorNodeId();
            cpuUnitsByAnchor.put(anchorNodeId, unit);
            roleIndex.publishRoles(anchorNodeId, plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds());
        }
    }

    private void publishGpuPartition(
            LoweredPartition partition,
            Map<Integer, LoweredPartition> partitionsByAnchor,
            Map<Integer, LoweredPartition> partitionsByStart,
            PartitionRoleIndex roleIndex
    ) {
        int anchorNodeId = resolveAnchorNodeId(partition);
        if (anchorNodeId < 0) {
            return;
        }
        partitionsByAnchor.put(anchorNodeId, partition);
        int startNodeId = resolveStartNodeId(partition);
        if (startNodeId >= 0) {
            partitionsByStart.put(startNodeId, partition);
        }
        for (LoweredExecutionUnit unit : partition.units()) {
            if (unit != null) {
                BackendPartitionExecutionPlan plan = partitionPlan(unit);
                roleIndex.publishRoles(anchorNodeId, plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds());
            }
        }
    }

    private int resolveAnchorNodeId(LoweredPartition partition) {
        if (partition == null || partition.units().isEmpty()) {
            return -1;
        }
        int anchorNodeId = -1;
        for (LoweredExecutionUnit unit : partition.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            BackendPartitionExecutionPlan plan = partitionPlan(unit);
            anchorNodeId = Math.max(anchorNodeId, plan == null ? unit.orderedNodeIds().getLast() : plan.anchorNodeId());
        }
        return anchorNodeId;
    }

    private int resolveStartNodeId(LoweredPartition partition) {
        if (partition == null || partition.units().isEmpty()) {
            return -1;
        }
        int startNodeId = Integer.MAX_VALUE;
        for (LoweredExecutionUnit unit : partition.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            BackendPartitionExecutionPlan plan = partitionPlan(unit);
            List<Integer> orderedNodeIds = plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds();
            if (!orderedNodeIds.isEmpty()) {
                startNodeId = Math.min(startNodeId, orderedNodeIds.getFirst());
            }
        }
        return startNodeId == Integer.MAX_VALUE ? -1 : startNodeId;
    }

    private BackendPartitionExecutionPlan partitionPlan(LoweredExecutionUnit unit) {
        return unit != null && unit.artifact() instanceof BackendPartitionExecutionPlan plan ? plan : null;
    }

    private boolean isCpuSpecializedUnit(BackendPartitionExecutionPlan plan) {
        return plan != null && plan.backendPayload() instanceof CpuSpecializedPrimitivePayload;
    }

    LoweredExecutionUnit cpuUnitForAnchor(int nodeId) {
        return cpuUnitsByAnchor.get(nodeId);
    }

    LoweredExecutionUnit cpuFusedUnitForStart(int nodeId) {
        return cpuFusedUnitsByStart.get(nodeId);
    }

    LoweredExecutionUnit cpuSpecializedUnitForStart(int nodeId) {
        return cpuSpecializedUnitsByStart.get(nodeId);
    }

    LoweredPartition metalPartitionForAnchor(int nodeId) {
        return metalPartitionsByAnchor.get(nodeId);
    }

    LoweredPartition metalPartitionForStart(int nodeId) {
        return metalPartitionsByStart.get(nodeId);
    }

    LoweredPartition cudaPartitionForAnchor(int nodeId) {
        return cudaPartitionsByAnchor.get(nodeId);
    }

    LoweredPartition cudaPartitionForStart(int nodeId) {
        return cudaPartitionsByStart.get(nodeId);
    }

    LoweredPartitionIndex fork() {
        return new LoweredPartitionIndex(
                cpuUnitsByAnchor,
                cpuFusedUnitsByStart,
                cpuSpecializedUnitsByStart,
                metalPartitionsByAnchor,
                metalPartitionsByStart,
                cudaPartitionsByAnchor,
                cudaPartitionsByStart
        );
    }
}
