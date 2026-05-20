package backend.prepare;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.region.RegionExecutionPlan;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class LoweredRegionIndex {
    private final Map<Integer, LoweredExecutionUnit> cpuUnitsByAnchor;
    private final Map<Integer, LoweredRegion> metalRegionsByAnchor;
    private final Map<Integer, LoweredRegion> cudaRegionsByAnchor;

    LoweredRegionIndex() {
        this.cpuUnitsByAnchor = new HashMap<>();
        this.metalRegionsByAnchor = new HashMap<>();
        this.cudaRegionsByAnchor = new HashMap<>();
    }

    private LoweredRegionIndex(
            Map<Integer, LoweredExecutionUnit> cpuUnitsByAnchor,
            Map<Integer, LoweredRegion> metalRegionsByAnchor,
            Map<Integer, LoweredRegion> cudaRegionsByAnchor
    ) {
        this.cpuUnitsByAnchor = new HashMap<>(cpuUnitsByAnchor);
        this.metalRegionsByAnchor = new HashMap<>(metalRegionsByAnchor);
        this.cudaRegionsByAnchor = new HashMap<>(cudaRegionsByAnchor);
    }

    void publish(List<LoweredRegion> loweredRegions, PartitionRoleIndex roleIndex) {
        cpuUnitsByAnchor.clear();
        metalRegionsByAnchor.clear();
        cudaRegionsByAnchor.clear();
        if (loweredRegions == null) {
            return;
        }
        for (LoweredRegion region : loweredRegions) {
            if (region == null) {
                continue;
            }
            if (region.target() == graph.compile.planning.partition.PartitionTarget.CPU) {
                publishCpuRegion(region, roleIndex);
            }
        }
        for (LoweredRegion region : loweredRegions) {
            if (region == null) {
                continue;
            }
            switch (region.target()) {
                case GPU_METAL -> publishGpuRegion(region, metalRegionsByAnchor, roleIndex);
                case GPU_CUDA -> publishGpuRegion(region, cudaRegionsByAnchor, roleIndex);
                default -> {
                }
            }
        }
    }

    private void publishCpuRegion(LoweredRegion region, PartitionRoleIndex roleIndex) {
        for (LoweredExecutionUnit unit : region.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            RegionExecutionPlan plan = regionPlan(unit);
            if (plan == null && unit.loweringFamily() != LoweringFamily.FUSED_NATIVE) {
                continue;
            }
            int anchorNodeId = plan == null ? unit.orderedNodeIds().getLast() : plan.anchorNodeId();
            cpuUnitsByAnchor.put(anchorNodeId, unit);
            roleIndex.publishRoles(anchorNodeId, plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds());
        }
    }

    private void publishGpuRegion(
            LoweredRegion region,
            Map<Integer, LoweredRegion> regionsByAnchor,
            PartitionRoleIndex roleIndex
    ) {
        int anchorNodeId = resolveAnchorNodeId(region);
        if (anchorNodeId < 0) {
            return;
        }
        regionsByAnchor.put(anchorNodeId, region);
        for (LoweredExecutionUnit unit : region.units()) {
            if (unit != null) {
                RegionExecutionPlan plan = regionPlan(unit);
                roleIndex.publishRoles(anchorNodeId, plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds());
            }
        }
    }

    private int resolveAnchorNodeId(LoweredRegion region) {
        if (region == null || region.units().isEmpty()) {
            return -1;
        }
        int anchorNodeId = -1;
        for (LoweredExecutionUnit unit : region.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            RegionExecutionPlan plan = regionPlan(unit);
            anchorNodeId = Math.max(anchorNodeId, plan == null ? unit.orderedNodeIds().getLast() : plan.anchorNodeId());
        }
        return anchorNodeId;
    }

    private RegionExecutionPlan regionPlan(LoweredExecutionUnit unit) {
        return unit != null && unit.artifact() instanceof RegionExecutionPlan plan ? plan : null;
    }

    LoweredExecutionUnit cpuUnitForAnchor(int nodeId) {
        return cpuUnitsByAnchor.get(nodeId);
    }

    LoweredRegion metalRegionForAnchor(int nodeId) {
        return metalRegionsByAnchor.get(nodeId);
    }

    LoweredRegion cudaRegionForAnchor(int nodeId) {
        return cudaRegionsByAnchor.get(nodeId);
    }

    LoweredRegionIndex fork() {
        return new LoweredRegionIndex(cpuUnitsByAnchor, metalRegionsByAnchor, cudaRegionsByAnchor);
    }
}
