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
    private final Map<Integer, LoweredExecutionUnit> cpuFusedUnitsByStart;
    private final Map<Integer, LoweredExecutionUnit> cpuNativeUnitsByStart;
    private final Map<Integer, LoweredRegion> metalRegionsByAnchor;
    private final Map<Integer, LoweredRegion> metalRegionsByStart;
    private final Map<Integer, LoweredRegion> cudaRegionsByAnchor;
    private final Map<Integer, LoweredRegion> cudaRegionsByStart;

    LoweredRegionIndex() {
        this.cpuUnitsByAnchor = new HashMap<>();
        this.cpuFusedUnitsByStart = new HashMap<>();
        this.cpuNativeUnitsByStart = new HashMap<>();
        this.metalRegionsByAnchor = new HashMap<>();
        this.metalRegionsByStart = new HashMap<>();
        this.cudaRegionsByAnchor = new HashMap<>();
        this.cudaRegionsByStart = new HashMap<>();
    }

    private LoweredRegionIndex(
            Map<Integer, LoweredExecutionUnit> cpuUnitsByAnchor,
            Map<Integer, LoweredExecutionUnit> cpuFusedUnitsByStart,
            Map<Integer, LoweredExecutionUnit> cpuNativeUnitsByStart,
            Map<Integer, LoweredRegion> metalRegionsByAnchor,
            Map<Integer, LoweredRegion> metalRegionsByStart,
            Map<Integer, LoweredRegion> cudaRegionsByAnchor,
            Map<Integer, LoweredRegion> cudaRegionsByStart
    ) {
        this.cpuUnitsByAnchor = new HashMap<>(cpuUnitsByAnchor);
        this.cpuFusedUnitsByStart = new HashMap<>(cpuFusedUnitsByStart);
        this.cpuNativeUnitsByStart = new HashMap<>(cpuNativeUnitsByStart);
        this.metalRegionsByAnchor = new HashMap<>(metalRegionsByAnchor);
        this.metalRegionsByStart = new HashMap<>(metalRegionsByStart);
        this.cudaRegionsByAnchor = new HashMap<>(cudaRegionsByAnchor);
        this.cudaRegionsByStart = new HashMap<>(cudaRegionsByStart);
    }

    void publish(List<LoweredRegion> loweredRegions, PartitionRoleIndex roleIndex) {
        cpuUnitsByAnchor.clear();
        cpuFusedUnitsByStart.clear();
        cpuNativeUnitsByStart.clear();
        metalRegionsByAnchor.clear();
        metalRegionsByStart.clear();
        cudaRegionsByAnchor.clear();
        cudaRegionsByStart.clear();
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
                case GPU_METAL -> publishGpuRegion(region, metalRegionsByAnchor, metalRegionsByStart, roleIndex);
                case GPU_CUDA -> publishGpuRegion(region, cudaRegionsByAnchor, cudaRegionsByStart, roleIndex);
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
            if (unit.loweringFamily() == LoweringFamily.FUSED_NATIVE) {
                cpuFusedUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
                continue;
            }
            int anchorNodeId = plan == null ? unit.orderedNodeIds().getLast() : plan.anchorNodeId();
            cpuUnitsByAnchor.put(anchorNodeId, unit);
            if (unit.loweringFamily() == LoweringFamily.CPU_NATIVE_REGION) {
                cpuNativeUnitsByStart.put((plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds()).getFirst(), unit);
            }
            roleIndex.publishRoles(anchorNodeId, plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds());
        }
    }

    private void publishGpuRegion(
            LoweredRegion region,
            Map<Integer, LoweredRegion> regionsByAnchor,
            Map<Integer, LoweredRegion> regionsByStart,
            PartitionRoleIndex roleIndex
    ) {
        int anchorNodeId = resolveAnchorNodeId(region);
        if (anchorNodeId < 0) {
            return;
        }
        regionsByAnchor.put(anchorNodeId, region);
        int startNodeId = resolveStartNodeId(region);
        if (startNodeId >= 0) {
            regionsByStart.put(startNodeId, region);
        }
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

    private int resolveStartNodeId(LoweredRegion region) {
        if (region == null || region.units().isEmpty()) {
            return -1;
        }
        int startNodeId = Integer.MAX_VALUE;
        for (LoweredExecutionUnit unit : region.units()) {
            if (unit == null || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            RegionExecutionPlan plan = regionPlan(unit);
            List<Integer> orderedNodeIds = plan == null ? unit.orderedNodeIds() : plan.orderedNodeIds();
            if (!orderedNodeIds.isEmpty()) {
                startNodeId = Math.min(startNodeId, orderedNodeIds.getFirst());
            }
        }
        return startNodeId == Integer.MAX_VALUE ? -1 : startNodeId;
    }

    private RegionExecutionPlan regionPlan(LoweredExecutionUnit unit) {
        return unit != null && unit.artifact() instanceof RegionExecutionPlan plan ? plan : null;
    }

    LoweredExecutionUnit cpuUnitForAnchor(int nodeId) {
        return cpuUnitsByAnchor.get(nodeId);
    }

    LoweredExecutionUnit cpuFusedUnitForStart(int nodeId) {
        return cpuFusedUnitsByStart.get(nodeId);
    }

    LoweredExecutionUnit cpuNativeUnitForStart(int nodeId) {
        return cpuNativeUnitsByStart.get(nodeId);
    }

    LoweredRegion metalRegionForAnchor(int nodeId) {
        return metalRegionsByAnchor.get(nodeId);
    }

    LoweredRegion metalRegionForStart(int nodeId) {
        return metalRegionsByStart.get(nodeId);
    }

    LoweredRegion cudaRegionForAnchor(int nodeId) {
        return cudaRegionsByAnchor.get(nodeId);
    }

    LoweredRegion cudaRegionForStart(int nodeId) {
        return cudaRegionsByStart.get(nodeId);
    }

    LoweredRegionIndex fork() {
        return new LoweredRegionIndex(
                cpuUnitsByAnchor,
                cpuFusedUnitsByStart,
                cpuNativeUnitsByStart,
                metalRegionsByAnchor,
                metalRegionsByStart,
                cudaRegionsByAnchor,
                cudaRegionsByStart
        );
    }
}
