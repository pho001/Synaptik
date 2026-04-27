package backend.prepare;

import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;

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
            switch (region.target()) {
                case CPU -> publishCpuRegion(region, roleIndex);
                case GPU_METAL -> publishGpuRegion(region, metalRegionsByAnchor, roleIndex);
                case GPU_CUDA -> publishGpuRegion(region, cudaRegionsByAnchor, roleIndex);
                default -> {
                }
            }
        }
    }

    private void publishCpuRegion(LoweredRegion region, PartitionRoleIndex roleIndex) {
        for (LoweredExecutionUnit unit : region.units()) {
            if (unit == null || unit.loweringFamily() != LoweringFamily.FUSED_NATIVE || unit.orderedNodeIds().isEmpty()) {
                continue;
            }
            int anchorNodeId = unit.orderedNodeIds().getLast();
            cpuUnitsByAnchor.put(anchorNodeId, unit);
            roleIndex.publishRoles(anchorNodeId, unit.orderedNodeIds());
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
                roleIndex.publishRoles(anchorNodeId, unit.orderedNodeIds());
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
            anchorNodeId = Math.max(anchorNodeId, unit.orderedNodeIds().getLast());
        }
        return anchorNodeId;
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
