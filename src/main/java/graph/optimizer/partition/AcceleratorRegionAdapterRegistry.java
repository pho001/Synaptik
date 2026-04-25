package graph.optimizer.partition;

import graph.optimizer.partition.apple.AppleGpuRegionLegalityAdapter;

public final class AcceleratorRegionAdapterRegistry {
    private AcceleratorRegionAdapterRegistry() {
    }

    public static AcceleratorRegionLegalityAdapter forTarget(AcceleratorTarget target) {
        AcceleratorTarget resolved = target == null ? AcceleratorTarget.NONE : target;
        return switch (resolved) {
            case GPU_METAL -> new AppleGpuRegionLegalityAdapter();
            default -> new UnsupportedAcceleratorRegionLegalityAdapter(resolved);
        };
    }
}
