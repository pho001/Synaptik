package backend.lowering.region;

import java.util.List;

public record CpuNativeRegionPayload(
        String providerKind,
        List<Integer> providerNodeIds,
        List<Integer> localKernelNodeIds,
        List<Integer> preparedWeightCacheCandidateNodeIds,
        List<RegionFallbackPlan> fallbackPlans
) implements RegionBackendPayload {
    public CpuNativeRegionPayload {
        providerKind = providerKind == null ? "" : providerKind;
        providerNodeIds = List.copyOf(providerNodeIds == null ? List.of() : providerNodeIds);
        localKernelNodeIds = List.copyOf(localKernelNodeIds == null ? List.of() : localKernelNodeIds);
        preparedWeightCacheCandidateNodeIds = List.copyOf(preparedWeightCacheCandidateNodeIds == null
                ? List.of()
                : preparedWeightCacheCandidateNodeIds);
        fallbackPlans = List.copyOf(fallbackPlans == null ? List.of() : fallbackPlans);
    }
}
