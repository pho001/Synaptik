package backend.lowering.region;

import java.util.List;

public record RegionFallbackPlan(
        List<Integer> orderedNodeIds,
        List<Integer> boundaryOutputNodeIds,
        String fallbackFamily,
        String reason,
        List<Integer> requiredInputMaterializations
) {
    public RegionFallbackPlan {
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of() : boundaryOutputNodeIds);
        fallbackFamily = fallbackFamily == null ? "" : fallbackFamily;
        reason = reason == null ? "" : reason;
        requiredInputMaterializations = List.copyOf(requiredInputMaterializations == null ? List.of() : requiredInputMaterializations);
    }

    public static RegionFallbackPlan none() {
        return new RegionFallbackPlan(List.of(), List.of(), "", "", List.of());
    }
}
