package backend.lowering.partition;

import java.util.List;

public record PartitionFallbackPlan(
        List<Integer> orderedNodeIds,
        List<Integer> boundaryOutputNodeIds,
        String fallbackFamily,
        String reason,
        List<Integer> requiredInputMaterializations
) {
    public PartitionFallbackPlan {
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of() : boundaryOutputNodeIds);
        fallbackFamily = fallbackFamily == null ? "" : fallbackFamily;
        reason = reason == null ? "" : reason;
        requiredInputMaterializations = List.copyOf(requiredInputMaterializations == null ? List.of() : requiredInputMaterializations);
    }

    public static PartitionFallbackPlan none() {
        return new PartitionFallbackPlan(List.of(), List.of(), "", "", List.of());
    }
}
