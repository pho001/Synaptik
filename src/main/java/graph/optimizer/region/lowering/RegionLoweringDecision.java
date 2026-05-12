package graph.optimizer.region.lowering;

import java.util.List;

/**
 * Policy decision explaining how a backend-owned region should treat an operation.
 */
public record RegionLoweringDecision(
        OperationSemanticLevel semanticLevel,
        RegionLoweringAction action,
        List<RegionLoweringForm> forms,
        String reason
) {
    public RegionLoweringDecision {
        semanticLevel = semanticLevel == null ? OperationSemanticLevel.UNKNOWN : semanticLevel;
        action = action == null ? RegionLoweringAction.REJECT_WITH_REASON : action;
        forms = List.copyOf(forms == null ? List.of() : forms);
        reason = reason == null ? "" : reason;
    }

    public static RegionLoweringDecision keep(OperationSemanticLevel level, RegionLoweringForm form, String reason) {
        return new RegionLoweringDecision(level, RegionLoweringAction.KEEP_AS_BACKEND_PRIMITIVE, List.of(form), reason);
    }

    public static RegionLoweringDecision lower(OperationSemanticLevel level, RegionLoweringForm form, String reason) {
        return new RegionLoweringDecision(level, RegionLoweringAction.LOWER_TO_BACKEND_DAG, List.of(form), reason);
    }

    public static RegionLoweringDecision fuse(OperationSemanticLevel level, RegionLoweringForm form, String reason) {
        return new RegionLoweringDecision(level, RegionLoweringAction.FUSE_WITH_NEIGHBORS, List.of(form), reason);
    }

    public static RegionLoweringDecision reject(OperationSemanticLevel level, String reason) {
        return new RegionLoweringDecision(level, RegionLoweringAction.REJECT_WITH_REASON, List.of(RegionLoweringForm.UNSUPPORTED), reason);
    }
}
