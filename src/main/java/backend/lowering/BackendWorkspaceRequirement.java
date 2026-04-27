package backend.lowering;

public record BackendWorkspaceRequirement(
        String unitId,
        String requirementKind,
        long bytes
) {
    public BackendWorkspaceRequirement {
        unitId = unitId == null ? "" : unitId;
        requirementKind = requirementKind == null ? "" : requirementKind;
        bytes = Math.max(0L, bytes);
    }
}
