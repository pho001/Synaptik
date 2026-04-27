package backend.lowering;

public record LoweringResult(
        LoweredRegion loweredRegion,
        java.util.List<BackendWorkspaceRequirement> workspaceRequirements
) {
    public LoweringResult {
        workspaceRequirements = java.util.List.copyOf(workspaceRequirements == null ? java.util.List.of() : workspaceRequirements);
    }
}
