package backend.lowering;

/**
 * Output of lowering one optimized region.
 *
 * @param loweredRegion backend-specific lowered region, or {@code null} if lowering did not apply
 * @param workspaceRequirements backend workspace requirements produced by lowering
 */
public record LoweringResult(
        LoweredRegion loweredRegion,
        java.util.List<BackendWorkspaceRequirement> workspaceRequirements
) {
    public LoweringResult {
        workspaceRequirements = java.util.List.copyOf(workspaceRequirements == null ? java.util.List.of() : workspaceRequirements);
    }
}
