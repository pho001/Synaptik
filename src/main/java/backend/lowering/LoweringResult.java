package backend.lowering;

/**
 * Output of lowering one optimized partition.
 *
 * @param loweredPartition backend-specific lowered partition, or {@code null} if lowering did not apply
 * @param workspaceRequirements backend workspace requirements produced by lowering
 */
public record LoweringResult(
        LoweredPartition loweredPartition,
        java.util.List<BackendWorkspaceRequirement> workspaceRequirements
) {
    public LoweringResult {
        workspaceRequirements = java.util.List.copyOf(workspaceRequirements == null ? java.util.List.of() : workspaceRequirements);
    }
}
