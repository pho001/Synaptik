package backend.lowering;

import java.util.List;

public record LoweringArtifacts(
        List<LoweredPartition> loweredPartitions,
        List<BackendWorkspaceRequirement> workspaceRequirements
) {
    public LoweringArtifacts {
        loweredPartitions = List.copyOf(loweredPartitions == null ? List.of() : loweredPartitions);
        workspaceRequirements = List.copyOf(workspaceRequirements == null ? List.of() : workspaceRequirements);
    }

    public static LoweringArtifacts empty() {
        return new LoweringArtifacts(List.of(), List.of());
    }
}
