package backend.lowering;

import java.util.List;

public record LoweringArtifacts(
        List<LoweredRegion> loweredRegions,
        List<BackendWorkspaceRequirement> workspaceRequirements
) {
    public LoweringArtifacts {
        loweredRegions = List.copyOf(loweredRegions == null ? List.of() : loweredRegions);
        workspaceRequirements = List.copyOf(workspaceRequirements == null ? List.of() : workspaceRequirements);
    }

    public static LoweringArtifacts empty() {
        return new LoweringArtifacts(List.of(), List.of());
    }
}
