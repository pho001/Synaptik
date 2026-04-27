package backend.lowering;

import java.util.List;
import java.util.Objects;

public record LoweredExecutionUnit(
        String unitId,
        LoweringFamily loweringFamily,
        List<Integer> orderedNodeIds,
        List<Integer> inputNodeIds,
        LoweredUnitArtifact artifact
) {
    public LoweredExecutionUnit {
        unitId = unitId == null ? "" : unitId;
        loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
    }

    public LoweredExecutionUnit(
            String unitId,
            LoweringFamily loweringFamily,
            List<Integer> orderedNodeIds
    ) {
        this(unitId, loweringFamily, orderedNodeIds, List.of(), null);
    }

    public LoweredExecutionUnit(
            String unitId,
            LoweringFamily loweringFamily,
            List<Integer> orderedNodeIds,
            List<Integer> inputNodeIds
    ) {
        this(unitId, loweringFamily, orderedNodeIds, inputNodeIds, null);
    }

    public <T extends LoweredUnitArtifact> T requireArtifact(Class<T> artifactType) {
        Objects.requireNonNull(artifactType, "artifactType cannot be null");
        if (!artifactType.isInstance(artifact)) {
            throw new IllegalStateException(
                    "Lowered unit " + unitId + " requires artifact " + artifactType.getSimpleName()
                            + " for loweringFamily=" + loweringFamily
            );
        }
        return artifactType.cast(artifact);
    }
}
