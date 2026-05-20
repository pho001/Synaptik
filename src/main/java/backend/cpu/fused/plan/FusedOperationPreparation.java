package backend.cpu.fused.plan;

import backend.lowering.LoweredUnitArtifact;

import java.util.List;
import java.util.Objects;

/**
 * Lowered fused-operation artifact attached to a fused execution unit.
 *
 * @param operation fused operation descriptor
 * @param runtimeInputNodeIds external graph value node ids that must be supplied to the executable
 */
public record FusedOperationPreparation(
        FusedOperation operation,
        List<Integer> runtimeInputNodeIds
) implements LoweredUnitArtifact {
    public FusedOperationPreparation {
        operation = Objects.requireNonNull(operation, "operation cannot be null");
        runtimeInputNodeIds = List.copyOf(runtimeInputNodeIds == null ? List.of() : runtimeInputNodeIds);
    }
}
