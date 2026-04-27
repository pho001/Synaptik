package backend.cpu.fused.plan;

import backend.lowering.LoweredUnitArtifact;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public record FusedOperationPreparation(
        FusedOperation operation,
        List<Tensor> runtimeInputs
) implements LoweredUnitArtifact {
    public FusedOperationPreparation {
        operation = Objects.requireNonNull(operation, "operation cannot be null");
        runtimeInputs = List.copyOf(runtimeInputs == null ? List.of() : runtimeInputs);
    }
}
