package backend.accelerator.lowering;

import backend.accelerator.dag.AcceleratorDagSpec;

import java.util.Objects;

public record AcceleratorSubgraphLoweringResult(
        int computeNodeId,
        AcceleratorMatMulSpec matMulSpec,
        AcceleratorDagSpec dagSpec,
        long estimatedWork
) {
    public AcceleratorSubgraphLoweringResult {
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
    }
}
