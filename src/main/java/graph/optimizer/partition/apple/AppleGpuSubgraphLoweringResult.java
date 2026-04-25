package graph.optimizer.partition.apple;

import graph.optimizer.partition.model.AcceleratorDagSpec;

import java.util.Objects;

public record AppleGpuSubgraphLoweringResult(
        int computeNodeId,
        AppleGpuMatMulSpec matMulSpec,
        AcceleratorDagSpec dagSpec,
        long estimatedWork
) {
    public AppleGpuSubgraphLoweringResult {
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
    }
}
