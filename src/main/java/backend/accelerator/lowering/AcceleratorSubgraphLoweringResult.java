package backend.accelerator.lowering;

import backend.accelerator.dag.AcceleratorDagSpec;

import java.util.Objects;

/**
 * Result of lowering a candidate partition to accelerator bridge inputs.
 *
 * @param computeNodeId compiled-node id that anchors the lowered partition
 * @param matMulSpec optional legacy matmul descriptor retained for bridge compatibility
 * @param dagSpec backend-neutral lowered DAG consumed by native graph bridges
 * @param estimatedWork planner cost estimate for backend selection
 */
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
