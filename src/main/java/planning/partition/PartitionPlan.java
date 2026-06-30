package planning.partition;

import backend.contract.ComputeBackend;

import java.util.List;

/**
 * Backend-specific execution plan attached to an accepted partition.
 *
 * <p>Partition planners use this interface without knowing the concrete backend lowerer. Preparation later consumes
 * the plan to select fused or backend-specific execution units.
 */
public interface PartitionPlan {
    /**
     * Returns the backend that can execute this plan.
     *
     * @return compute backend
     */
    ComputeBackend backend();

    /**
     * Returns the anchor node used to seed the partition.
     *
     * @return anchor node id
     */
    int anchorNodeId();

    /**
     * Returns node ids included in the partition.
     *
     * @return ordered node ids
     */
    List<Integer> nodeIds();

    /**
     * Returns graph node ids that feed the partition from outside.
     *
     * @return external input node ids
     */
    List<Integer> externalInputNodeIds();

    /**
     * Returns node ids whose values leave the partition.
     *
     * @return produced output node ids
     */
    List<Integer> producedOutputNodeIds();

    /**
     * Returns estimated backend work for scoring and diagnostics.
     *
     * @return non-negative work estimate
     */
    long estimatedWork();

}
