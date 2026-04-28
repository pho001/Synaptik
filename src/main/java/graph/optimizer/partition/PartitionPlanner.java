package graph.optimizer.partition;

/**
 * Strategy interface for selecting backend partitions from compiled nodes.
 *
 * <p>A planner receives a {@link PartitionPlanningRequest} with target backend, legality adapter, scoring policy, and
 * required materialization outputs. It returns accepted partitions, optional backend plans, and compile trace decisions.
 */
public interface PartitionPlanner {
    /**
     * Plans partitions for the supplied request.
     *
     * @param request planning request
     * @return planning result; implementations should return {@link PartitionPlanningResult#empty()} when no work is
     * available
     */
    PartitionPlanningResult plan(PartitionPlanningRequest request);
}
