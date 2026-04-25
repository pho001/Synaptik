package graph.optimizer.partition;

public interface AcceleratorPartitionPlanner {
    PartitionPlanningResult plan(PartitionPlanningRequest request);
}
