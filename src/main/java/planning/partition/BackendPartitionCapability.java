package planning.partition;

import config.compile.BackendPlanningCostConfig;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import planning.value.GraphValueRef;

import graph.model.CompiledNode;

import java.util.Set;

/**
 * Backend-specific partition capability used by partition planners.
 *
 * <p>The planner owns search strategy; the capability owns backend facts. It decides which nodes can seed or join a
 * partition, whether dependencies may remain outside the partition, and whether a structurally valid candidate can be lowered
 * to an executable {@link PartitionPlan}.
 */
public interface BackendPartitionCapability {
    /**
     * Returns the partition target this capability describes.
     *
     * @return backend target
     */
    PartitionTarget target();

    /**
     * Resolves the static materialization cost preset for this backend target.
     *
     * @param costConfig compile-time cost configuration
     * @return backend-specific static cost preset, or conservative defaults
     */
    default AcceleratorPartitionScoreModel.StaticCostPreset costPreset(BackendPlanningCostConfig costConfig) {
        return AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
    }

    /**
     * Returns whether a node can be part of a partition for this backend.
     *
     * @param node compiled node to test
     * @param context planning context
     * @return {@code true} when the backend can execute the node inside a partition
     */
    boolean canExecute(CompiledNode node, PartitionPlanningContext context);

    /**
     * Returns whether a node can start a candidate partition.
     *
     * @param node compiled node to test
     * @param context planning context
     * @return {@code true} when the node may anchor a partition search
     */
    boolean canSeed(CompiledNode node, PartitionPlanningContext context);

    /**
     * Returns this backend's static search priority for a candidate node.
     *
     * <p>The partition planner owns traversal, budgets, and structural bonuses. Backend capabilities own
     * operation-family preferences so adding backend coverage does not require editing the generic planner.
     *
     * @param node compiled node to prioritize
     * @param context planning context
     * @return larger values are visited earlier
     */
    default int partitionPriority(CompiledNode node, PartitionPlanningContext context) {
        return 0;
    }

    /**
     * Returns whether {@code producer} may remain outside the selected node set while feeding {@code consumer}.
     *
     * @param producer producer node outside or inside the candidate
     * @param consumer consumer node in the candidate
     * @param selectedNodeIds current candidate node ids
     * @param context planning context
     * @return {@code true} if the dependency can be represented as an external input
     */
    boolean canUseExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    );

    /**
     * Builds a backend-neutral structural candidate from selected node ids.
     *
     * @param selectedNodeIds selected node ids in the candidate partition
     * @param context planning context
     * @param requiredMaterializedValueRefs values that must survive the partition boundary
     * @return structural candidate, or {@code null} if the selected set is not representable
     */
    PartitionCandidate createCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<GraphValueRef> requiredMaterializedValueRefs
    );

    /**
     * Lowers a structural candidate to a backend executable plan.
     *
     * @param candidate structural candidate
     * @param context planning context
     * @return backend plan, or {@code null} if lowering rejects the candidate
     */
    PartitionPlan createPlan(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    );
}
