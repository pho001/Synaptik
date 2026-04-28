package graph.optimizer.partition;

import graph.CompiledNode;

import java.util.Set;

/**
 * Backend-specific legality and lowering hook used by partition planners.
 *
 * <p>The planner owns search strategy; the adapter owns backend facts. It decides which nodes can seed or join a
 * region, whether dependencies may remain outside the region, and whether a structurally valid candidate can be lowered
 * to an executable {@link PartitionPlan}.
 */
public interface RegionLegalityAdapter {
    /**
     * Returns the partition target this adapter describes.
     *
     * @return backend target
     */
    PartitionTarget target();

    /**
     * Returns whether a node can be part of a region for this backend.
     *
     * @param node compiled node to test
     * @param context planning context
     * @return {@code true} when the backend can execute the node inside a partition
     */
    boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context);

    /**
     * Returns whether a node can start a candidate partition.
     *
     * @param node compiled node to test
     * @param context planning context
     * @return {@code true} when the node may anchor a partition search
     */
    boolean canSeed(CompiledNode node, PartitionPlanningContext context);

    /**
     * Returns whether {@code producer} may remain outside the selected node set while feeding {@code consumer}.
     *
     * @param producer producer node outside or inside the candidate
     * @param consumer consumer node in the candidate
     * @param selectedNodeIds current candidate node ids
     * @param context planning context
     * @return {@code true} if the dependency can be represented as an external input
     */
    boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    );

    /**
     * Builds a backend-neutral structural candidate from selected node ids.
     *
     * @param selectedNodeIds selected node ids in the candidate region
     * @param context planning context
     * @param requiredMaterializedValueRefs values that must survive the region boundary
     * @return structural candidate, or {@code null} if the selected set is not representable
     */
    PartitionCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<PartitionValueRef> requiredMaterializedValueRefs
    );

    /**
     * Lowers a structural candidate to a backend executable plan.
     *
     * @param candidate structural candidate
     * @param context planning context
     * @return backend plan, or {@code null} if lowering rejects the candidate
     */
    PartitionPlan tryCreatePlan(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    );
}
