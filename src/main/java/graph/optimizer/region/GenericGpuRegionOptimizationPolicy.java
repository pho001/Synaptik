package graph.optimizer.region;

import graph.CompiledNode;
import graph.optimizer.partition.Partition;
import graph.optimizer.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generic accelerator region policy.
 *
 * <p>The policy fuses an entire partition only when every node is fusable and there is a single output. Mixed GPU
 * partitions preserve the selected region and fuse maximal elementwise subchains as region-internal units.
 */
public final class GenericGpuRegionOptimizationPolicy implements RegionOptimizationPolicy {
    /**
     * Builds generic accelerator execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return fused whole-partition unit, mixed subchain units, or single-operation units
     */
    @Override
    public List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (RegionOptimizationUnitSupport.shouldFuseWholePartition(partition, context)) {
            return List.of(RegionOptimizationUnitSupport.buildFusedUnit(partition, context));
        }
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<GraphValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        int index = 0;
        while (index < partition.orderedNodeIds().size()) {
            int nodeId = partition.orderedNodeIds().get(index);
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                index++;
                continue;
            }
            List<Integer> epilogueSpan = RegionOptimizationUnitSupport.epilogueSpanAt(partition, index, context);
            if (!epilogueSpan.isEmpty()) {
                out.add(RegionOptimizationUnitSupport.buildEpilogueSubregionUnit(
                        partition,
                        epilogueSpan,
                        context,
                        materialized,
                        RegionOptimizationUnitSupport.unitOutputsForChain(partition, epilogueSpan, context)
                ));
                index += epilogueSpan.size();
                continue;
            }
            if (!RegionOptimizationUnitSupport.isSubchainFusable(node)) {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(
                        partition,
                        nodeId,
                        node,
                        selected,
                        materialized,
                        context
                ));
                index++;
                continue;
            }

            List<Integer> chain = new ArrayList<>();
            chain.add(nodeId);
            int cursor = index + 1;
            while (cursor < partition.orderedNodeIds().size()) {
                int candidateId = partition.orderedNodeIds().get(cursor);
                if (!RegionOptimizationUnitSupport.isSubchainFusable(context.compiledNode(candidateId))
                        || !RegionOptimizationUnitSupport.consumesUnitOutput(context.compiledNode(candidateId), chain)) {
                    break;
                }
                chain.add(candidateId);
                cursor++;
            }

            if (chain.size() >= 2) {
                out.add(RegionOptimizationUnitSupport.buildFusedSubchainUnit(
                        partition,
                        chain,
                        context,
                        materialized,
                        RegionOptimizationUnitSupport.unitOutputsForChain(partition, chain, context)
                ));
                index = cursor;
            } else {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(
                        partition,
                        nodeId,
                        node,
                        selected,
                        materialized,
                        context
                ));
                index++;
            }
        }
        return List.copyOf(out);
    }
}
