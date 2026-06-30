package planning.region;

import config.optimizer.CpuFusionMode;
import graph.model.CompiledNode;
import planning.partition.Partition;
import planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * CPU region unit planner that fuses fully fusable partitions or shorter fusable subchains.
 */
final class CpuRegionPlanningPolicy {
    /**
     * Builds CPU execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region planning context
     * @return fused or single-operation units
     */
    List<ExecutionUnit> buildUnits(Partition partition, RegionPlanningContext context) {
        if (context.cpuFusionConfig().mode() == CpuFusionMode.OFF) {
            return ExecutionUnitFactory.buildSingleOpUnits(partition, context);
        }
        if (partition.orderedNodeIds().size() <= context.cpuFusionConfig().maxChainNodes()
                && ElementwiseFusionPlanner.shouldFuseWholePartition(partition, context)) {
            return List.of(ExecutionUnitFactory.buildFusedUnit(partition, context));
        }
        return buildMixedCpuUnits(partition, context);
    }

    private List<ExecutionUnit> buildMixedCpuUnits(Partition partition, RegionPlanningContext context) {
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<GraphValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        List<Integer> ordered = partition.orderedNodeIds();
        int index = 0;
        while (index < ordered.size()) {
            int nodeId = ordered.get(index);
            CompiledNode node = context.compiledNode(nodeId);
            if (!ElementwiseFusionPlanner.isSubchainFusable(node)) {
                out.add(ExecutionUnitFactory.buildSingleOpUnit(partition, nodeId, node, selected, materialized, context));
                index++;
                continue;
            }
            List<Integer> chain = new ArrayList<>();
            chain.add(nodeId);
            int cursor = index + 1;
            while (cursor < ordered.size() && chain.size() < context.cpuFusionConfig().maxChainNodes()) {
                int candidateId = ordered.get(cursor);
                CompiledNode candidate = context.compiledNode(candidateId);
                if (!ElementwiseFusionPlanner.isSubchainFusable(candidate)
                        || !ElementwiseFusionPlanner.consumesUnitOutput(candidate, chain)) {
                    break;
                }
                chain.add(candidateId);
                cursor++;
            }
            List<GraphValueRef> chainOutputs = ElementwiseFusionPlanner.unitOutputsForChain(partition, chain, context);
            boolean singlePublishedOutput = chainOutputs.size() == 1
                    && chainOutputs.getFirst().equals(GraphValueRef.node(chain.getLast()));
            if (chain.size() > 1 && singlePublishedOutput) {
                out.add(ExecutionUnitFactory.buildFusedSubchainUnit(partition, chain, context, materialized, chainOutputs));
                index = cursor;
            } else {
                out.add(ExecutionUnitFactory.buildSingleOpUnit(partition, nodeId, node, selected, materialized, context));
                index++;
            }
        }
        return List.copyOf(out);
    }
}
