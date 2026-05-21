package graph.compile.planning.region;

import graph.CompiledNode;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.Partition;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Backend-neutral region unit planner.
 *
 * <p>This planner describes structural graph units only. Backend lowerers decide whether those units map to a physical
 * backend graph, a fused primitive, or single-operation execution.
 */
final class StructuralRegionUnitPlanner {
    private StructuralRegionUnitPlanner() {
    }

    static List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
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
