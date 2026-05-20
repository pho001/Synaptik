package graph.compile.planning.region;

import config.optimizer.CpuFusionMode;
import graph.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * CPU region unit planner that fuses fully fusable partitions or shorter fusable subchains.
 */
final class CpuRegionOptimizationPolicy {
    /**
     * Builds CPU execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return fused or single-operation units
     */
    List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (context.cpuFusionConfig().mode() == CpuFusionMode.OFF) {
            return RegionOptimizationUnitSupport.buildSingleOpUnits(partition, context);
        }
        if (partition.orderedNodeIds().size() <= context.cpuFusionConfig().maxChainNodes()
                && RegionOptimizationUnitSupport.shouldFuseWholePartition(partition, context)) {
            return List.of(RegionOptimizationUnitSupport.buildFusedUnit(partition, context));
        }
        return buildMixedCpuUnits(partition, context);
    }

    private List<ExecutionUnit> buildMixedCpuUnits(Partition partition, RegionOptimizationContext context) {
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<GraphValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        List<Integer> ordered = partition.orderedNodeIds();
        int index = 0;
        while (index < ordered.size()) {
            int nodeId = ordered.get(index);
            CompiledNode node = context.compiledNode(nodeId);
            if (!RegionOptimizationUnitSupport.isSubchainFusable(node)) {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(partition, nodeId, node, selected, materialized, context));
                index++;
                continue;
            }
            List<Integer> chain = new ArrayList<>();
            chain.add(nodeId);
            int cursor = index + 1;
            while (cursor < ordered.size() && chain.size() < context.cpuFusionConfig().maxChainNodes()) {
                int candidateId = ordered.get(cursor);
                CompiledNode candidate = context.compiledNode(candidateId);
                if (!RegionOptimizationUnitSupport.isSubchainFusable(candidate)
                        || !RegionOptimizationUnitSupport.consumesUnitOutput(candidate, chain)) {
                    break;
                }
                chain.add(candidateId);
                cursor++;
            }
            List<GraphValueRef> chainOutputs = RegionOptimizationUnitSupport.unitOutputsForChain(partition, chain, context);
            boolean singlePublishedOutput = chainOutputs.size() == 1
                    && chainOutputs.getFirst().equals(GraphValueRef.node(chain.getLast()));
            if (chain.size() > 1 && singlePublishedOutput) {
                out.add(RegionOptimizationUnitSupport.buildFusedSubchainUnit(partition, chain, context, materialized, chainOutputs));
                index = cursor;
            } else {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(partition, nodeId, node, selected, materialized, context));
                index++;
            }
        }
        return List.copyOf(out);
    }
}
