package graph.optimizer.region;

import config.optimizer.CpuFusionMode;
import graph.CompiledNode;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * CPU region policy that fuses fully fusable partitions or shorter fusable subchains.
 */
public final class CpuRegionOptimizationPolicy implements RegionOptimizationPolicy {
    /**
     * Builds CPU execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return fused or single-operation units
     */
    @Override
    public List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (context.cpuFusionConfig().mode() == CpuFusionMode.OFF) {
            return RegionOptimizationUnitSupport.buildSingleOpUnits(partition, context);
        }
        if (partition.orderedNodeIds().size() <= context.cpuFusionConfig().maxChainNodes()
                && RegionOptimizationUnitSupport.shouldFuseWholePartition(partition, context)) {
            return List.of(RegionOptimizationUnitSupport.buildFusedUnit(partition));
        }
        return buildMixedCpuUnits(partition, context);
    }

    private List<ExecutionUnit> buildMixedCpuUnits(Partition partition, RegionOptimizationContext context) {
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<PartitionValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        List<Integer> ordered = partition.orderedNodeIds();
        int index = 0;
        while (index < ordered.size()) {
            int nodeId = ordered.get(index);
            CompiledNode node = context.compiledNode(nodeId);
            if (!RegionOptimizationUnitSupport.isSubchainFusable(node)) {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(partition, nodeId, node, selected, materialized));
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
            List<RegionValueRef> chainOutputs = RegionOptimizationUnitSupport.unitOutputsForChain(partition, chain, context);
            boolean singlePublishedOutput = chainOutputs.size() == 1
                    && chainOutputs.getFirst().equals(RegionValueRef.ofNode(chain.getLast()));
            if (chain.size() > 1 && singlePublishedOutput) {
                out.add(RegionOptimizationUnitSupport.buildFusedSubchainUnit(partition, chain, context, materialized, chainOutputs));
                index = cursor;
            } else {
                out.add(RegionOptimizationUnitSupport.buildSingleOpUnit(partition, nodeId, node, selected, materialized));
                index++;
            }
        }
        return List.copyOf(out);
    }
}
