package graph.optimizer.region;

import graph.CompiledNode;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import operations.Operation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

final class RegionOptimizationUnitSupport {
    private RegionOptimizationUnitSupport() {
    }

    static boolean shouldFuseWholePartition(Partition partition, RegionOptimizationContext context) {
        if (partition == null || partition.orderedNodeIds().size() < 2) {
            return false;
        }
        if (partition.outputValueRefs().size() != 1) {
            return false;
        }
        if (partition.target() == PartitionTarget.NONE) {
            return false;
        }
        for (int nodeId : partition.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                return false;
            }
            Operation.OpType opType = node.operation().opType();
            if (opType == null || !opType.isFusable()) {
                return false;
            }
        }
        return true;
    }

    static ExecutionUnit buildFusedUnit(Partition partition) {
        List<RegionValueRef> outputs = partition.outputValueRefs().stream().map(RegionOptimizationUnitSupport::toRegionValueRef).toList();
        Set<PartitionValueRef> outputSet = Set.copyOf(partition.outputValueRefs());
        List<RegionValueRef> virtuals = partition.values().stream()
                .map(PartitionValue::ref)
                .filter(ref -> !outputSet.contains(ref))
                .map(RegionOptimizationUnitSupport::toRegionValueRef)
                .toList();
        List<RegionValueRef> materializedOutputs = partition.requiredMaterializedValueRefs().stream()
                .filter(outputSet::contains)
                .map(RegionOptimizationUnitSupport::toRegionValueRef)
                .toList();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-0",
                ExecutionUnitKind.FUSED_ELEMENTWISE,
                partition.target(),
                partition.externalInputNodeIds().stream().map(RegionValueRef::ofNode).toList(),
                outputs,
                materializedOutputs,
                virtuals,
                partition.orderedNodeIds(),
                partition.estimatedWork(),
                partition.externalInputNodeIds(),
                new RegionOptimizationTrace(List.of("fused-whole-partition"))
        );
    }

    static List<ExecutionUnit> buildSingleOpUnits(Partition partition, RegionOptimizationContext context) {
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<PartitionValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        for (int nodeId : partition.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            out.add(buildSingleOpUnit(partition, nodeId, node, selected, materialized));
        }
        return List.copyOf(out);
    }

    static ExecutionUnit buildSingleOpUnit(
            Partition partition,
            int nodeId,
            CompiledNode node,
            Set<Integer> selected,
            Set<PartitionValueRef> materialized
    ) {
        List<RegionValueRef> inputRefs = node.inputIds().stream()
                .filter(selected::contains)
                .map(RegionValueRef::ofNode)
                .toList();
        PartitionValueRef selfRef = PartitionValueRef.ofNode(nodeId);
        List<RegionValueRef> outputRefs = List.of(toRegionValueRef(selfRef));
        boolean continuationOutput = partition.outputValueRefs().contains(selfRef) && !materialized.contains(selfRef);
        List<RegionValueRef> materializedOutputs = materialized.contains(selfRef) ? outputRefs : List.of();
        List<RegionValueRef> virtualOutputs = materialized.contains(selfRef) || continuationOutput ? List.of() : outputRefs;
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + nodeId,
                ExecutionUnitKind.UNIT_KERNEL,
                partition.target(),
                inputRefs,
                outputRefs,
                materializedOutputs,
                virtualOutputs,
                List.of(nodeId),
                Math.max(1L, node.flatDataSize()),
                node.inputIds().stream().filter(inputId -> !selected.contains(inputId)).toList(),
                new RegionOptimizationTrace(List.of("single-op:" + nodeId))
        );
    }

    static ExecutionUnit buildFusedSubchainUnit(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context,
            Set<PartitionValueRef> materialized,
            List<RegionValueRef> outputRefs
    ) {
        Set<Integer> chainSet = Set.copyOf(chain);
        LinkedHashSet<RegionValueRef> inputRefs = new LinkedHashSet<>();
        LinkedHashSet<Integer> externalInputIds = new LinkedHashSet<>();
        for (int nodeId : chain) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!chainSet.contains(inputId)) {
                    inputRefs.add(RegionValueRef.ofNode(inputId));
                    externalInputIds.add(inputId);
                }
            }
        }
        List<RegionValueRef> materializedOutputs = outputRefs.stream()
                .filter(ref -> materialized.contains(PartitionValueRef.ofNode(nodeIdFromRef(ref))))
                .toList();
        List<RegionValueRef> virtualOutputs = outputRefs.stream()
                .filter(ref -> !materializedOutputs.contains(ref))
                .toList();
        long estimatedWork = chain.stream()
                .map(context::compiledNode)
                .filter(java.util.Objects::nonNull)
                .mapToLong(CompiledNode::flatDataSize)
                .sum();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + chain.getFirst() + "-fused",
                ExecutionUnitKind.FUSED_ELEMENTWISE,
                partition.target(),
                List.copyOf(inputRefs),
                List.copyOf(outputRefs),
                materializedOutputs,
                virtualOutputs,
                List.copyOf(chain),
                Math.max(1L, estimatedWork),
                List.copyOf(externalInputIds),
                new RegionOptimizationTrace(List.of("fused-subchain:" + chain))
        );
    }

    static List<RegionValueRef> unitOutputsForChain(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context
    ) {
        Set<Integer> chainSet = Set.copyOf(chain);
        LinkedHashSet<RegionValueRef> outputRefs = new LinkedHashSet<>();
        for (int nodeId : chain) {
            boolean escapesUnit = partition.outputValueRefs().contains(PartitionValueRef.ofNode(nodeId));
            if (!escapesUnit) {
                for (int candidateId : partition.orderedNodeIds()) {
                    if (chainSet.contains(candidateId)) {
                        continue;
                    }
                    CompiledNode candidate = context.compiledNode(candidateId);
                    if (candidate != null && candidate.inputIds().contains(nodeId)) {
                        escapesUnit = true;
                        break;
                    }
                }
            }
            if (escapesUnit) {
                outputRefs.add(RegionValueRef.ofNode(nodeId));
            }
        }
        if (outputRefs.isEmpty()) {
            outputRefs.add(RegionValueRef.ofNode(chain.getLast()));
        }
        return List.copyOf(outputRefs);
    }

    static boolean isSubchainFusable(CompiledNode node) {
        return node != null
                && node.operation() != null
                && node.operation().opType() != null
                && node.operation().opType().isFusable();
    }

    static boolean consumesUnitOutput(CompiledNode candidate, List<Integer> chain) {
        if (candidate == null || candidate.inputIds().isEmpty()) {
            return false;
        }
        int lastNodeId = chain.getLast();
        return candidate.inputIds().contains(lastNodeId);
    }

    static RegionValueRef toRegionValueRef(PartitionValueRef ref) {
        return RegionValueRef.ofNode(ref.producerNodeId());
    }

    private static int nodeIdFromRef(RegionValueRef ref) {
        if (ref == null || ref.valueId() == null || !ref.valueId().startsWith("node-")) {
            return -1;
        }
        try {
            return Integer.parseInt(ref.valueId().substring("node-".length()));
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }
}
