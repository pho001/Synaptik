package graph.compile.planning.region;

import graph.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
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

    static ExecutionUnit buildFusedUnit(Partition partition, RegionOptimizationContext context) {
        List<GraphValueRef> outputs = List.copyOf(partition.outputValueRefs());
        Set<GraphValueRef> outputSet = Set.copyOf(partition.outputValueRefs());
        List<GraphValueRef> virtuals = partition.values().stream()
                .map(PartitionValue::ref)
                .filter(ref -> !outputSet.contains(ref))
                .toList();
        List<GraphValueRef> materializedOutputs = partition.requiredMaterializedValueRefs().stream()
                .filter(outputSet::contains)
                .toList();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-0",
                ExecutionUnitKind.FUSED_ELEMENTWISE,
                partition.target(),
                partition.externalInputNodeIds().stream().map(GraphValueRef::node).toList(),
                outputs,
                materializedOutputs,
                virtuals,
                partition.orderedNodeIds(),
                partition.estimatedWork(),
                partition.externalInputNodeIds(),
                new RegionOptimizationTrace(unitTraceEvents(
                        "fused-whole-partition",
                        partition,
                        partition.orderedNodeIds(),
                        context
                ))
        );
    }

    static List<ExecutionUnit> buildSingleOpUnits(Partition partition, RegionOptimizationContext context) {
        List<ExecutionUnit> out = new ArrayList<>(partition.orderedNodeIds().size());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<GraphValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        for (int nodeId : partition.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            out.add(buildSingleOpUnit(partition, nodeId, node, selected, materialized, context));
        }
        return List.copyOf(out);
    }

    static ExecutionUnit buildSingleOpUnit(
            Partition partition,
            int nodeId,
            CompiledNode node,
            Set<Integer> selected,
            Set<GraphValueRef> materialized,
            RegionOptimizationContext context
    ) {
        List<GraphValueRef> inputRefs = node.inputIds().stream()
                .filter(selected::contains)
                .map(GraphValueRef::node)
                .toList();
        GraphValueRef selfRef = GraphValueRef.node(nodeId);
        List<GraphValueRef> outputRefs = List.of(selfRef);
        boolean continuationOutput = partition.outputValueRefs().contains(selfRef) && !materialized.contains(selfRef);
        List<GraphValueRef> materializedOutputs = materialized.contains(selfRef) ? outputRefs : List.of();
        List<GraphValueRef> virtualOutputs = materialized.contains(selfRef) || continuationOutput ? List.of() : outputRefs;
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
                new RegionOptimizationTrace(unitTraceEvents(
                        "single-op:" + nodeId,
                        partition,
                        List.of(nodeId),
                        context
                ))
        );
    }

    static ExecutionUnit buildFusedSubchainUnit(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context,
            Set<GraphValueRef> materialized,
            List<GraphValueRef> outputRefs
    ) {
        return buildSubchainUnit(
                partition,
                chain,
                context,
                materialized,
                outputRefs,
                ExecutionUnitKind.FUSED_ELEMENTWISE,
                "-fused",
                "fused-subchain:"
        );
    }

    static ExecutionUnit buildEpilogueSubregionUnit(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context,
            Set<GraphValueRef> materialized,
            List<GraphValueRef> outputRefs
    ) {
        return buildSubchainUnit(
                partition,
                chain,
                context,
                materialized,
                outputRefs,
                ExecutionUnitKind.MATMUL_EPILOGUE,
                "-epilogue",
                "matmul-epilogue:"
        );
    }

    private static ExecutionUnit buildSubchainUnit(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context,
            Set<GraphValueRef> materialized,
            List<GraphValueRef> outputRefs,
            ExecutionUnitKind kind,
            String unitSuffix,
            String tracePrefix
    ) {
        Set<Integer> chainSet = Set.copyOf(chain);
        LinkedHashSet<GraphValueRef> inputRefs = new LinkedHashSet<>();
        LinkedHashSet<Integer> externalInputIds = new LinkedHashSet<>();
        for (int nodeId : chain) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!chainSet.contains(inputId)) {
                    inputRefs.add(GraphValueRef.node(inputId));
                    externalInputIds.add(inputId);
                }
            }
        }
        List<GraphValueRef> materializedOutputs = outputRefs.stream()
                .filter(materialized::contains)
                .toList();
        List<GraphValueRef> virtualOutputs = outputRefs.stream()
                .filter(ref -> !materializedOutputs.contains(ref))
                .toList();
        long estimatedWork = chain.stream()
                .map(context::compiledNode)
                .filter(java.util.Objects::nonNull)
                .mapToLong(CompiledNode::flatDataSize)
                .sum();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + chain.getFirst() + unitSuffix,
                kind,
                partition.target(),
                List.copyOf(inputRefs),
                List.copyOf(outputRefs),
                materializedOutputs,
                virtualOutputs,
                List.copyOf(chain),
                Math.max(1L, estimatedWork),
                List.copyOf(externalInputIds),
                new RegionOptimizationTrace(unitTraceEvents(
                        tracePrefix + chain,
                        partition,
                        chain,
                        context
                ))
        );
    }

    private static List<String> unitTraceEvents(
            String baseEvent,
            Partition partition,
            List<Integer> nodeIds,
            RegionOptimizationContext context
    ) {
        ArrayList<String> events = new ArrayList<>();
        events.add(baseEvent);
        if (partition == null || nodeIds == null || nodeIds.isEmpty() || context == null) {
            return List.copyOf(events);
        }
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            Operation.OpType opType = node == null || node.operation() == null ? null : node.operation().opType();
            events.add("region-unit-node:node=" + nodeId
                    + ",op=" + (opType == null ? "UNKNOWN" : opType.name())
                    + ",target=" + partition.target().name());
        }
        return List.copyOf(events);
    }

    static List<GraphValueRef> unitOutputsForChain(
            Partition partition,
            List<Integer> chain,
            RegionOptimizationContext context
    ) {
        Set<Integer> chainSet = Set.copyOf(chain);
        LinkedHashSet<GraphValueRef> outputRefs = new LinkedHashSet<>();
        for (int nodeId : chain) {
            boolean escapesUnit = partition.outputValueRefs().contains(GraphValueRef.node(nodeId));
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
                outputRefs.add(GraphValueRef.node(nodeId));
            }
        }
        if (outputRefs.isEmpty()) {
            outputRefs.add(GraphValueRef.node(chain.getLast()));
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

    static List<Integer> epilogueSpanAt(Partition partition, int startIndex, RegionOptimizationContext context) {
        if (partition == null || context == null || startIndex < 0 || startIndex >= partition.orderedNodeIds().size()) {
            return List.of();
        }
        List<Integer> ordered = partition.orderedNodeIds();
        int firstNodeId = ordered.get(startIndex);
        CompiledNode first = context.compiledNode(firstNodeId);
        if (!isMatmulOrLinear(first)) {
            return List.of();
        }
        if (first.operation().opType() == Operation.OpType.LINEAR) {
            if (first.inputIds().size() < 3 || startIndex + 1 >= ordered.size()) {
                return List.of();
            }
            int activationNodeId = ordered.get(startIndex + 1);
            CompiledNode activation = context.compiledNode(activationNodeId);
            if (isActivation(activation)
                    && activation.inputIds().contains(firstNodeId)
                    && hasNoLaterSelectedConsumer(ordered, startIndex + 1, activationNodeId, context)) {
                return List.of(firstNodeId, activationNodeId);
            }
            return List.of();
        }
        if (startIndex + 2 >= ordered.size()) {
            return List.of();
        }
        int addNodeId = ordered.get(startIndex + 1);
        int activationNodeId = ordered.get(startIndex + 2);
        CompiledNode add = context.compiledNode(addNodeId);
        CompiledNode activation = context.compiledNode(activationNodeId);
        if (add == null || add.operation() == null || add.operation().opType() != Operation.OpType.ADD
                || !add.inputIds().contains(firstNodeId)
                || !isActivation(activation)
                || !activation.inputIds().contains(addNodeId)
                || !hasNoLaterSelectedConsumer(ordered, startIndex + 2, activationNodeId, context)) {
            return List.of();
        }
        return List.of(firstNodeId, addNodeId, activationNodeId);
    }

    private static boolean isMatmulOrLinear(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        return opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR;
    }

    private static boolean isActivation(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        return opType == Operation.OpType.RELU || opType == Operation.OpType.SIGMOID || opType == Operation.OpType.TANH;
    }

    private static boolean hasNoLaterSelectedConsumer(
            List<Integer> ordered,
            int producerIndex,
            int producerNodeId,
            RegionOptimizationContext context
    ) {
        for (int i = producerIndex + 1; i < ordered.size(); i++) {
            CompiledNode candidate = context.compiledNode(ordered.get(i));
            if (candidate != null && candidate.inputIds().contains(producerNodeId)) {
                return false;
            }
        }
        return true;
    }

}
