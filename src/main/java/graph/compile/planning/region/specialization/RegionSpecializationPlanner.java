package graph.compile.planning.region.specialization;

import graph.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.ExecutionUnitKind;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.region.RegionOptimizationTrace;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Plans graph-level specialized primitive execution units before memory planning.
 */
public final class RegionSpecializationPlanner {
    private RegionSpecializationPlanner() {
    }

    /**
     * Tries to replace a structurally matched region with specialized primitive units.
     *
     * <p>The first pass is deliberately narrow: accepted candidates must cover the full partition. Partial-region
     * packing can be added once specialized primitive lowering is executable for mixed regions.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @param capability backend acceptance hook
     * @return specialization result with trace events
     */
    public static RegionSpecializationResult tryBuildUnits(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCapability capability
    ) {
        if (partition == null || context == null) {
            return RegionSpecializationResult.empty();
        }
        RegionSpecializationCapability effectiveCapability = capability == null
                ? RegionSpecializationCapability.rejecting()
                : capability;
        ArrayList<String> events = new ArrayList<>();
        List<RegionSpecializationCandidate> candidates = specializationCandidates(partition, context);
        for (RegionSpecializationCandidate candidate : candidates) {
            events.add("specialization-candidate-found:kind=" + candidate.kind().name()
                    + ",nodes=" + candidate.orderedNodeIds()
                    + ",output=" + candidate.outputValueRef()
                    + ",summary=" + candidate.summary());
            RegionSpecializationDecision decision = effectiveCapability.evaluate(partition.target(), candidate);
            if (!decision.accepted()) {
                events.add("specialization-candidate-rejected:kind=" + candidate.kind().name()
                        + ",nodes=" + candidate.orderedNodeIds()
                        + ",reason=" + decision.reason());
                continue;
            }
            if (!coversWholePartition(candidate, partition)
                    && !canBuildWithInputProducerUnits(candidate, partition, context)) {
                events.add("specialization-candidate-rejected:kind=" + candidate.kind().name()
                        + ",nodes=" + candidate.orderedNodeIds()
                        + ",reason=partial-region-specialization-not-yet-supported");
                continue;
            }
            if (!coversWholePartition(candidate, partition)) {
                List<ExecutionUnit> units = buildWithInputProducerUnits(partition, context, candidate, decision.reason());
                events.add("specialization-candidate-accepted:kind=" + candidate.kind().name()
                        + ",unit=" + units.getLast().unitId()
                        + ",reason=" + decision.reason()
                        + ",input-producer-units=" + (units.size() - 1));
                return new RegionSpecializationResult(true, units, events);
            }
            ExecutionUnit unit = buildSpecializedUnit(partition, context, candidate, decision.reason());
            events.add("specialization-candidate-accepted:kind=" + candidate.kind().name()
                    + ",unit=" + unit.unitId()
                    + ",reason=" + decision.reason());
            return new RegionSpecializationResult(true, List.of(unit), events);
        }
        return new RegionSpecializationResult(false, List.of(), events);
    }

    private static boolean coversWholePartition(
            RegionSpecializationCandidate candidate,
            Partition partition
    ) {
        return candidate.orderedNodeIds().equals(partition.orderedNodeIds());
    }

    private static boolean canBuildWithInputProducerUnits(
            RegionSpecializationCandidate candidate,
            Partition partition,
            RegionOptimizationContext context
    ) {
        if (candidate.kind() != RegionSpecializationKind.SDPA_BACKWARD) {
            return false;
        }
        Set<Integer> partitionNodeIds = Set.copyOf(partition.orderedNodeIds());
        Set<Integer> candidateNodeIds = Set.copyOf(candidate.orderedNodeIds());
        if (!partitionNodeIds.containsAll(candidateNodeIds)) {
            return false;
        }
        int anchorIndex = partition.orderedNodeIds().indexOf(candidate.anchorNodeId());
        if (anchorIndex < 0) {
            return false;
        }
        for (int candidateNodeId : candidate.orderedNodeIds()) {
            if (partition.orderedNodeIds().indexOf(candidateNodeId) > anchorIndex) {
                return false;
            }
        }
        Set<Integer> omittedNodeIds = partition.orderedNodeIds().stream()
                .filter(nodeId -> !candidateNodeIds.contains(nodeId))
                .collect(java.util.stream.Collectors.toSet());
        for (int omittedNodeId : omittedNodeIds) {
            if (partition.orderedNodeIds().indexOf(omittedNodeId) > anchorIndex) {
                return false;
            }
        }
        Set<Integer> neededProducerNodeIds = new LinkedHashSet<>();
        ArrayList<Integer> worklist = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .filter(omittedNodeIds::contains)
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
        for (int index = 0; index < worklist.size(); index++) {
            int nodeId = worklist.get(index);
            if (!neededProducerNodeIds.add(nodeId)) {
                continue;
            }
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            node.inputIds().stream()
                    .filter(omittedNodeIds::contains)
                    .forEach(worklist::add);
        }
        return neededProducerNodeIds.equals(omittedNodeIds);
    }

    private static List<ExecutionUnit> buildWithInputProducerUnits(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCandidate candidate,
            String acceptanceReason
    ) {
        Set<Integer> candidateNodeIds = Set.copyOf(candidate.orderedNodeIds());
        Set<Integer> selected = Set.copyOf(partition.orderedNodeIds());
        Set<GraphValueRef> materialized = Set.copyOf(partition.requiredMaterializedValueRefs());
        int lastInputProducerIndex = lastInputProducerIndex(partition, candidateNodeIds);
        List<Integer> specializedNodeIds = candidate.orderedNodeIds().stream()
                .filter(nodeId -> partition.orderedNodeIds().indexOf(nodeId) > lastInputProducerIndex)
                .toList();
        ArrayList<ExecutionUnit> units = new ArrayList<>();
        for (int nodeId : partition.orderedNodeIds()) {
            if (candidateNodeIds.contains(nodeId)) {
                if (partition.orderedNodeIds().indexOf(nodeId) > lastInputProducerIndex) {
                    if (nodeId == candidate.anchorNodeId()) {
                        units.add(buildSpecializedUnit(
                                partition,
                                context,
                                candidate,
                                acceptanceReason,
                                specializedNodeIds
                        ));
                    }
                    continue;
                }
            }
            units.add(buildInputProducerUnit(partition, context, nodeId, selected, materialized));
        }
        return List.copyOf(units);
    }

    private static int lastInputProducerIndex(Partition partition, Set<Integer> candidateNodeIds) {
        int lastInputProducerIndex = -1;
        for (int index = 0; index < partition.orderedNodeIds().size(); index++) {
            if (!candidateNodeIds.contains(partition.orderedNodeIds().get(index))) {
                lastInputProducerIndex = index;
            }
        }
        return lastInputProducerIndex;
    }

    private static ExecutionUnit buildInputProducerUnit(
            Partition partition,
            RegionOptimizationContext context,
            int nodeId,
            Set<Integer> selected,
            Set<GraphValueRef> materialized
    ) {
        CompiledNode node = context.compiledNode(nodeId);
        if (node == null) {
            throw new IllegalStateException("Missing compiled node for SDPA_BACKWARD input producer nodeId=" + nodeId);
        }
        GraphValueRef selfRef = GraphValueRef.node(nodeId);
        List<GraphValueRef> outputRefs = List.of(selfRef);
        boolean continuationOutput = partition.outputValueRefs().contains(selfRef) && !materialized.contains(selfRef);
        List<GraphValueRef> materializedOutputs = materialized.contains(selfRef) ? outputRefs : List.of();
        List<GraphValueRef> virtualOutputs = materialized.contains(selfRef) || continuationOutput ? List.of() : outputRefs;
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + nodeId + "-sdpa-input",
                ExecutionUnitKind.UNIT_KERNEL,
                partition.target(),
                node.inputIds().stream()
                        .filter(selected::contains)
                        .map(GraphValueRef::node)
                        .toList(),
                outputRefs,
                materializedOutputs,
                virtualOutputs,
                List.of(nodeId),
                Math.max(1L, node.flatDataSize()),
                node.inputIds().stream()
                        .filter(inputId -> !selected.contains(inputId))
                        .toList(),
                new RegionOptimizationTrace(inputProducerTraceEvents(partition, context, nodeId))
        );
    }

    private static ExecutionUnit buildSpecializedUnit(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCandidate candidate,
            String acceptanceReason
    ) {
        return buildSpecializedUnit(partition, context, candidate, acceptanceReason, candidate.orderedNodeIds());
    }

    private static ExecutionUnit buildSpecializedUnit(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCandidate candidate,
            String acceptanceReason,
            List<Integer> executionNodeIds
    ) {
        GraphValueRef output = candidate.outputValueRef();
        List<GraphValueRef> outputRefs = List.of(output);
        List<GraphValueRef> materializedOutputs = partition.requiredMaterializedValueRefs().contains(output)
                ? outputRefs
                : List.of();
        Set<GraphValueRef> publishedOutputs = Set.copyOf(outputRefs);
        List<Integer> orderedNodeIds = List.copyOf(executionNodeIds);
        List<GraphValueRef> virtualOutputs = orderedNodeIds.stream()
                .map(GraphValueRef::node)
                .filter(ref -> !publishedOutputs.contains(ref))
                .toList();
        return new ExecutionUnit(
                partition.partitionId() + "-unit-" + candidate.anchorNodeId() + "-" + unitSuffix(candidate),
                ExecutionUnitKind.SPECIALIZED_PRIMITIVE,
                partition.target(),
                candidate.inputValueRefs(),
                outputRefs,
                materializedOutputs,
                virtualOutputs,
                orderedNodeIds,
                estimatedWork(candidate, context),
                requiredPreparedInputNodeIds(candidate),
                new RegionOptimizationTrace(unitTraceEvents(partition, context, candidate, acceptanceReason)),
                candidate
        );
    }

    private static List<RegionSpecializationCandidate> specializationCandidates(
            Partition partition,
            RegionOptimizationContext context
    ) {
        ArrayList<RegionSpecializationCandidate> out = new ArrayList<>();
        out.addAll(MseLossSpecializationDetector.findCandidates(partition, context));
        out.addAll(SdpaBackwardSpecializationDetector.findCandidates(partition, context));
        out.addAll(MatmulBiasReluSpecializationDetector.findCandidates(partition, context));
        out.addAll(MatmulBiasSpecializationDetector.findCandidates(partition, context));
        out.addAll(MatmulReluSpecializationDetector.findCandidates(partition, context));
        return List.copyOf(out);
    }

    private static String unitSuffix(RegionSpecializationCandidate candidate) {
        return candidate.kind().name().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    private static long estimatedWork(
            RegionSpecializationCandidate candidate,
            RegionOptimizationContext context
    ) {
        long work = candidate.orderedNodeIds().stream()
                .map(context::compiledNode)
                .filter(java.util.Objects::nonNull)
                .mapToLong(CompiledNode::flatDataSize)
                .sum();
        return Math.max(1L, work);
    }

    private static List<Integer> requiredPreparedInputNodeIds(RegionSpecializationCandidate candidate) {
        LinkedHashSet<Integer> out = new LinkedHashSet<>();
        for (GraphValueRef inputValueRef : candidate.inputValueRefs()) {
            if (inputValueRef != null && inputValueRef.nodeId() >= 0) {
                out.add(inputValueRef.nodeId());
            }
        }
        return List.copyOf(out);
    }

    private static List<String> unitTraceEvents(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCandidate candidate,
            String acceptanceReason
    ) {
        ArrayList<String> events = new ArrayList<>();
        events.add("specialized-primitive:" + candidate.kind().name());
        events.add("specialized-primitive-accepted:" + acceptanceReason);
        events.add("specialized-primitive-summary:" + candidate.summary());
        for (int nodeId : candidate.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            Operation.OpType opType = node == null || node.operation() == null ? null : node.operation().opType();
            events.add("region-unit-node:node=" + nodeId
                    + ",op=" + (opType == null ? "UNKNOWN" : opType.name())
                    + ",target=" + partition.target().name());
        }
        return List.copyOf(events);
    }

    private static List<String> inputProducerTraceEvents(
            Partition partition,
            RegionOptimizationContext context,
            int nodeId
    ) {
        CompiledNode node = context.compiledNode(nodeId);
        Operation.OpType opType = node == null || node.operation() == null ? null : node.operation().opType();
        return List.of(
                "sdpa-backward-input-producer:" + nodeId,
                "region-unit-node:node=" + nodeId
                        + ",op=" + (opType == null ? "UNKNOWN" : opType.name())
                        + ",target=" + partition.target().name()
        );
    }
}
