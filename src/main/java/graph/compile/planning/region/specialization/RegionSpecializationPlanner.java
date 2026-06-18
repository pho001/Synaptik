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
            if (!coversWholePartition(candidate, partition)) {
                events.add("specialization-candidate-rejected:kind=" + candidate.kind().name()
                        + ",nodes=" + candidate.orderedNodeIds()
                        + ",reason=partial-region-specialization-not-yet-supported");
                continue;
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

    private static ExecutionUnit buildSpecializedUnit(
            Partition partition,
            RegionOptimizationContext context,
            RegionSpecializationCandidate candidate,
            String acceptanceReason
    ) {
        GraphValueRef output = candidate.outputValueRef();
        List<GraphValueRef> outputRefs = List.of(output);
        List<GraphValueRef> materializedOutputs = partition.requiredMaterializedValueRefs().contains(output)
                ? outputRefs
                : List.of();
        Set<GraphValueRef> publishedOutputs = Set.copyOf(outputRefs);
        List<GraphValueRef> virtualOutputs = candidate.orderedNodeIds().stream()
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
                candidate.orderedNodeIds(),
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
}
