package graph.compile.planning.partition;

import graph.model.CompiledNode;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;
import graph.compile.planning.value.GraphValueRef;
import graph.execution.trace.PartitionDecisionTrace;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

final class PartitionAssembly {
    private PartitionAssembly() {
    }

    static Partition acceptedPartition(
            PartitionPlanningRequest request,
            PartitionCandidate candidate,
            PartitionPlan attachedPlan,
            String reason,
            int rejectedNodeId,
            int explored
    ) {
        return acceptedPartition(request, candidate, attachedPlan, reason, rejectedNodeId, explored, false, null, List.of());
    }

    static Partition acceptedPartition(
            PartitionPlanningRequest request,
            PartitionCandidate candidate,
            PartitionPlan attachedPlan,
            String reason,
            int rejectedNodeId,
            int explored,
            boolean budgetHit,
            AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary,
            List<PartitionDecisionTrace.CandidateCostTrace> finalists
    ) {
        PartitionPlanningContext context = request.context();
        List<PartitionEdge> internalEdges = internalEdges(candidate.orderedNodeIds(), context);
        List<PartitionEdge> boundaryEdges = boundaryEdges(candidate.orderedNodeIds(), context);
        List<PartitionBoundaryReason> boundaryReasons = boundaryEdges.stream()
                .map(ignored -> PartitionBoundaryReason.fromReason(reason))
                .toList();
        List<PartitionValue> values = candidate.orderedNodeIds().stream()
                .map(nodeId -> new PartitionValue(GraphValueRef.node(nodeId), nodeId))
                .toList();
        List<GraphValueRef> outputValueRefs = candidate.outputNodeIds().stream().map(GraphValueRef::node).toList();
        List<GraphValueRef> requiredMaterialized = candidate.outputNodeIds().stream()
                .map(GraphValueRef::node)
                .filter(request.requiredMaterializedValueRefs()::contains)
                .toList();
        long estimatedWork = attachedPlan == null ? 0L : attachedPlan.estimatedWork();
        PartitionDecisionTrace trace = new PartitionDecisionTrace(
                request.strategy(),
                request.target(),
                candidate.anchorNodeId(),
                true,
                reason,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds(),
                opNames(candidate.orderedNodeIds(), context),
                estimatedWork,
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                explored,
                budgetHit,
                rejectedNodeId,
                costSummary,
                finalists
        );
        return new Partition(
                partitionId(request.target(), candidate.anchorNodeId()),
                request.target(),
                candidate.orderedNodeIds(),
                values,
                internalEdges,
                candidate.externalInputIds(),
                outputValueRefs,
                candidate.anchorNodeId(),
                requiredMaterialized,
                boundaryEdges,
                boundaryReasons,
                estimatedWork,
                metricsFor(candidate, context),
                request.strategy(),
                trace
        );
    }

    static List<String> opNames(List<Integer> nodeIds, PartitionPlanningContext context) {
        List<String> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null && node.operation() != null) {
                out.add(node.operation().opType().name());
            }
        }
        return List.copyOf(out);
    }

    static AcceleratorPartitionScoreModel.CandidateMetrics metricsFor(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        int internalEdgeCount = 0;
        int mergeNodeCount = 0;
        int tailDepth = Math.max(0, candidate.orderedNodeIds().size() - 1);
        Set<Integer> selected = Set.copyOf(candidate.orderedNodeIds());
        for (int nodeId : candidate.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            int selectedInputs = 0;
            for (int inputId : node.inputIds()) {
                if (selected.contains(inputId)) {
                    internalEdgeCount++;
                    selectedInputs++;
                }
            }
            if (selectedInputs > 1) {
                mergeNodeCount++;
            }
        }
        return new AcceleratorPartitionScoreModel.CandidateMetrics(
                candidate.orderedNodeIds().size(),
                internalEdgeCount,
                candidate.externalInputIds().size(),
                mergeNodeCount,
                tailDepth
        );
    }

    private static List<PartitionEdge> internalEdges(List<Integer> nodeIds, PartitionPlanningContext context) {
        Set<Integer> selected = Set.copyOf(nodeIds);
        List<PartitionEdge> edges = new ArrayList<>();
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (selected.contains(inputId)) {
                    edges.add(new PartitionEdge(inputId, nodeId));
                }
            }
        }
        return List.copyOf(edges);
    }

    private static List<PartitionEdge> boundaryEdges(List<Integer> nodeIds, PartitionPlanningContext context) {
        Set<Integer> selected = Set.copyOf(nodeIds);
        List<PartitionEdge> edges = new ArrayList<>();
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!selected.contains(inputId)) {
                    edges.add(new PartitionEdge(inputId, nodeId));
                }
            }
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer != null && !selected.contains(consumer.id())) {
                    edges.add(new PartitionEdge(nodeId, consumer.id()));
                }
            }
        }
        return List.copyOf(edges);
    }

    private static String partitionId(PartitionTarget target, int anchorNodeId) {
        String prefix = target == null ? "partition" : target.name().toLowerCase(Locale.ROOT);
        return prefix + "-" + anchorNodeId;
    }
}
