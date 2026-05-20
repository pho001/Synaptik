package graph.compile.planning.partition;

import graph.compile.planning.value.GraphValueRef;

import config.optimizer.CpuRegionBoundaryPolicy;
import config.optimizer.CpuRegionPolicy;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * CPU-specific planner that builds natural execution regions instead of closed accelerator-style partitions.
 *
 * <p>The planner walks the compiled graph in topological order, groups supported CPU nodes into bounded execution
 * regions, and allows later region optimization to split those regions into fused loops and unit kernels. It does not
 * require all supported producers to be absorbed before a node can enter the region; CPU fused-loop planning can treat
 * such values as materialized inputs or unit boundaries later.</p>
 */
public final class CpuNaturalExecutionRegionPlanner implements PartitionPlanner {
    @Override
    public PartitionPlanningResult plan(PartitionPlanningRequest request) {
        if (request == null || request.target() != PartitionTarget.CPU) {
            return PartitionPlanningResult.empty();
        }
        PartitionPlanningContext context = request.context();
        List<CompiledNode> nodes = context.compiledNodes();
        if (nodes.isEmpty()) {
            return PartitionPlanningResult.empty();
        }
        List<Partition> partitions = new ArrayList<>();
        java.util.LinkedHashMap<String, PartitionPlan> plansByPartitionId = new java.util.LinkedHashMap<>();
        List<PartitionDecisionTrace> decisions = new ArrayList<>();
        int index = 0;
        while (index < nodes.size()) {
            CompiledNode start = nodes.get(index);
            if (!isSupportedCpuNode(start, request)) {
                decisions.add(rejectedDecision(request, start, "unsupported-or-non-cpu-node"));
                index++;
                continue;
            }
            LinkedHashSet<Integer> selected = new LinkedHashSet<>();
            int cursor = index;
            while (cursor < nodes.size() && selected.size() < request.policy().maxSearchNodes()) {
                CompiledNode candidate = nodes.get(cursor);
                if (isSupportedCpuNode(candidate, request)) {
                    selected.add(candidate.id());
                    cursor++;
                    continue;
                }
                if (isSkippableExternalValue(candidate)) {
                    cursor++;
                    continue;
                }
                break;
            }
            PartitionCandidate candidate = request.adapter().tryCreateStructuralCandidate(
                    selected,
                    context,
                    request.requiredMaterializedValueRefs()
            );
            PartitionPlan plan = candidate == null ? null : request.adapter().tryCreatePlan(candidate, context);
            if (candidate == null || plan == null) {
                decisions.add(rejectedDecision(request, start, "cpu-natural-region-lowerer-rejected"));
                index++;
                continue;
            }
            Partition partition = buildPartition(request, candidate, plan, selected.size(), cursor < nodes.size());
            partitions.add(partition);
            plansByPartitionId.put(partition.partitionId(), plan);
            decisions.add(partition.debugTrace());
            index = Math.max(cursor, index + 1);
        }
        int accepted = partitions.size();
        return new PartitionPlanningResult(
                partitions,
                plansByPartitionId,
                new PartitionCompileTrace(
                        request.strategy(),
                        request.target(),
                        decisions.size(),
                        accepted,
                        decisions.size() - accepted,
                        decisions
                )
        );
    }

    private boolean isSupportedCpuNode(CompiledNode node, PartitionPlanningRequest request) {
        return node != null
                && node.backend() == request.target().backend()
                && request.adapter().isNodeSupported(node, request.context())
                && allowedByCpuRegionPolicy(node, request);
    }

    private boolean allowedByCpuRegionPolicy(CompiledNode node, PartitionPlanningRequest request) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        if (request.cpuRegionConfig().policy() == CpuRegionPolicy.ELEMENTWISE_ISLANDS
                || request.cpuRegionConfig().boundaryPolicy() == CpuRegionBoundaryPolicy.ELEMENTWISE_ONLY) {
            return node.operation().opType().isFusable();
        }
        return true;
    }

    private boolean isSkippableExternalValue(CompiledNode node) {
        return node != null && node.operation() == null;
    }

    private PartitionDecisionTrace rejectedDecision(
            PartitionPlanningRequest request,
            CompiledNode node,
            String reason
    ) {
        int nodeId = node == null ? -1 : node.id();
        return new PartitionDecisionTrace(
                request.strategy(),
                request.target(),
                Math.max(0, nodeId),
                false,
                reason,
                nodeId < 0 ? List.of() : List.of(nodeId),
                nodeId < 0 ? List.of() : List.of(nodeId),
                opNames(nodeId < 0 ? List.of() : List.of(nodeId), request.context()),
                0L,
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                0,
                false,
                nodeId
        );
    }

    private Partition buildPartition(
            PartitionPlanningRequest request,
            PartitionCandidate candidate,
            PartitionPlan attachedPlan,
            int explored,
            boolean budgetHit
    ) {
        PartitionPlanningContext context = request.context();
        List<PartitionEdge> internalEdges = internalEdges(candidate.orderedNodeIds(), context);
        List<PartitionEdge> boundaryEdges = boundaryEdges(candidate.orderedNodeIds(), context);
        String reason = budgetHit ? "cpu-natural-region-budget-stop" : "cpu-natural-region-frontier-stop";
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
        PartitionDecisionTrace trace = new PartitionDecisionTrace(
                request.strategy(),
                request.target(),
                candidate.anchorNodeId(),
                true,
                reason,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds(),
                opNames(candidate.orderedNodeIds(), context),
                attachedPlan.estimatedWork(),
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                explored,
                budgetHit,
                -1
        );
        return new Partition(
                partitionId(request.target(), candidate.anchorNodeId()),
                ExecutionRegionKind.CPU_EXECUTION,
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
                attachedPlan.estimatedWork(),
                metricsFor(candidate, context),
                request.strategy(),
                trace
        );
    }

    private AcceleratorPartitionScoreModel.CandidateMetrics metricsFor(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        int internalEdgeCount = 0;
        int mergeNodeCount = 0;
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
                Math.max(0, candidate.orderedNodeIds().size() - 1)
        );
    }

    private List<PartitionEdge> internalEdges(List<Integer> nodeIds, PartitionPlanningContext context) {
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

    private List<PartitionEdge> boundaryEdges(List<Integer> nodeIds, PartitionPlanningContext context) {
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

    private List<String> opNames(List<Integer> nodeIds, PartitionPlanningContext context) {
        List<String> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null && node.operation() != null) {
                out.add(node.operation().opType().name());
            }
        }
        return List.copyOf(out);
    }

    private String partitionId(PartitionTarget target, int anchorNodeId) {
        String prefix = target == null ? "cpu" : target.name().toLowerCase(java.util.Locale.ROOT);
        return prefix + "-natural-" + anchorNodeId;
    }
}
