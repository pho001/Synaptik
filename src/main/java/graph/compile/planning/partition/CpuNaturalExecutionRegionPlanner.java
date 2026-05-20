package graph.compile.planning.partition;

import config.optimizer.CpuRegionBoundaryPolicy;
import config.optimizer.CpuRegionPolicy;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

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
            boolean budgetHit = cursor < nodes.size();
            String reason = budgetHit ? "cpu-natural-region-budget-stop" : "cpu-natural-region-frontier-stop";
            Partition partition = PartitionAssembly.acceptedPartition(
                    request,
                    candidate,
                    plan,
                    reason,
                    -1,
                    selected.size(),
                    budgetHit,
                    null,
                    List.of()
            );
            partitions.add(partition);
            plansByPartitionId.put(partition.partitionId(), plan);
            decisions.add(partition.debugTrace());
            index = Math.max(cursor, index + 1);
        }
        int accepted = partitions.size();
        return new PartitionPlanningResult(
                partitions,
                plansByPartitionId,
                PartitionCompileTrace.forJob(request.strategy(), request.target(), decisions)
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
                PartitionAssembly.opNames(nodeId < 0 ? List.of() : List.of(nodeId), request.context()),
                0L,
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                0,
                false,
                nodeId
        );
    }

}
