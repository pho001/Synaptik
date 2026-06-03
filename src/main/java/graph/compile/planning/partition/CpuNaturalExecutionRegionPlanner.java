package graph.compile.planning.partition;

import config.optimizer.CpuRegionBoundaryPolicy;
import config.optimizer.CpuRegionPolicy;
import graph.CompiledNode;
import graph.compile.planning.value.GraphValueRef;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import operations.Operation;

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
            List<Integer> exactEpilogueNodeIds = exactMatmulEpilogueNodeIds(start, request);
            if (!exactEpilogueNodeIds.isEmpty()) {
                LinkedHashSet<Integer> selected = new LinkedHashSet<>(exactEpilogueNodeIds);
                PartitionCandidate candidate = request.capability().createCandidate(
                        selected,
                        context,
                        request.requiredMaterializedValueRefs()
                );
                PartitionPlan plan = candidate == null ? null : request.capability().createPlan(candidate, context);
                if (candidate != null && plan != null) {
                    Partition partition = PartitionAssembly.acceptedPartition(
                            request,
                            candidate,
                            plan,
                            "cpu-natural-exact-matmul-epilogue",
                            -1,
                            selected.size(),
                            false,
                            null,
                            List.of()
                    );
                    partitions.add(partition);
                    plansByPartitionId.put(partition.partitionId(), plan);
                    decisions.add(partition.debugTrace());
                    index = Math.max(index + 1, indexOfLastNode(nodes, exactEpilogueNodeIds.getLast()) + 1);
                    continue;
                }
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
            PartitionCandidate candidate = request.capability().createCandidate(
                    selected,
                    context,
                    request.requiredMaterializedValueRefs()
            );
            PartitionPlan plan = candidate == null ? null : request.capability().createPlan(candidate, context);
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
                && request.capability().canExecute(node, request.context())
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

    private List<Integer> exactMatmulEpilogueNodeIds(CompiledNode start, PartitionPlanningRequest request) {
        Operation.OpType startOp = opType(start);
        if ((startOp != Operation.OpType.MATMUL && startOp != Operation.OpType.LINEAR)
                || requiredMaterialized(request, start.id())) {
            return List.of();
        }
        CompiledNode firstConsumer = soleConsumer(start.id(), request);
        if (firstConsumer == null || !isSupportedCpuNode(firstConsumer, request)) {
            return List.of();
        }
        if (startOp == Operation.OpType.LINEAR) {
            if (opType(firstConsumer) == Operation.OpType.RELU && reluConsumes(firstConsumer, start.id())) {
                return List.of(start.id(), firstConsumer.id());
            }
            return List.of();
        }
        if (opType(firstConsumer) == Operation.OpType.ADD) {
            return exactMatmulAddReluNodeIds(start, firstConsumer, request);
        }
        if (opType(firstConsumer) == Operation.OpType.RELU && reluConsumes(firstConsumer, start.id())) {
            return List.of(start.id(), firstConsumer.id());
        }
        return List.of();
    }

    private List<Integer> exactMatmulAddReluNodeIds(
            CompiledNode matmul,
            CompiledNode add,
            PartitionPlanningRequest request
    ) {
        if (!addConsumesMatmulWithExternalBias(add, matmul.id()) || requiredMaterialized(request, add.id())) {
            return List.of();
        }
        CompiledNode relu = soleConsumer(add.id(), request);
        if (relu == null
                || !isSupportedCpuNode(relu, request)
                || opType(relu) != Operation.OpType.RELU
                || !reluConsumes(relu, add.id())) {
            return List.of();
        }
        return List.of(matmul.id(), add.id(), relu.id());
    }

    private boolean addConsumesMatmulWithExternalBias(CompiledNode add, int matmulNodeId) {
        if (add == null || add.inputIds().size() != 2) {
            return false;
        }
        int first = add.inputIds().get(0);
        int second = add.inputIds().get(1);
        return (first == matmulNodeId && second != matmulNodeId)
                || (second == matmulNodeId && first != matmulNodeId);
    }

    private boolean reluConsumes(CompiledNode relu, int producerNodeId) {
        return relu != null && relu.inputIds().size() == 1 && relu.inputIds().getFirst() == producerNodeId;
    }

    private CompiledNode soleConsumer(int nodeId, PartitionPlanningRequest request) {
        List<CompiledNode> consumers = request.context().consumersFor(nodeId).stream()
                .filter(java.util.Objects::nonNull)
                .toList();
        return consumers.size() == 1 ? consumers.getFirst() : null;
    }

    private boolean requiredMaterialized(PartitionPlanningRequest request, int nodeId) {
        return request.requiredMaterializedValueRefs().contains(GraphValueRef.node(nodeId));
    }

    private Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }

    private int indexOfLastNode(List<CompiledNode> nodes, int nodeId) {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            if (nodes.get(i).id() == nodeId) {
                return i;
            }
        }
        return -1;
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
