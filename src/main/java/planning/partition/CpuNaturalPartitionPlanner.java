package planning.partition;

import config.optimizer.CpuPartitionBoundaryPolicy;
import config.optimizer.CpuPartitionPolicy;
import graph.model.CompiledNode;
import planning.value.GraphValueRef;
import trace.compile.PartitionCompileTrace;
import trace.compile.PartitionDecisionTrace;
import operations.Operation;
import operations.linalg.linear;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * CPU-specific planner that builds natural execution partitions instead of closed accelerator-style partitions.
 *
 * <p>The planner walks the compiled graph in topological order, groups supported CPU nodes into bounded execution
 * partitions, and allows later partition planning to split those partitions into fused loops and unit kernels. It does not
 * require all supported producers to be absorbed before a node can enter the partition; CPU fused-loop planning can treat
 * such values as materialized inputs or unit boundaries later.</p>
 */
public final class CpuNaturalPartitionPlanner implements PartitionPlanner {
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
                decisions.add(rejectedDecision(request, start, "cpu-natural-partition-lowerer-rejected"));
                index++;
                continue;
            }
            boolean budgetHit = cursor < nodes.size();
            String reason = budgetHit ? "cpu-natural-partition-budget-stop" : "cpu-natural-partition-frontier-stop";
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
                PartitionCompileTrace.forJob(request.strategy().name(), request.target().name(), decisions)
        );
    }

    private boolean isSupportedCpuNode(CompiledNode node, PartitionPlanningRequest request) {
        return node != null
                && node.backend() == request.target().backend()
                && request.capability().canExecute(node, request.context())
                && allowedByCpuPartitionPolicy(node, request);
    }

    private boolean allowedByCpuPartitionPolicy(CompiledNode node, PartitionPlanningRequest request) {
        if (node == null || node.operation() == null || node.operation().opType() == null) {
            return false;
        }
        if (request.cpuPartitionConfig().policy() == CpuPartitionPolicy.ELEMENTWISE_ISLANDS
                || request.cpuPartitionConfig().boundaryPolicy() == CpuPartitionBoundaryPolicy.ELEMENTWISE_ONLY) {
            return node.operation().isFusable();
        }
        return true;
    }

    private List<Integer> exactMatmulEpilogueNodeIds(CompiledNode start, PartitionPlanningRequest request) {
        Operation.OpType startOp = opType(start);
        if ((startOp != Operation.OpType.MATMUL && startOp != Operation.OpType.LINEAR)
                || requiredMaterialized(request, start.id())) {
            return List.of();
        }
        if (startOp == Operation.OpType.LINEAR) {
            return exactLinearEpilogueNodeIds(start, request);
        }
        CompiledNode firstConsumer = soleConsumer(start.id(), request);
        if (firstConsumer == null || !isSupportedCpuNode(firstConsumer, request)) {
            return List.of();
        }
        if (opType(firstConsumer) == Operation.OpType.ADD) {
            return exactMatmulAddEpilogueNodeIds(start, firstConsumer, request);
        }
        if (opType(firstConsumer) == Operation.OpType.RELU
                && reluConsumes(firstConsumer, start.id())
                && !hasFusableCpuContinuation(firstConsumer, request)) {
            return List.of(start.id(), firstConsumer.id());
        }
        return List.of();
    }

    private List<Integer> exactLinearEpilogueNodeIds(CompiledNode linearNode, PartitionPlanningRequest request) {
        if (!(linearNode.operation() instanceof linear linearOp) || !linearOp.hasBias()) {
            return List.of();
        }
        CompiledNode firstConsumer = soleConsumer(linearNode.id(), request);
        if (firstConsumer == null) {
            return List.of(linearNode.id());
        }
        if (!isSupportedCpuNode(firstConsumer, request)) {
            return List.of();
        }
        if (opType(firstConsumer) == Operation.OpType.RELU
                && reluConsumes(firstConsumer, linearNode.id())
                && !hasFusableCpuContinuation(firstConsumer, request)) {
            return List.of(linearNode.id(), firstConsumer.id());
        }
        if (!hasFusableCpuContinuation(linearNode, request)) {
            return List.of(linearNode.id());
        }
        return List.of();
    }

    private List<Integer> exactMatmulAddEpilogueNodeIds(
            CompiledNode matmul,
            CompiledNode add,
            PartitionPlanningRequest request
    ) {
        if (!addConsumesMatmulWithExternalBias(add, matmul.id()) || requiredMaterialized(request, add.id())) {
            return List.of();
        }
        CompiledNode relu = soleConsumer(add.id(), request);
        if (relu != null
                && isSupportedCpuNode(relu, request)
                && opType(relu) == Operation.OpType.RELU
                && reluConsumes(relu, add.id())) {
            if (hasFusableCpuContinuation(relu, request)) {
                return List.of();
            }
            return List.of(matmul.id(), add.id(), relu.id());
        }
        if (!hasFusableCpuContinuation(add, request)) {
            return List.of(matmul.id(), add.id());
        }
        return List.of();
    }

    private boolean hasFusableCpuContinuation(CompiledNode node, PartitionPlanningRequest request) {
        if (node == null) {
            return false;
        }
        for (CompiledNode consumer : request.context().consumersFor(node.id())) {
            if (consumer != null
                    && consumer.inputIds().contains(node.id())
                    && isSupportedCpuNode(consumer, request)
                    && consumer.operation() != null
                    && consumer.operation().isFusable()) {
                return true;
            }
        }
        return false;
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
                request.strategy().name(),
                request.target().name(),
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
