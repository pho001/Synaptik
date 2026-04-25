package graph.optimizer.partition;

import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.execution.trace.AcceleratorPartitionCompileTrace;
import graph.execution.trace.AcceleratorPartitionDecisionTrace;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class GreedyMaxRegionPartitionPlanner implements AcceleratorPartitionPlanner {
    private record AttemptResult(
            AcceleratorPartitionPlan plan,
            AcceleratorPartitionDecisionTrace decision
    ) {
    }

    private record ExpansionResult(
            LinkedHashSet<Integer> selectedNodeIds,
            AcceleratorStructuralCandidate candidate,
            AcceleratorPartitionPlan plan
    ) {
    }

    private record Rejection(
            String reason,
            int nodeId
    ) {
    }

    @Override
    public PartitionPlanningResult plan(PartitionPlanningRequest request) {
        if (request == null || request.target().isNone()) {
            return PartitionPlanningResult.empty();
        }
        BackendPrepareContext context = request.context();
        List<CompiledNode> nodes = context.compiledNodes();
        if (nodes.isEmpty()) {
            return PartitionPlanningResult.empty();
        }
        boolean[] covered = new boolean[nodes.size()];
        List<AcceleratorPartitionPlan> plans = new ArrayList<>();
        List<AcceleratorPartitionDecisionTrace> decisions = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            CompiledNode current = nodes.get(i);
            if (current.backend() != request.target().backend()) {
                continue;
            }
            if (covered[i]) {
                decisions.add(new AcceleratorPartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        i,
                        false,
                        "covered-by-earlier-partition",
                        List.of(i),
                        List.of(i),
                        opNames(List.of(i), context),
                        0L,
                        Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY,
                        0,
                        false,
                        -1
                ));
                continue;
            }
            AttemptResult attempt = tryBuildPlan(i, request, covered);
            decisions.add(attempt.decision());
            if (attempt.plan() == null) {
                continue;
            }
            for (int nodeId : attempt.plan().nodeIds()) {
                covered[nodeId] = true;
            }
            plans.add(attempt.plan());
        }
        int accepted = (int) decisions.stream().filter(AcceleratorPartitionDecisionTrace::accepted).count();
        return new PartitionPlanningResult(
                plans,
                new AcceleratorPartitionCompileTrace(
                        request.strategy(),
                        request.target(),
                        decisions.size(),
                        accepted,
                        decisions.size() - accepted,
                        decisions
                )
        );
    }

    private AttemptResult tryBuildPlan(int startIndex, PartitionPlanningRequest request, boolean[] covered) {
        BackendPrepareContext context = request.context();
        CompiledNode start = context.compiledNodes().get(startIndex);
        if (!request.adapter().canSeed(start, context)) {
            return new AttemptResult(
                    null,
                    new AcceleratorPartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "unsupported-start-node",
                            List.of(start.id()),
                            List.of(start.id()),
                            opNames(List.of(start.id()), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            0,
                            false,
                            -1
                    )
            );
        }

        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        Rejection seedRejection = absorbWithProducerClosure(start.id(), selected, request, covered);
        if (seedRejection != null) {
            return new AttemptResult(
                    null,
                    new AcceleratorPartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            seedRejection.reason(),
                            List.of(start.id()),
                            List.of(start.id()),
                            opNames(List.of(start.id()), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            0,
                            false,
                            seedRejection.nodeId()
                    )
            );
        }
        AcceleratorStructuralCandidate bestCandidate = request.adapter().tryCreateStructuralCandidate(selected, context);
        AcceleratorPartitionPlan bestPlan = bestCandidate == null ? null : request.adapter().tryCreatePlan(bestCandidate, context);
        if (bestPlan == null) {
            return new AttemptResult(
                    null,
                    new AcceleratorPartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "lowerer-rejected",
                            List.of(start.id()),
                            bestCandidate == null ? List.of(start.id()) : bestCandidate.orderedNodeIds(),
                            opNames(bestCandidate == null ? List.of(start.id()) : bestCandidate.orderedNodeIds(), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            0,
                            false,
                            -1
                    )
            );
        }

        int explored = 0;
        boolean budgetHit = false;
        Rejection lastRejection = new Rejection("frontier-exhausted", -1);
        while (true) {
            List<Integer> frontier = expandableConsumers(selected, request, covered);
            if (frontier.isEmpty()) {
                break;
            }
            boolean progressed = false;
            for (int frontierNodeId : frontier) {
                explored++;
                if (selected.size() >= request.policy().maxSearchNodes()) {
                    budgetHit = true;
                    lastRejection = new Rejection("max-search-nodes", frontierNodeId);
                    break;
                }
                ExpansionResult expanded = tryExpand(selected, frontierNodeId, request, covered);
                if (expanded != null) {
                    selected = expanded.selectedNodeIds();
                    bestCandidate = expanded.candidate();
                    bestPlan = expanded.plan();
                    progressed = true;
                    lastRejection = new Rejection("frontier-exhausted", -1);
                    break;
                }
                lastRejection = new Rejection(classifyRejection(selected, frontierNodeId, request, covered), frontierNodeId);
            }
            if (budgetHit || !progressed) {
                break;
            }
        }

        return new AttemptResult(
                bestPlan,
                new AcceleratorPartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startIndex,
                        true,
                        budgetHit ? "budget-stop" : lastRejection.reason(),
                        bestPlan.nodeIds(),
                        bestCandidate == null ? bestPlan.nodeIds() : bestCandidate.orderedNodeIds(),
                        opNames(bestPlan.nodeIds(), context),
                        bestPlan.estimatedWork(),
                        Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY,
                        explored,
                        budgetHit,
                        lastRejection.nodeId()
                )
        );
    }

    private ExpansionResult tryExpand(
            LinkedHashSet<Integer> selected,
            int frontierNodeId,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        LinkedHashSet<Integer> expanded = new LinkedHashSet<>(selected);
        Rejection rejection = absorbWithProducerClosure(frontierNodeId, expanded, request, covered);
        if (rejection != null) {
            return null;
        }
        rejection = absorbSupportedConsumerClosure(expanded, request, covered);
        if (rejection != null) {
            return null;
        }
        AcceleratorStructuralCandidate candidate = request.adapter().tryCreateStructuralCandidate(expanded, request.context());
        if (candidate == null) {
            return null;
        }
        AcceleratorPartitionPlan plan = request.adapter().tryCreatePlan(candidate, request.context());
        if (plan == null) {
            return null;
        }
        return new ExpansionResult(expanded, candidate, plan);
    }

    private Rejection absorbWithProducerClosure(
            int nodeId,
            LinkedHashSet<Integer> expanded,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(nodeId);
        while (!stack.isEmpty()) {
            int currentNodeId = stack.pop();
            if (expanded.contains(currentNodeId)) {
                continue;
            }
            if (expanded.size() >= request.policy().maxSearchNodes()) {
                return new Rejection("max-search-nodes", currentNodeId);
            }
            CompiledNode current = request.context().compiledNode(currentNodeId);
            if (current == null) {
                return new Rejection("missing-node", currentNodeId);
            }
            if (covered[currentNodeId]) {
                return new Rejection("covered-by-earlier-partition", currentNodeId);
            }
            if (current.backend() != request.target().backend() || !request.adapter().isNodeSupported(current, request.context())) {
                return new Rejection("unsupported-node", currentNodeId);
            }
            expanded.add(currentNodeId);
            List<Integer> inputs = current.inputIds();
            for (int i = inputs.size() - 1; i >= 0; i--) {
                int inputId = inputs.get(i);
                if (expanded.contains(inputId)) {
                    continue;
                }
                CompiledNode producer = request.context().compiledNode(inputId);
                if (producer == null) {
                    return new Rejection("missing-input-node", inputId);
                }
                boolean sameTargetSupported = producer.backend() == request.target().backend()
                        && !covered[inputId]
                        && request.adapter().isNodeSupported(producer, request.context());
                if (sameTargetSupported) {
                    stack.push(inputId);
                    continue;
                }
                if (!request.adapter().canUseAsExternalInput(producer, current, expanded, request.context())) {
                    return new Rejection("external-input-not-allowed", inputId);
                }
            }
        }
        return null;
    }

    private Rejection absorbSupportedConsumerClosure(
            LinkedHashSet<Integer> expanded,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        boolean changed;
        do {
            changed = false;
            List<Integer> currentNodes = List.copyOf(expanded);
            for (int nodeId : currentNodes) {
                for (CompiledNode consumer : request.context().consumersFor(nodeId)) {
                    if (consumer == null || expanded.contains(consumer.id())) {
                        continue;
                    }
                    if (covered[consumer.id()]) {
                        return new Rejection("covered-by-earlier-partition", consumer.id());
                    }
                    if (consumer.backend() != request.target().backend()
                            || !request.adapter().isNodeSupported(consumer, request.context())) {
                        continue;
                    }
                    Rejection rejection = absorbWithProducerClosure(consumer.id(), expanded, request, covered);
                    if (rejection != null) {
                        return rejection;
                    }
                    changed = true;
                }
            }
        } while (changed);
        return null;
    }

    private String classifyRejection(
            Set<Integer> selected,
            int frontierNodeId,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        CompiledNode node = request.context().compiledNode(frontierNodeId);
        if (node == null) {
            return "missing-node";
        }
        if (covered[frontierNodeId]) {
            return "covered-by-earlier-partition";
        }
        if (node.backend() != request.target().backend() || !request.adapter().isNodeSupported(node, request.context())) {
            return "unsupported-node";
        }
        for (int inputId : node.inputIds()) {
            if (selected.contains(inputId)) {
                continue;
            }
            CompiledNode producer = request.context().compiledNode(inputId);
            if (producer == null) {
                return "missing-input-node";
            }
            boolean sameTargetSupported = producer.backend() == request.target().backend()
                    && !covered[inputId]
                    && request.adapter().isNodeSupported(producer, request.context());
            if (sameTargetSupported) {
                return "lowerer-rejected";
            }
            if (!request.adapter().canUseAsExternalInput(producer, node, selected, request.context())) {
                return "external-input-not-allowed";
            }
        }
        return "lowerer-rejected";
    }

    private List<Integer> expandableConsumers(
            Set<Integer> selectedNodeIds,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        LinkedHashSet<Integer> out = new LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            for (CompiledNode consumer : request.context().consumersFor(nodeId)) {
                if (consumer == null
                        || selectedNodeIds.contains(consumer.id())
                        || covered[consumer.id()]
                        || consumer.backend() != request.target().backend()) {
                    continue;
                }
                out.add(consumer.id());
            }
        }
        return out.stream()
                .sorted(Comparator
                        .comparingInt((Integer nodeId) -> isMergeCompleting(nodeId, selectedNodeIds, request.context()) ? 0 : 1)
                        .thenComparingInt(Integer::intValue))
                .toList();
    }

    private boolean isMergeCompleting(int nodeId, Set<Integer> selectedNodeIds, BackendPrepareContext context) {
        CompiledNode node = context.compiledNode(nodeId);
        if (node == null || node.inputIds().size() < 2) {
            return false;
        }
        int selectedInputs = 0;
        for (int inputId : node.inputIds()) {
            if (selectedNodeIds.contains(inputId)) {
                selectedInputs++;
            }
        }
        return selectedInputs > 1;
    }

    private List<String> opNames(List<Integer> nodeIds, BackendPrepareContext context) {
        List<String> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null && node.operation() != null) {
                out.add(node.operation().opType().name());
            }
        }
        return List.copyOf(out);
    }
}
