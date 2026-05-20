package graph.compile.planning.partition;

import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Greedy planner that expands each seed into the largest legal contiguous-ish backend region it can find.
 *
 * <p>The planner iterates graph nodes in order, skips nodes already covered by accepted partitions, grows through
 * supported consumers while respecting backend legality, and records a {@link PartitionDecisionTrace} for every seed it
 * considers. It favors fast deterministic planning over exhaustive candidate search.
 */
public final class GreedyMaxRegionPartitionPlanner implements PartitionPlanner {
    private record AttemptResult(
            Partition partition,
            PartitionPlan attachedPlan,
            PartitionDecisionTrace decision
    ) {
    }

    private record ExpansionResult(
            LinkedHashSet<Integer> selectedNodeIds,
            PartitionCandidate candidate,
            PartitionPlan attachedPlan
    ) {
    }

    private record Rejection(
            String reason,
            int nodeId
    ) {
    }

    /**
     * Plans greedy maximum regions for the request target.
     *
     * @param request planning request
     * @return accepted partitions, attached backend plans, and trace decisions
     */
    @Override
    public PartitionPlanningResult plan(PartitionPlanningRequest request) {
        if (request == null || request.target().isNone()) {
            return PartitionPlanningResult.empty();
        }
        PartitionPlanningContext context = request.context();
        List<CompiledNode> nodes = context.compiledNodes();
        if (nodes.isEmpty()) {
            return PartitionPlanningResult.empty();
        }
        boolean[] covered = new boolean[nodes.size()];
        List<Partition> partitions = new ArrayList<>();
        java.util.LinkedHashMap<String, PartitionPlan> plansByPartitionId = new java.util.LinkedHashMap<>();
        List<PartitionDecisionTrace> decisions = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            CompiledNode current = nodes.get(i);
            if (!request.canConsiderNode(current)) {
                continue;
            }
            if (covered[i]) {
                decisions.add(PartitionDecisionTrace.coveredByEarlierPartition(
                        request.strategy(),
                        request.target(),
                        i,
                        PartitionAssembly.opNames(List.of(i), context)
                ));
                continue;
            }
            AttemptResult attempt = tryBuildPlan(i, request, covered);
            decisions.add(attempt.decision());
            if (attempt.partition() == null) {
                continue;
            }
            if (attempt.attachedPlan() != null) {
                plansByPartitionId.put(attempt.partition().partitionId(), attempt.attachedPlan());
            }
            for (int nodeId : attempt.partition().orderedNodeIds()) {
                covered[nodeId] = true;
            }
            partitions.add(attempt.partition());
        }
        return new PartitionPlanningResult(
                partitions,
                plansByPartitionId,
                PartitionCompileTrace.forJob(request.strategy(), request.target(), decisions)
        );
    }

    private AttemptResult tryBuildPlan(int startIndex, PartitionPlanningRequest request, boolean[] covered) {
        PartitionPlanningContext context = request.context();
        CompiledNode start = context.compiledNodes().get(startIndex);
        if (!request.adapter().canSeed(start, context)) {
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "unsupported-start-node",
                            List.of(start.id()),
                            List.of(start.id()),
                            PartitionAssembly.opNames(List.of(start.id()), context),
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
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            seedRejection.reason(),
                            List.of(start.id()),
                            List.of(start.id()),
                            PartitionAssembly.opNames(List.of(start.id()), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            0,
                            false,
                            seedRejection.nodeId()
                    )
            );
        }
        seedRejection = absorbSupportedConsumerClosure(selected, request, covered);
        boolean initialBudgetStop = seedRejection != null && "max-search-nodes".equals(seedRejection.reason());
        if (seedRejection != null && !initialBudgetStop) {
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            seedRejection.reason(),
                            List.of(start.id()),
                            List.copyOf(selected),
                            PartitionAssembly.opNames(List.copyOf(selected), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            0,
                            false,
                            seedRejection.nodeId()
                    )
            );
        }
        PartitionCandidate bestCandidate = request.adapter().tryCreateStructuralCandidate(
                selected,
                context,
                request.requiredMaterializedValueRefs()
        );
        PartitionPlan bestPlan = bestCandidate == null ? null : request.adapter().tryCreatePlan(bestCandidate, context);
        if (bestPlan == null) {
            String reason = bestCandidate == null ? "missing-structural-candidate" : "lowerer-rejected";
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            reason,
                            List.of(start.id()),
                            bestCandidate == null ? List.copyOf(selected) : bestCandidate.orderedNodeIds(),
                            PartitionAssembly.opNames(bestCandidate == null ? List.copyOf(selected) : bestCandidate.orderedNodeIds(), context),
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
        Rejection lastRejection = initialBudgetStop ? seedRejection : new Rejection("frontier-exhausted", -1);
        boolean budgetHit = initialBudgetStop;
        while (!budgetHit) {
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
                    bestPlan = expanded.attachedPlan();
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
                PartitionAssembly.acceptedPartition(
                        request,
                        bestCandidate,
                        bestPlan,
                        budgetHit ? "budget-stop" : lastRejection.reason(),
                        lastRejection.nodeId(),
                        explored
                ),
                bestPlan,
                new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startIndex,
                        true,
                        budgetHit ? "budget-stop" : lastRejection.reason(),
                        bestPlan.nodeIds(),
                        bestCandidate == null ? bestPlan.nodeIds() : bestCandidate.orderedNodeIds(),
                        PartitionAssembly.opNames(bestPlan.nodeIds(), context),
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
        PartitionCandidate candidate = request.adapter().tryCreateStructuralCandidate(
                expanded,
                request.context(),
                request.requiredMaterializedValueRefs()
        );
        if (candidate == null) {
            return null;
        }
        PartitionPlan plan = request.adapter().tryCreatePlan(candidate, request.context());
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
            if (crossesSelectedPhase(expanded, currentNodeId, request)) {
                return new Rejection("producer-closure-phase-boundary", currentNodeId);
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
            if (!request.canConsiderNode(current) || !request.adapter().isNodeSupported(current, request.context())) {
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
                boolean externalInputAllowed = request.adapter().canUseAsExternalInput(
                        producer,
                        current,
                        expanded,
                        request.context()
                );
                boolean sameTargetSupported = request.canConsiderNode(producer)
                        && !covered[inputId]
                        && request.adapter().isNodeSupported(producer, request.context());
                if (sameTargetSupported && !externalInputAllowed) {
                    if (crossesSelectedPhase(expanded, inputId, request)) {
                        return new Rejection("producer-closure-phase-boundary", inputId);
                    }
                    stack.push(inputId);
                    continue;
                }
                if (!externalInputAllowed) {
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
                    if (crossesSelectedPhase(expanded, consumer.id(), request)) {
                        continue;
                    }
                    if (covered[consumer.id()]) {
                        return new Rejection("covered-by-earlier-partition", consumer.id());
                    }
                    if (!request.canConsiderNode(consumer)
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
        if (crossesSelectedPhase(selected, frontierNodeId, request)) {
            return "consumer-closure-phase-boundary";
        }
        if (!request.canConsiderNode(node) || !request.adapter().isNodeSupported(node, request.context())) {
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
            boolean externalInputAllowed = request.adapter().canUseAsExternalInput(
                    producer,
                    node,
                    selected,
                    request.context()
            );
            boolean sameTargetSupported = request.canConsiderNode(producer)
                    && !covered[inputId]
                    && request.adapter().isNodeSupported(producer, request.context());
            if (sameTargetSupported && !externalInputAllowed) {
                if (crossesSelectedPhase(selected, inputId, request)) {
                    return "producer-closure-phase-boundary";
                }
            }
            if (!externalInputAllowed) {
                return "external-input-not-allowed";
            }
        }
        LinkedHashSet<Integer> expanded = new LinkedHashSet<>(selected);
        Rejection rejection = absorbWithProducerClosure(frontierNodeId, expanded, request, covered);
        if (rejection != null) {
            return rejection.reason();
        }
        rejection = absorbSupportedConsumerClosure(expanded, request, covered);
        if (rejection != null) {
            return rejection.reason();
        }
        PartitionCandidate candidate = request.adapter().tryCreateStructuralCandidate(
                expanded,
                request.context(),
                request.requiredMaterializedValueRefs()
        );
        if (candidate == null) {
            return "missing-structural-candidate";
        }
        PartitionPlan plan = request.adapter().tryCreatePlan(candidate, request.context());
        if (plan == null) {
            return "lowerer-rejected";
        }
        return "unknown-expansion-rejected";
    }

    private boolean crossesSelectedPhase(
            Set<Integer> selectedNodeIds,
            int candidateNodeId,
            PartitionPlanningRequest request
    ) {
        PartitionPlanningContext context = request == null ? null : request.context();
        if (selectedNodeIds == null || selectedNodeIds.isEmpty() || context == null || candidateNodeId < 0) {
            return false;
        }
        if (allowsMixedTrainingPhases(request)) {
            return false;
        }
        CompiledNode candidate = context.compiledNode(candidateNodeId);
        if (candidate == null) {
            return false;
        }
        Boolean selectedBackward = null;
        for (int selectedNodeId : selectedNodeIds) {
            CompiledNode selected = context.compiledNode(selectedNodeId);
            if (selected == null) {
                continue;
            }
            if (selectedBackward == null) {
                selectedBackward = selected.backwardNode();
            } else if (selectedBackward.booleanValue() != selected.backwardNode()) {
                return false;
            }
        }
        return selectedBackward != null && selectedBackward.booleanValue() != candidate.backwardNode();
    }

    private boolean allowsMixedTrainingPhases(PartitionPlanningRequest request) {
        return RegionExpansionPolicy.allowsMixedTrainingPhases(request);
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
                        || !request.canConsiderNode(consumer)) {
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

    private boolean isMergeCompleting(int nodeId, Set<Integer> selectedNodeIds, PartitionPlanningContext context) {
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

}
