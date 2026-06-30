package graph.compile.planning.partition;

import graph.model.CompiledNode;
import graph.compile.planning.value.GraphValueRef;
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
 * Planner that expands each seed into the largest legal backend region it can find.
 *
 * <p>The algorithm is shared by node-order greedy planning and anchor-first planning. The only intentional difference
 * between those modes is seed/frontier ordering and the anchor-first preference for pulling supported producer nodes
 * into accelerator regions during automatic discovery.</p>
 */
public final class MaxRegionPartitionPlanner implements PartitionPlanner {
    public enum SeedOrdering {
        NODE_ORDER,
        ANCHOR_FIRST
    }

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

    private record Rejection(String reason, int nodeId) {
    }

    private final SeedOrdering seedOrdering;

    public MaxRegionPartitionPlanner(SeedOrdering seedOrdering) {
        this.seedOrdering = seedOrdering == null ? SeedOrdering.NODE_ORDER : seedOrdering;
    }

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
        for (int seedNodeId : seedOrder(request)) {
            CompiledNode current = context.compiledNode(seedNodeId);
            if (current == null || !request.canConsiderNode(current)) {
                continue;
            }
            if (covered[seedNodeId]) {
                decisions.add(PartitionDecisionTrace.coveredByEarlierPartition(
                        request.strategy(),
                        request.target(),
                        seedNodeId,
                        PartitionAssembly.opNames(List.of(seedNodeId), context)
                ));
                continue;
            }
            AttemptResult attempt = tryBuildPlan(seedNodeId, request, covered);
            decisions.add(attempt.decision());
            if (attempt.partition() == null) {
                continue;
            }
            if (attempt.attachedPlan() != null) {
                plansByPartitionId.put(attempt.partition().partitionId(), attempt.attachedPlan());
            }
            for (int acceptedNodeId : attempt.partition().orderedNodeIds()) {
                covered[acceptedNodeId] = true;
            }
            partitions.add(attempt.partition());
        }
        if (seedOrdering == SeedOrdering.ANCHOR_FIRST) {
            partitions.sort(Comparator.comparingInt(Partition::anchorSeedNodeId));
        }
        return new PartitionPlanningResult(
                partitions,
                plansByPartitionId,
                PartitionCompileTrace.forJob(request.strategy(), request.target(), decisions)
        );
    }

    private List<Integer> seedOrder(PartitionPlanningRequest request) {
        if (seedOrdering == SeedOrdering.NODE_ORDER) {
            return request.context().compiledNodes().stream()
                    .map(CompiledNode::id)
                    .toList();
        }
        return request.context().compiledNodes().stream()
                .filter(node -> request.canConsiderNode(node))
                .filter(node -> request.capability().canSeed(node, request.context()))
                .sorted(anchorComparator(request))
                .map(CompiledNode::id)
                .toList();
    }

    private Comparator<CompiledNode> anchorComparator(PartitionPlanningRequest request) {
        Comparator<CompiledNode> comparator = Comparator
                .comparingInt((CompiledNode node) -> seedPriority(node, request))
                .reversed();
        if (allowsMixedTrainingPhases(request)) {
            return comparator.thenComparing(CompiledNode::id, Comparator.reverseOrder());
        }
        return comparator.thenComparingInt(CompiledNode::id);
    }

    private int seedPriority(CompiledNode node, PartitionPlanningRequest request) {
        if (node == null || node.operation() == null) {
            return 0;
        }
        int priority = request.capability().partitionPriority(node, request.context());
        if (allowsMixedTrainingPhases(request)) {
            if (request.requiredMaterializedValueRefs().contains(GraphValueRef.node(node.id()))) {
                priority += 30_000;
            }
            if (request.context().consumersFor(node.id()).isEmpty()) {
                priority += 20_000;
            }
            if (node.backwardNode()) {
                priority += 5_000;
            }
        }
        return priority;
    }

    private AttemptResult tryBuildPlan(int startNodeId, PartitionPlanningRequest request, boolean[] covered) {
        PartitionPlanningContext context = request.context();
        CompiledNode start = context.compiledNode(startNodeId);
        if (!request.capability().canSeed(start, context)) {
            return rejected(request, startNodeId, "unsupported-start-node", 0, false, -1);
        }

        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        Rejection seedRejection = absorbWithProducerClosure(start.id(), selected, request, covered);
        if (seedRejection != null) {
            return rejected(request, startNodeId, seedRejection.reason(), 0, false, seedRejection.nodeId());
        }
        seedRejection = absorbSupportedConsumerClosure(selected, request, covered);
        boolean initialBudgetStop = seedOrdering == SeedOrdering.NODE_ORDER
                && seedRejection != null
                && "max-search-nodes".equals(seedRejection.reason());
        if (seedRejection != null && !initialBudgetStop) {
            return rejected(
                    request,
                    startNodeId,
                    seedRejection.reason(),
                    0,
                    false,
                    seedRejection.nodeId(),
                    List.copyOf(selected)
            );
        }

        PartitionCandidate bestCandidate = request.capability().createCandidate(
                selected,
                context,
                request.requiredMaterializedValueRefs()
        );
        PartitionPlan bestPlan = bestCandidate == null ? null : request.capability().createPlan(bestCandidate, context);
        if (bestPlan == null) {
            String reason = bestCandidate == null ? "missing-structural-candidate" : "lowerer-rejected";
            return rejected(
                    request,
                    startNodeId,
                    reason,
                    0,
                    false,
                    -1,
                    bestCandidate == null ? List.copyOf(selected) : bestCandidate.orderedNodeIds()
            );
        }

        int explored = 0;
        boolean budgetHit = initialBudgetStop;
        Rejection lastRejection = initialBudgetStop ? seedRejection : new Rejection("frontier-exhausted", -1);
        while (!budgetHit) {
            List<Integer> frontier = expandableFrontier(selected, request, covered);
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

        String reason = acceptedReason(budgetHit, lastRejection.reason());
        return new AttemptResult(
                PartitionAssembly.acceptedPartition(
                        request,
                        bestCandidate,
                        bestPlan,
                        reason,
                        lastRejection.nodeId(),
                        explored,
                        budgetHit,
                        null,
                        List.of()
                ),
                bestPlan,
                new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startNodeId,
                        true,
                        reason,
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

    private String acceptedReason(boolean budgetHit, String rejectionReason) {
        if (seedOrdering == SeedOrdering.ANCHOR_FIRST) {
            return budgetHit ? "anchor-budget-stop" : "anchor-" + rejectionReason;
        }
        return budgetHit ? "budget-stop" : rejectionReason;
    }

    private AttemptResult rejected(
            PartitionPlanningRequest request,
            int startNodeId,
            String reason,
            int explored,
            boolean budgetHit,
            int rejectedNodeId
    ) {
        return rejected(request, startNodeId, reason, explored, budgetHit, rejectedNodeId, List.of(startNodeId));
    }

    private AttemptResult rejected(
            PartitionPlanningRequest request,
            int startNodeId,
            String reason,
            int explored,
            boolean budgetHit,
            int rejectedNodeId,
            List<Integer> structuralNodeIds
    ) {
        return new AttemptResult(
                null,
                null,
                new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startNodeId,
                        false,
                        reason,
                        List.of(startNodeId),
                        structuralNodeIds,
                        PartitionAssembly.opNames(structuralNodeIds, request.context()),
                        0L,
                        Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY,
                        explored,
                        budgetHit,
                        rejectedNodeId
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
        PartitionCandidate candidate = request.capability().createCandidate(
                expanded,
                request.context(),
                request.requiredMaterializedValueRefs()
        );
        if (candidate == null) {
            return null;
        }
        PartitionPlan plan = request.capability().createPlan(candidate, request.context());
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
            if (!request.canConsiderNode(current) || !request.capability().canExecute(current, request.context())) {
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
                boolean externalInputAllowed = request.capability().canUseExternalInput(
                        producer,
                        current,
                        expanded,
                        request.context()
                );
                boolean sameTargetSupported = request.canConsiderNode(producer)
                        && !covered[inputId]
                        && request.capability().canExecute(producer, request.context());
                boolean autoDiscoveryCpuProducer = seedOrdering == SeedOrdering.ANCHOR_FIRST
                        && request.sourcePolicy() == PartitionSourcePolicy.CPU_OR_TARGET_BACKEND
                        && producer.backend() != request.target().backend();
                if (sameTargetSupported && (!externalInputAllowed || autoDiscoveryCpuProducer)) {
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
                            || !request.capability().canExecute(consumer, request.context())) {
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

    private List<Integer> expandableFrontier(
            Set<Integer> selectedNodeIds,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        return seedOrdering == SeedOrdering.ANCHOR_FIRST
                ? expandableNeighbors(selectedNodeIds, request, covered)
                : expandableConsumers(selectedNodeIds, request, covered);
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

    private List<Integer> expandableNeighbors(
            Set<Integer> selectedNodeIds,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        LinkedHashSet<Integer> out = new LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode selected = request.context().compiledNode(nodeId);
            if (selected != null) {
                for (int inputId : selected.inputIds()) {
                    addExpandable(inputId, selectedNodeIds, request, covered, out);
                }
            }
            for (CompiledNode consumer : request.context().consumersFor(nodeId)) {
                if (consumer != null) {
                    addExpandable(consumer.id(), selectedNodeIds, request, covered, out);
                }
            }
        }
        return out.stream()
                .sorted(Comparator
                        .comparingInt((Integer nodeId) -> neighborPriority(nodeId, selectedNodeIds, request)).reversed()
                        .thenComparingInt(Integer::intValue))
                .toList();
    }

    private void addExpandable(
            int nodeId,
            Set<Integer> selectedNodeIds,
            PartitionPlanningRequest request,
            boolean[] covered,
            LinkedHashSet<Integer> out
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        if (node == null
                || selectedNodeIds.contains(nodeId)
                || covered[nodeId]
                || !request.canConsiderNode(node)
                || !request.capability().canExecute(node, request.context())) {
            return;
        }
        out.add(nodeId);
    }

    private int neighborPriority(int nodeId, Set<Integer> selectedNodeIds, PartitionPlanningRequest request) {
        PartitionPlanningContext context = request.context();
        CompiledNode node = context.compiledNode(nodeId);
        int priority = request.capability().partitionPriority(node, context);
        if (isMergeCompleting(nodeId, selectedNodeIds, context)) {
            priority += 1_000;
        }
        if (hasSelectedConsumer(nodeId, selectedNodeIds, context)) {
            priority += 500;
        }
        return priority;
    }

    private boolean hasSelectedConsumer(int nodeId, Set<Integer> selectedNodeIds, PartitionPlanningContext context) {
        for (CompiledNode consumer : context.consumersFor(nodeId)) {
            if (consumer != null && selectedNodeIds.contains(consumer.id())) {
                return true;
            }
        }
        return false;
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
            return seedOrdering == SeedOrdering.ANCHOR_FIRST
                    ? phaseBoundaryReason(selected, frontierNodeId, request.context())
                    : "consumer-closure-phase-boundary";
        }
        if (!request.canConsiderNode(node) || !request.capability().canExecute(node, request.context())) {
            return "unsupported-node";
        }
        if (seedOrdering == SeedOrdering.NODE_ORDER) {
            String inputRejection = classifyInputRejection(selected, node, request, covered);
            if (inputRejection != null) {
                return inputRejection;
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
        PartitionCandidate candidate = request.capability().createCandidate(
                expanded,
                request.context(),
                request.requiredMaterializedValueRefs()
        );
        if (candidate == null) {
            return "missing-structural-candidate";
        }
        PartitionPlan plan = request.capability().createPlan(candidate, request.context());
        if (plan == null) {
            return "lowerer-rejected";
        }
        return "unknown-expansion-rejected";
    }

    private String classifyInputRejection(
            Set<Integer> selected,
            CompiledNode node,
            PartitionPlanningRequest request,
            boolean[] covered
    ) {
        for (int inputId : node.inputIds()) {
            if (selected.contains(inputId)) {
                continue;
            }
            CompiledNode producer = request.context().compiledNode(inputId);
            if (producer == null) {
                return "missing-input-node";
            }
            boolean externalInputAllowed = request.capability().canUseExternalInput(
                    producer,
                    node,
                    selected,
                    request.context()
            );
            boolean sameTargetSupported = request.canConsiderNode(producer)
                    && !covered[inputId]
                    && request.capability().canExecute(producer, request.context());
            if (sameTargetSupported && !externalInputAllowed && crossesSelectedPhase(selected, inputId, request)) {
                return "producer-closure-phase-boundary";
            }
            if (!externalInputAllowed) {
                return "external-input-not-allowed";
            }
        }
        return null;
    }

    private String phaseBoundaryReason(Set<Integer> selectedNodeIds, int candidateNodeId, PartitionPlanningContext context) {
        CompiledNode candidate = context.compiledNode(candidateNodeId);
        if (candidate == null) {
            return "phase-boundary";
        }
        for (int selectedNodeId : selectedNodeIds) {
            CompiledNode selected = context.compiledNode(selectedNodeId);
            if (selected == null) {
                continue;
            }
            if (candidate.inputIds().contains(selected.id())) {
                return "consumer-closure-phase-boundary";
            }
            if (selected.inputIds().contains(candidate.id())) {
                return "producer-closure-phase-boundary";
            }
        }
        return "phase-boundary";
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
