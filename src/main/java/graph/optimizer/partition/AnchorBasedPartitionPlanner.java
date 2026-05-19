package graph.optimizer.partition;

import graph.optimizer.GraphValueRef;

import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import operations.Operation;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Anchor-first accelerator planner.
 *
 * <p>Unlike node-order greedy planning, this planner starts from high-value compute anchors before cheap layout/view
 * nodes can become covered singleton regions. It then uses the same legality and lowering adapter as the other
 * planners, so accepted regions are executable by the backend-specific lowering pipeline.</p>
 */
public final class AnchorBasedPartitionPlanner implements PartitionPlanner {
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
        for (int nodeId : anchorOrder(request)) {
            CompiledNode current = context.compiledNode(nodeId);
            if (current == null || !request.canConsiderNode(current)) {
                continue;
            }
            if (covered[nodeId]) {
                decisions.add(new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        nodeId,
                        false,
                        "covered-by-earlier-partition",
                        List.of(nodeId),
                        List.of(nodeId),
                        opNames(List.of(nodeId), context),
                        0L,
                        Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY,
                        0,
                        false,
                        -1
                ));
                continue;
            }
            AttemptResult attempt = tryBuildPlan(nodeId, request, covered);
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
        partitions.sort(Comparator.comparingInt(Partition::anchorSeedNodeId));
        int accepted = (int) decisions.stream().filter(PartitionDecisionTrace::accepted).count();
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

    private List<Integer> anchorOrder(PartitionPlanningRequest request) {
        return request.context().compiledNodes().stream()
                .filter(node -> request.canConsiderNode(node))
                .filter(node -> request.adapter().canSeed(node, request.context()))
                .sorted(anchorComparator(request))
                .map(CompiledNode::id)
                .toList();
    }

    private Comparator<CompiledNode> anchorComparator(PartitionPlanningRequest request) {
        Comparator<CompiledNode> comparator = Comparator
                .comparingInt((CompiledNode node) -> anchorPriority(node, request))
                .reversed();
        if (allowsMixedTrainingPhases(request)) {
            return comparator.thenComparing(CompiledNode::id, Comparator.reverseOrder());
        }
        return comparator.thenComparingInt(CompiledNode::id);
    }

    private int anchorPriority(CompiledNode node) {
        return anchorPriority(node, null);
    }

    private int anchorPriority(CompiledNode node, PartitionPlanningRequest request) {
        if (node == null || node.operation() == null) {
            return 0;
        }
        Operation.OpType opType = node.operation().opType();
        int priority = switch (opType) {
            case SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> 10_000;
            case SCALED_DOT_PRODUCT_ATTENTION -> 9_500;
            case CONV2D, CONV2D_GEMM, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_INPUT_GEMM,
                    CONV2D_BACKWARD_WEIGHT, CONV2D_BACKWARD_WEIGHT_GEMM -> 9_000;
            case MATMUL, LINEAR -> 8_500;
            case CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD,
                    NLL_LOSS -> 8_000;
            case LAYER_NORM, RMS_NORM -> 7_500;
            case SOFTMAX, LOG_SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX_GRAD -> 7_000;
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD -> 6_500;
            case MAX_POOL2D, AVG_POOL2D, MAX_POOL2D_BACKWARD_INPUT, AVG_POOL2D_BACKWARD_INPUT -> 6_000;
            case ADD, SUB, MUL, DIV, MIN, MAX, RELU, TANH, FAST_TANH, SIGMOID, EXP, FAST_EXP,
                    ERF, LOG, SQRT, NEG, ABS, FLOOR, CEIL, SIGN, INV, POW, MUL_SCALAR -> 4_000;
            case RESHAPE, PERMUTE, CONTIGUOUS, EXPAND, EXPAND_DIMS, SQUEEZE, SELECT, SLICE, CONCAT, NOOP -> 1_000;
            case SLICE_GRAD, SLICE_SCATTER_ADD, GATHER_AXIS, GATHER_AXIS_GRAD, GATHER_ND, GATHER_ND_GRAD,
                 SCATTER_AXIS_ADD -> 2_000;
            default -> 2_000;
        };
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
        if (!request.adapter().canSeed(start, context)) {
            return rejected(request, startNodeId, "unsupported-start-node", 0, false, -1);
        }

        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        Rejection seedRejection = absorbWithProducerClosure(start.id(), selected, request, covered);
        if (seedRejection != null) {
            return rejected(request, startNodeId, seedRejection.reason(), 0, false, seedRejection.nodeId());
        }
        seedRejection = absorbSupportedConsumerClosure(selected, request, covered);
        if (seedRejection != null) {
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
        PartitionCandidate bestCandidate = request.adapter().tryCreateStructuralCandidate(
                selected,
                context,
                request.requiredMaterializedValueRefs()
        );
        PartitionPlan bestPlan = bestCandidate == null ? null : request.adapter().tryCreatePlan(bestCandidate, context);
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
        boolean budgetHit = false;
        Rejection lastRejection = new Rejection("frontier-exhausted", -1);
        while (true) {
            List<Integer> frontier = expandableNeighbors(selected, request, covered);
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
                buildPartition(
                        request,
                        bestCandidate,
                        bestPlan,
                        budgetHit ? "anchor-budget-stop" : "anchor-" + lastRejection.reason(),
                        lastRejection.nodeId(),
                        explored
                ),
                bestPlan,
                new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startNodeId,
                        true,
                        budgetHit ? "anchor-budget-stop" : "anchor-" + lastRejection.reason(),
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
                        opNames(structuralNodeIds, request.context()),
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
                boolean autoDiscoveryCpuProducer = request.sourcePolicy() == PartitionSourcePolicy.CPU_OR_TARGET_BACKEND
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
                        .comparingInt((Integer nodeId) -> neighborPriority(nodeId, selectedNodeIds, request.context())).reversed()
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
                || !request.adapter().isNodeSupported(node, request.context())) {
            return;
        }
        out.add(nodeId);
    }

    private int neighborPriority(int nodeId, Set<Integer> selectedNodeIds, PartitionPlanningContext context) {
        CompiledNode node = context.compiledNode(nodeId);
        int priority = anchorPriority(node);
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
            return phaseBoundaryReason(selected, frontierNodeId, request.context());
        }
        if (!request.canConsiderNode(node) || !request.adapter().isNodeSupported(node, request.context())) {
            return "unsupported-node";
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

    private boolean crossesSelectedPhase(Set<Integer> selectedNodeIds, int candidateNodeId, PartitionPlanningRequest request) {
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
        return request != null
                && request.context().supportsBackward()
                && (request.target() == PartitionTarget.GPU_METAL || request.target() == PartitionTarget.GPU_CUDA);
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

    private Partition buildPartition(
            PartitionPlanningRequest request,
            PartitionCandidate candidate,
            PartitionPlan attachedPlan,
            String reason,
            int rejectedNodeId,
            int explored
    ) {
        PartitionPlanningContext context = request.context();
        var metrics = metricsFor(candidate, context);
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
        PartitionDecisionTrace trace = new PartitionDecisionTrace(
                request.strategy(),
                request.target(),
                candidate.anchorNodeId(),
                true,
                reason,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds(),
                opNames(candidate.orderedNodeIds(), context),
                attachedPlan == null ? 0L : attachedPlan.estimatedWork(),
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                explored,
                reason.contains("budget-stop"),
                rejectedNodeId
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
                attachedPlan == null ? 0L : attachedPlan.estimatedWork(),
                metrics,
                request.strategy(),
                trace
        );
    }

    private AcceleratorPartitionScoreModel.CandidateMetrics metricsFor(PartitionCandidate candidate, PartitionPlanningContext context) {
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

    private String partitionId(PartitionTarget target, int anchorNodeId) {
        String prefix = target == null ? "partition" : target.name().toLowerCase(java.util.Locale.ROOT);
        return prefix + "-anchor-" + anchorNodeId;
    }
}
