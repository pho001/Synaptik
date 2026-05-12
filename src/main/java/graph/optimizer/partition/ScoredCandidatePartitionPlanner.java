package graph.optimizer.partition;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.LayoutClass;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Candidate-search planner that scores structural regions before accepting a backend plan.
 *
 * <p>Compared with {@link GreedyMaxRegionPartitionPlanner}, this planner explores more candidate shapes and uses
 * {@link AcceleratorPartitionScoreModel} to balance node count, internal edges, external inputs, tail depth, and backend
 * work estimates. Search limits are supplied by the request policy and reported in partition trace metadata.
 */
public final class ScoredCandidatePartitionPlanner implements PartitionPlanner {
    private record AttemptResult(
            Partition partition,
            PartitionPlan attachedPlan,
            PartitionDecisionTrace decision
    ) {
    }

    private record SearchOutcome(
            PartitionCandidate bestAccepted,
            double bestAcceptedScore,
            long bestAcceptedWork,
            AcceleratorPartitionScoreModel.MaterializationCostSummary bestAcceptedCostSummary,
            PartitionCandidate bestStructural,
            double bestStructuralScore,
            int exploredCandidates,
            boolean searchBudgetHit,
            List<PartitionDecisionTrace.CandidateCostTrace> topRejectedFinalists
    ) {
    }

    /**
     * Plans scored candidate regions for the request target.
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
                decisions.add(new PartitionDecisionTrace(
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
        SearchOutcome search = searchBestCandidate(start.id(), request, covered);
        PartitionCandidate accepted = search.bestAccepted();
        PartitionCandidate structural = search.bestStructural();
        if (structural == null) {
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "missing-structural-candidate",
                            List.of(start.id()),
                            List.of(start.id()),
                            opNames(List.of(start.id()), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            Double.NEGATIVE_INFINITY,
                            search.exploredCandidates(),
                            search.searchBudgetHit(),
                            -1,
                            null,
                            search.topRejectedFinalists()
                    )
            );
        }
        if (accepted == null) {
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "lowerer-rejected",
                            structural.orderedNodeIds(),
                            structural.orderedNodeIds(),
                            opNames(structural.orderedNodeIds(), context),
                            0L,
                            Double.NEGATIVE_INFINITY,
                            search.bestStructuralScore(),
                            search.exploredCandidates(),
                            search.searchBudgetHit(),
                            -1,
                            null,
                            search.topRejectedFinalists()
                    )
            );
        }
        PartitionPlan plan = request.adapter().tryCreatePlan(accepted, context);
        if (plan == null) {
            return new AttemptResult(
                    null,
                    null,
                    new PartitionDecisionTrace(
                            request.strategy(),
                            request.target(),
                            startIndex,
                            false,
                            "lowerer-rejected",
                            accepted.orderedNodeIds(),
                            structural.orderedNodeIds(),
                            opNames(structural.orderedNodeIds(), context),
                            0L,
                            search.bestAcceptedScore(),
                            search.bestStructuralScore(),
                            search.exploredCandidates(),
                            search.searchBudgetHit(),
                            -1
                    )
            );
        }
        return new AttemptResult(
                buildPartition(
                        request,
                        accepted,
                        plan,
                        "lowered",
                        -1,
                        search.exploredCandidates(),
                        search.searchBudgetHit(),
                        search.bestAcceptedCostSummary(),
                        search.topRejectedFinalists()
                ),
                plan,
                new PartitionDecisionTrace(
                        request.strategy(),
                        request.target(),
                        startIndex,
                        true,
                        "lowered",
                        accepted.orderedNodeIds(),
                        structural.orderedNodeIds(),
                        opNames(structural.orderedNodeIds(), context),
                        plan.estimatedWork(),
                        search.bestAcceptedScore(),
                        search.bestStructuralScore(),
                        search.exploredCandidates(),
                        search.searchBudgetHit(),
                        -1,
                        search.bestAcceptedCostSummary(),
                        search.topRejectedFinalists()
                )
        );
    }

    private SearchOutcome searchBestCandidate(int startNodeId, PartitionPlanningRequest request, boolean[] covered) {
        SearchAccumulator accumulator = new SearchAccumulator();
        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        selected.add(startNodeId);
        exploreCandidateSearch(selected, request, covered, new HashSet<>(), accumulator);
        return new SearchOutcome(
                accumulator.bestAccepted,
                accumulator.bestAcceptedScore,
                accumulator.bestAcceptedWork,
                accumulator.bestAcceptedCostSummary,
                accumulator.bestStructural,
                accumulator.bestStructuralScore,
                accumulator.exploredCandidates,
                accumulator.searchBudgetHit,
                accumulator.topRejectedFinalists()
        );
    }

    private void exploreCandidateSearch(
            LinkedHashSet<Integer> selected,
            PartitionPlanningRequest request,
            boolean[] covered,
            Set<String> visited,
            SearchAccumulator accumulator
    ) {
        String key = selected.stream().sorted().map(String::valueOf).reduce((a, b) -> a + "," + b).orElse("");
        if (!visited.add(key)) {
            return;
        }
        accumulator.exploredCandidates++;
        if (accumulator.exploredCandidates > request.policy().maxVisitedCandidates()) {
            accumulator.searchBudgetHit = true;
            return;
        }
        PartitionCandidate structural = request.adapter().tryCreateStructuralCandidate(
                selected,
                request.context(),
                request.requiredMaterializedValueRefs()
        );
        if (structural != null) {
            AcceleratorPartitionScoreModel.CandidateMetrics metrics = metricsFor(structural, request.context());
            double structuralScore = AcceleratorPartitionScoreModel.structuralScore(metrics, request.policy());
            if (isBetterStructural(structural, structuralScore, accumulator.bestStructural, accumulator.bestStructuralScore)) {
                accumulator.bestStructural = structural;
                accumulator.bestStructuralScore = structuralScore;
            }
            PartitionPlan acceptedPlan = request.adapter().tryCreatePlan(structural, request.context());
            if (acceptedPlan != null) {
                AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary =
                        costSummaryFor(structural, acceptedPlan, metrics, request);
                double acceptedScore = costSummary.finalScore();
                if ("rejected-materialization-cost".equals(costSummary.reasonCode())) {
                    accumulator.addFinalist(structural, "rejected-materialization-cost", costSummary);
                } else if (!"accepted-static-profitable".equals(costSummary.reasonCode())) {
                    accumulator.addFinalist(structural, costSummary.reasonCode(), costSummary);
                } else if (isBetterAccepted(
                        structural,
                        acceptedScore,
                        acceptedPlan.estimatedWork(),
                        accumulator.bestAccepted,
                        accumulator.bestAcceptedScore,
                        accumulator.bestAcceptedWork
                )) {
                    if (accumulator.bestAccepted != null) {
                        accumulator.addFinalist(
                                accumulator.bestAccepted,
                                "not-selected-lower-score",
                                accumulator.bestAcceptedCostSummary
                        );
                    }
                    accumulator.bestAccepted = structural;
                    accumulator.bestAcceptedScore = acceptedScore;
                    accumulator.bestAcceptedWork = acceptedPlan.estimatedWork();
                    accumulator.bestAcceptedCostSummary = costSummary;
                } else {
                    accumulator.addFinalist(structural, "not-selected-lower-score", costSummary);
                }
            }
        }
        if (selected.size() >= request.policy().maxSearchNodes()) {
            accumulator.searchBudgetHit = true;
            return;
        }
        List<Integer> frontier = expandableConsumers(selected, request, covered);
        for (int nodeId : frontier) {
            LinkedHashSet<Integer> expanded = new LinkedHashSet<>(selected);
            expanded.add(nodeId);
            exploreCandidateSearch(expanded, request, covered, visited, accumulator);
        }
    }

    private AcceleratorPartitionScoreModel.MaterializationCostSummary costSummaryFor(
            PartitionCandidate candidate,
            PartitionPlan acceptedPlan,
            AcceleratorPartitionScoreModel.CandidateMetrics metrics,
            PartitionPlanningRequest request
    ) {
        AcceleratorPartitionScoreModel.StaticCostPreset preset = request.target() == PartitionTarget.GPU_METAL
                ? AcceleratorPartitionScoreModel.StaticCostPreset.fromMetalTransferModel(request.metalTransferModel())
                : AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
        return AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                acceptedPlan.estimatedWork(),
                materializationSignalsFor(candidate, request.context()),
                request.policy(),
                preset
        );
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
                        || !request.canConsiderNode(consumer)
                        || !request.adapter().isNodeSupported(consumer, request.context())) {
                    continue;
                }
                if (inputsResolvable(consumer, selectedNodeIds)) {
                    out.add(consumer.id());
                }
            }
        }
        return out.stream().sorted(Comparator.naturalOrder()).toList();
    }

    private boolean inputsResolvable(CompiledNode node, Set<Integer> candidateNodeIds) {
        for (int inputId : node.inputIds()) {
            if (!candidateNodeIds.contains(inputId) && inputId > node.id()) {
                return false;
            }
        }
        return true;
    }

    private AcceleratorPartitionScoreModel.CandidateMetrics metricsFor(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        int internalEdgeCount = 0;
        int mergeNodeCount = 0;
        int tailDepth = Math.max(0, candidate.orderedNodeIds().size() - 1);
        Set<Integer> selected = new HashSet<>(candidate.orderedNodeIds());
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

    private AcceleratorPartitionScoreModel.TransferMetrics transferMetricsFor(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        Set<Integer> outputs = Set.copyOf(candidate.outputNodeIds());
        long inputBytes = 0L;
        long outputBytes = 0L;
        long avoidedIntermediateBytes = 0L;
        for (int nodeId : candidate.externalInputIds()) {
            inputBytes += nodeBytes(context.descriptor(nodeId));
        }
        for (int nodeId : candidate.outputNodeIds()) {
            outputBytes += nodeBytes(context.descriptor(nodeId));
        }
        for (int nodeId : candidate.orderedNodeIds()) {
            if (!outputs.contains(nodeId)) {
                avoidedIntermediateBytes += nodeBytes(context.descriptor(nodeId));
            }
        }
        return new AcceleratorPartitionScoreModel.TransferMetrics(inputBytes, outputBytes, avoidedIntermediateBytes);
    }

    private AcceleratorPartitionScoreModel.MaterializationSignals materializationSignalsFor(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    ) {
        AcceleratorPartitionScoreModel.TransferMetrics transfers = transferMetricsFor(candidate, context);
        return new AcceleratorPartitionScoreModel.MaterializationSignals(
                candidate.externalInputIds().size() + candidate.outputNodeIds().size(),
                transfers.inputBytes(),
                transfers.outputBytes(),
                0L,
                layoutFallbackBytesFor(candidate, context),
                transfers.avoidedIntermediateBytes(),
                "BUFFER_BINDING",
                layoutClassFor(candidate, context)
        );
    }

    private long layoutFallbackBytesFor(PartitionCandidate candidate, PartitionPlanningContext context) {
        long bytes = 0L;
        for (int nodeId : candidate.externalInputIds()) {
            CompiledTensorDescriptor descriptor = context.descriptor(nodeId);
            if (requiresDenseInputMaterialization(descriptor)) {
                bytes += nodeBytes(descriptor);
            }
        }
        for (int nodeId : candidate.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (isExplicitDenseLayoutMaterialization(node)) {
                bytes += nodeBytes(context.descriptor(nodeId));
            }
        }
        for (int nodeId : candidate.outputNodeIds()) {
            CompiledTensorDescriptor descriptor = context.descriptor(nodeId);
            if (descriptor != null && (!descriptor.contiguous() || descriptor.hasStorageOffset())) {
                bytes += nodeBytes(descriptor);
            }
        }
        return bytes;
    }

    private static boolean requiresDenseInputMaterialization(CompiledTensorDescriptor descriptor) {
        return descriptor != null && (!descriptor.contiguous() || descriptor.hasStorageOffset());
    }

    private static boolean isExplicitDenseLayoutMaterialization(CompiledNode node) {
        return node != null
                && node.operation() != null
                && switch (node.operation().opType()) {
                    case CONTIGUOUS -> true;
                    default -> false;
                };
    }

    private String layoutClassFor(PartitionCandidate candidate, PartitionPlanningContext context) {
        boolean hasStrided = false;
        boolean hasOffset = false;
        for (int nodeId : candidate.outputNodeIds()) {
            CompiledTensorDescriptor descriptor = context.descriptor(nodeId);
            if (descriptor == null) {
                continue;
            }
            if (descriptor.layoutClass() == LayoutClass.BROADCAST_ZERO_STRIDE) {
                return "BROADCAST_ZERO_STRIDE";
            }
            if (!descriptor.contiguous()) {
                hasStrided = true;
            }
            if (descriptor.hasStorageOffset()) {
                hasOffset = true;
            }
        }
        if (hasStrided) {
            return "PERMUTED_OR_STRIDED_VIEW";
        }
        if (hasOffset) {
            return "NON_ZERO_OFFSET_VIEW";
        }
        return "DENSE_CONTIGUOUS";
    }

    private static long nodeBytes(CompiledTensorDescriptor descriptor) {
        if (descriptor == null) {
            return 0L;
        }
        return descriptor.logicalByteLength();
    }

    private boolean isBetterStructural(
            PartitionCandidate candidate,
            double candidateScore,
            PartitionCandidate currentBest,
            double currentBestScore
    ) {
        if (candidate == null) {
            return false;
        }
        if (currentBest == null) {
            return true;
        }
        if (Double.compare(candidateScore, currentBestScore) != 0) {
            return candidateScore > currentBestScore;
        }
        if (candidate.orderedNodeIds().size() != currentBest.orderedNodeIds().size()) {
            return candidate.orderedNodeIds().size() > currentBest.orderedNodeIds().size();
        }
        return candidate.anchorNodeId() < currentBest.anchorNodeId();
    }

    private boolean isBetterAccepted(
            PartitionCandidate candidate,
            double candidateScore,
            long candidateWork,
            PartitionCandidate currentBest,
            double currentBestScore,
            long currentBestWork
    ) {
        if (candidate == null) {
            return false;
        }
        if (currentBest == null) {
            return true;
        }
        if (Double.compare(candidateScore, currentBestScore) != 0) {
            return candidateScore > currentBestScore;
        }
        if (candidateWork != currentBestWork) {
            return candidateWork > currentBestWork;
        }
        if (candidate.orderedNodeIds().size() != currentBest.orderedNodeIds().size()) {
            return candidate.orderedNodeIds().size() > currentBest.orderedNodeIds().size();
        }
        return candidate.anchorNodeId() < currentBest.anchorNodeId();
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
                .map(nodeId -> new PartitionValue(PartitionValueRef.ofNode(nodeId), nodeId))
                .toList();
        List<PartitionValueRef> outputValueRefs = candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList();
        List<PartitionValueRef> requiredMaterialized = candidate.outputNodeIds().stream()
                .map(PartitionValueRef::ofNode)
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
                attachedPlan == null ? 0L : attachedPlan.estimatedWork(),
                metricsFor(candidate, context),
                request.strategy(),
                trace
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
        return prefix + "-" + anchorNodeId;
    }

    private static final class SearchAccumulator {
        private PartitionCandidate bestAccepted;
        private double bestAcceptedScore = Double.NEGATIVE_INFINITY;
        private long bestAcceptedWork;
        private AcceleratorPartitionScoreModel.MaterializationCostSummary bestAcceptedCostSummary;
        private PartitionCandidate bestStructural;
        private double bestStructuralScore = Double.NEGATIVE_INFINITY;
        private int exploredCandidates;
        private boolean searchBudgetHit;
        private final List<PartitionDecisionTrace.CandidateCostTrace> topRejectedFinalists = new ArrayList<>();

        private void addFinalist(
                PartitionCandidate candidate,
                String reason,
                AcceleratorPartitionScoreModel.MaterializationCostSummary summary
        ) {
            if (candidate == null || summary == null) {
                return;
            }
            topRejectedFinalists.add(new PartitionDecisionTrace.CandidateCostTrace(
                    candidate.orderedNodeIds(),
                    reason,
                    summary.finalScore(),
                    summary.boundaryCount(),
                    summary.estimatedTransferBytes(),
                    summary.layoutFallbackBytes(),
                    summary.estimatedComputeWork(),
                    summary.preset()
            ));
            topRejectedFinalists.sort(
                    Comparator.comparingDouble(PartitionDecisionTrace.CandidateCostTrace::finalScore).reversed()
            );
            while (topRejectedFinalists.size() > 3) {
                topRejectedFinalists.removeLast();
            }
        }

        private List<PartitionDecisionTrace.CandidateCostTrace> topRejectedFinalists() {
            return List.copyOf(topRejectedFinalists);
        }
    }
}
