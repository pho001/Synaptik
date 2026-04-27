package graph.optimizer.partition;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.optimizer.PartitionConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.optimizer.OptimizationRule;
import graph.optimizer.intent.BackendIntentPropagator;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class PartitionIntentRule implements OptimizationRule {
    private final PartitionConfig config;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;

    public PartitionIntentRule() {
        this(PartitionConfig.defaults());
    }

    public PartitionIntentRule(PartitionConfig config) {
        this(config, BackendPartitionDescriptorRegistry.defaults());
    }

    public PartitionIntentRule(PartitionConfig config, BackendPartitionDescriptorRegistry backendPartitionDescriptors) {
        this.config = config == null ? PartitionConfig.defaults() : config;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
    }

    @Override
    public OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return state;
        }
        BackendIntentPropagator.propagateBackwardClosure(sortedGraph);
        OptimizerState rewritten = state.withGraph(sortedGraph, state.forwardOutput());
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(rewritten.graph());
        PartitionTarget target = resolvePartitionTarget(compiledNodes);
        if (target.isNone()) {
            return rewritten.withPartitions(List.of());
        }
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                rewritten.supportsBackward()
                        ? RuntimeConfig.trainingDefaults()
                        : RuntimeConfig.inferenceDefaults(),
                rewritten.supportsBackward(),
                compiledNodes,
                buildConsumerMap(compiledNodes)
        );
        PartitionPlanningResult planning = selectPlanner(config.plannerStrategy()).plan(
                new PartitionPlanningRequest(
                        config.plannerStrategy(),
                        target,
                        planningContext,
                        AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config),
                        backendPartitionDescriptors.legalityAdapterFor(target),
                        requiredMaterializedValueRefs(compiledNodes, rewritten.forwardOutput())
                )
        );
        return rewritten.withPartitions(planning.partitions(), planning.plansByPartitionId());
    }

    private PartitionTarget resolvePartitionTarget(List<CompiledNode> compiledNodes) {
        PartitionTarget configured = config.target();
        if (configured != null && !configured.isAuto()) {
            return configured;
        }
        boolean cpuSeen = false;
        for (CompiledNode node : compiledNodes) {
            PartitionTarget target = PartitionTarget.fromBackend(node.backend());
            if (target == PartitionTarget.GPU_METAL || target == PartitionTarget.GPU_CUDA) {
                return target;
            }
            if (target == PartitionTarget.CPU) {
                cpuSeen = true;
            }
        }
        return cpuSeen ? PartitionTarget.CPU : PartitionTarget.NONE;
    }

    private PartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new ScoredCandidatePartitionPlanner();
        };
    }

    private static Map<Integer, List<CompiledNode>> buildConsumerMap(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private static Set<PartitionValueRef> requiredMaterializedValueRefs(List<CompiledNode> compiledNodes, Tensor forwardOutput) {
        java.util.LinkedHashSet<PartitionValueRef> required = new java.util.LinkedHashSet<>();
        if (forwardOutput != null) {
            int forwardOutputIndex = compiledNodes.stream()
                    .filter(node -> node.semanticTensor() == forwardOutput)
                    .map(CompiledNode::id)
                    .findFirst()
                    .orElse(-1);
            if (forwardOutputIndex >= 0) {
                required.add(PartitionValueRef.ofNode(forwardOutputIndex));
            }
        }
        for (CompiledNode node : compiledNodes) {
            Tensor gradient = node.semanticTensor().getGradient();
            if (gradient == null) {
                continue;
            }
            int gradientNodeId = compiledNodes.stream()
                    .filter(candidate -> candidate.semanticTensor() == gradient)
                    .map(CompiledNode::id)
                    .findFirst()
                    .orElse(-1);
            if (gradientNodeId >= 0) {
                required.add(PartitionValueRef.ofNode(gradientNodeId));
            }
        }
        return Set.copyOf(required);
    }
}
