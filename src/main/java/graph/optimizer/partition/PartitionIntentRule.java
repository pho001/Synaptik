package graph.optimizer.partition;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.optimizer.CpuRegionConfig;
import config.optimizer.CpuRegionPolicy;
import config.optimizer.OffloadConfig;
import config.optimizer.OffloadPolicy;
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

/**
 * Optimizer stage that turns backend intent into partition plans.
 *
 * <p>The rule first propagates backend intent through backward closures, snapshots the graph into {@link CompiledNode}
 * values, resolves a target backend, and delegates candidate construction to a {@link PartitionPlanner}. Accepted
 * partitions and backend-specific {@link PartitionPlan}s are stored on {@link OptimizerState} for the later region
 * optimization and memory planning stages.
 *
 * <p>The rule mutates backend intent metadata on tensors while applying propagation and is intended for single-threaded
 * compile-time use.
 */
public final class PartitionIntentRule implements OptimizationRule {
    private final PartitionConfig config;
    private final OffloadConfig offloadConfig;
    private final CpuRegionConfig cpuRegionConfig;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;

    /**
     * Creates a partition rule with default configuration and backend descriptors.
     */
    public PartitionIntentRule() {
        this(PartitionConfig.defaults(), OffloadConfig.defaults(), CpuRegionConfig.defaults());
    }

    /**
     * Creates a partition rule with default backend descriptors.
     *
     * @param config partition configuration, or {@code null} for defaults
     */
    public PartitionIntentRule(PartitionConfig config) {
        this(config, OffloadConfig.defaults(), CpuRegionConfig.defaults(), BackendPartitionDescriptorRegistry.defaults());
    }

    /**
     * Creates a partition rule with graph region policies and default backend descriptors.
     *
     * @param config shared partition search configuration
     * @param offloadConfig accelerator/offload policy
     * @param cpuRegionConfig CPU execution region policy
     */
    public PartitionIntentRule(
            PartitionConfig config,
            OffloadConfig offloadConfig,
            CpuRegionConfig cpuRegionConfig
    ) {
        this(config, offloadConfig, cpuRegionConfig, BackendPartitionDescriptorRegistry.defaults());
    }

    /**
     * Creates a partition rule.
     *
     * @param config partition configuration, or {@code null} for defaults
     * @param backendPartitionDescriptors registry that supplies backend legality adapters and lowerers
     */
    public PartitionIntentRule(PartitionConfig config, BackendPartitionDescriptorRegistry backendPartitionDescriptors) {
        this(config, OffloadConfig.defaults(), CpuRegionConfig.defaults(), backendPartitionDescriptors);
    }

    /**
     * Creates a partition rule.
     *
     * @param config partition configuration, or {@code null} for defaults
     * @param offloadConfig accelerator/offload policy, or {@code null} for defaults
     * @param cpuRegionConfig CPU execution region policy, or {@code null} for defaults
     * @param backendPartitionDescriptors registry that supplies backend legality adapters and lowerers
     */
    public PartitionIntentRule(
            PartitionConfig config,
            OffloadConfig offloadConfig,
            CpuRegionConfig cpuRegionConfig,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this.config = config == null ? PartitionConfig.defaults() : config;
        this.offloadConfig = offloadConfig == null ? OffloadConfig.defaults() : offloadConfig;
        this.cpuRegionConfig = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
    }

    /**
     * Plans backend partitions for the state's graph.
     *
     * @param state optimizer state after rewrite/CSE
     * @return state with partition metadata attached, or empty partition metadata when no target is available
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return state;
        }
        BackendIntentPropagator.propagateBackwardClosure(sortedGraph);
        OptimizerState rewritten = state.withGraph(sortedGraph, state.forwardOutput());
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(rewritten.graph());
        List<PlanningJob> jobs = resolvePlanningJobs(compiledNodes);
        if (jobs.isEmpty()) {
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
        List<Partition> partitions = new ArrayList<>();
        Map<String, PartitionPlan> plansByPartitionId = new java.util.LinkedHashMap<>();
        Set<PartitionValueRef> requiredMaterialized = requiredMaterializedValueRefs(compiledNodes, rewritten.forwardOutput());
        for (PlanningJob job : jobs) {
            PartitionPlanningResult planning = selectPlanner(job.strategy()).plan(
                    new PartitionPlanningRequest(
                            job.strategy(),
                            job.target(),
                            planningContext,
                            job.policy(),
                            backendPartitionDescriptors.legalityAdapterFor(job.target()),
                            job.sourcePolicy(),
                            requiredMaterialized,
                            job.cpuRegionConfig()
                    )
            );
            partitions.addAll(planning.partitions());
            plansByPartitionId.putAll(planning.plansByPartitionId());
        }
        return rewritten.withPartitions(partitions, plansByPartitionId);
    }

    private List<PlanningJob> resolvePlanningJobs(List<CompiledNode> compiledNodes) {
        PartitionTarget configured = config.target();
        if (configured != null && !configured.isAuto()) {
            if (configured.isNone()) {
                return List.of();
            }
            return List.of(new PlanningJob(
                    configured,
                    configured == PartitionTarget.CPU
                            ? PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION
                            : config.plannerStrategy(),
                    configured == PartitionTarget.CPU
                            ? cpuPlannerPolicy()
                            : AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config),
                    PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                    configured == PartitionTarget.CPU ? cpuRegionConfig : CpuRegionConfig.defaults()
            ));
        }
        boolean cpuSeen = false;
        boolean metalSeen = false;
        boolean cudaSeen = false;
        for (CompiledNode node : compiledNodes) {
            PartitionTarget target = PartitionTarget.fromBackend(node.backend());
            if (target == PartitionTarget.CPU) {
                cpuSeen = true;
            } else if (target == PartitionTarget.GPU_METAL) {
                metalSeen = true;
            } else if (target == PartitionTarget.GPU_CUDA) {
                cudaSeen = true;
            }
        }
        List<PlanningJob> jobs = new ArrayList<>();
        boolean planAcceleratorOffload = offloadConfig.policy() == OffloadPolicy.ACCELERATOR_IF_PROFITABLE;
        if (planAcceleratorOffload || metalSeen || cudaSeen) {
            PartitionPlannerStrategy acceleratorStrategy = acceleratorStrategy();
            if (acceleratorStrategy == null && (metalSeen || cudaSeen)) {
                acceleratorStrategy = config.plannerStrategy();
            }
            if (acceleratorStrategy != null) {
                if (metalSeen || (planAcceleratorOffload && cpuSeen)) {
                    jobs.add(new PlanningJob(
                            PartitionTarget.GPU_METAL,
                            acceleratorStrategy,
                            AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config),
                            planAcceleratorOffload
                                    ? PartitionSourcePolicy.CPU_OR_TARGET_BACKEND
                                    : PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                            CpuRegionConfig.defaults()
                    ));
                }
                if (cudaSeen) {
                    jobs.add(new PlanningJob(
                            PartitionTarget.GPU_CUDA,
                            acceleratorStrategy,
                            AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config),
                            PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                            CpuRegionConfig.defaults()
                    ));
                }
            }
        }
        if (cpuSeen && cpuRegionConfig.policy() != CpuRegionPolicy.OFF) {
            jobs.add(new PlanningJob(
                    PartitionTarget.CPU,
                    PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION,
                    cpuPlannerPolicy(),
                    PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                    cpuRegionConfig
            ));
        }
        return List.copyOf(jobs);
    }

    private PartitionPlannerStrategy acceleratorStrategy() {
        return switch (offloadConfig.acceleratorRegionPolicy()) {
            case OFF -> null;
            case GREEDY_CLOSED_REGIONS -> PartitionPlannerStrategy.GREEDY_MAX_REGION;
            case SCORED_PROFITABLE_REGIONS -> PartitionPlannerStrategy.SCORED_CANDIDATE_SEARCH;
        };
    }

    private AcceleratorPartitionScoreModel.PlannerPolicy cpuPlannerPolicy() {
        AcceleratorPartitionScoreModel.PlannerPolicy base = AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config);
        return new AcceleratorPartitionScoreModel.PlannerPolicy(
                cpuRegionConfig.maxRegionNodes(),
                base.maxVisitedCandidates(),
                base.nodeWeight(),
                base.internalEdgeWeight(),
                base.mergeNodeBonus(),
                base.tailDepthWeight(),
                base.externalInputPenalty(),
                base.workWeight()
        );
    }

    private PartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new ScoredCandidatePartitionPlanner();
            case CPU_NATURAL_EXECUTION_REGION -> new CpuNaturalExecutionRegionPlanner();
        };
    }

    private record PlanningJob(
            PartitionTarget target,
            PartitionPlannerStrategy strategy,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            PartitionSourcePolicy sourcePolicy,
            CpuRegionConfig cpuRegionConfig
    ) {
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
