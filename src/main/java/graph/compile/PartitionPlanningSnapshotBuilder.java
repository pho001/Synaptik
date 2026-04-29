package graph.compile;

import backend.ComputeBackend;
import backend.partition.BackendPartitionDescriptorRegistry;
import config.optimizer.CpuRegionConfig;
import config.optimizer.CpuRegionPolicy;
import config.optimizer.OffloadConfig;
import config.optimizer.OffloadPolicy;
import config.optimizer.PartitionConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.CpuNaturalExecutionRegionPlanner;
import graph.optimizer.partition.GreedyMaxRegionPartitionPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionPlanner;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionPlanningRequest;
import graph.optimizer.partition.PartitionPlanningResult;
import graph.optimizer.partition.PartitionSourcePolicy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValueRef;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Builds a partition planning snapshot from compiled nodes.
 *
 * <p>This helper mirrors the partition optimizer stage at the compile artifact boundary so preparation can inspect
 * accepted partitions, backend plans, accelerator candidates, and partition trace metadata without re-running the full
 * optimizer pipeline.
 */
public final class PartitionPlanningSnapshotBuilder {
    private PartitionPlanningSnapshotBuilder() {
    }

    /**
     * Immutable partition planning snapshot.
     *
     * @param partitions accepted partitions
     * @param backendPlans attached backend plans
     * @param backendSelectionCandidates non-CPU candidates available for backend selection
     * @param trace partition planning trace
     */
    public record Snapshot(
            List<Partition> partitions,
            List<PartitionPlan> backendPlans,
            List<BackendCandidatePartition> backendSelectionCandidates,
            PartitionCompileTrace trace
    ) {
        public Snapshot {
            partitions = List.copyOf(partitions == null ? List.of() : partitions);
            backendPlans = List.copyOf(backendPlans == null ? List.of() : backendPlans);
            backendSelectionCandidates = List.copyOf(backendSelectionCandidates == null ? List.of() : backendSelectionCandidates);
            trace = trace == null ? PartitionCompileTrace.empty() : trace;
        }

        /**
         * Returns an empty snapshot.
         *
         * @return empty snapshot
         */
        public static Snapshot empty() {
            return new Snapshot(List.of(), List.of(), List.of(), PartitionCompileTrace.empty());
        }
    }

    /**
     * Builds a snapshot with the default backend partition descriptor registry.
     *
     * @param partitionConfig partition configuration
     * @param supportsBackward whether the compiled graph contains backward work
     * @param compiledNodes compiled node snapshots
     * @param forwardOutput compiled forward output node
     * @param gradientBindings gradient publication bindings
     * @return partition planning snapshot
     */
    public static Snapshot build(
            PartitionConfig partitionConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings
    ) {
        return build(
                partitionConfig,
                OffloadConfig.defaults(),
                CpuRegionConfig.defaults(),
                supportsBackward,
                compiledNodes,
                forwardOutput,
                gradientBindings,
                BackendPartitionDescriptorRegistry.defaults()
        );
    }

    /**
     * Builds a snapshot with an explicit backend partition descriptor registry.
     *
     * @param partitionConfig partition configuration
     * @param supportsBackward whether the compiled graph contains backward work
     * @param compiledNodes compiled node snapshots
     * @param forwardOutput compiled forward output node
     * @param gradientBindings gradient publication bindings
     * @param backendPartitionDescriptors backend descriptor registry
     * @return partition planning snapshot
     */
    public static Snapshot build(
            PartitionConfig partitionConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        return build(
                partitionConfig,
                OffloadConfig.defaults(),
                CpuRegionConfig.defaults(),
                supportsBackward,
                compiledNodes,
                forwardOutput,
                gradientBindings,
                backendPartitionDescriptors
        );
    }

    /**
     * Builds a snapshot with explicit graph region/offload policies.
     *
     * @param partitionConfig shared partition search configuration
     * @param offloadConfig accelerator/offload policy
     * @param cpuRegionConfig CPU execution region policy
     * @param supportsBackward whether the compiled graph contains backward work
     * @param compiledNodes compiled node snapshots
     * @param forwardOutput compiled forward output node
     * @param gradientBindings gradient publication bindings
     * @param backendPartitionDescriptors backend descriptor registry
     * @return partition planning snapshot
     */
    public static Snapshot build(
            PartitionConfig partitionConfig,
            OffloadConfig offloadConfig,
            CpuRegionConfig cpuRegionConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        List<CompiledNode> nodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        if (nodes.isEmpty()) {
            return Snapshot.empty();
        }
        BackendPartitionDescriptorRegistry descriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
        PartitionConfig resolvedConfig = partitionConfig == null ? PartitionConfig.defaults() : partitionConfig;
        OffloadConfig resolvedOffload = offloadConfig == null ? OffloadConfig.defaults() : offloadConfig;
        CpuRegionConfig resolvedCpuRegion = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
        List<PlanningJob> jobs = resolvePlanningJobs(resolvedConfig, resolvedOffload, resolvedCpuRegion, nodes);
        if (jobs.isEmpty()) {
            return Snapshot.empty();
        }

        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                supportsBackward ? RuntimeConfig.trainingDefaults() : RuntimeConfig.inferenceDefaults(),
                supportsBackward,
                nodes,
                buildConsumerMap(nodes)
        );
        List<Partition> partitions = new ArrayList<>();
        java.util.LinkedHashMap<String, PartitionPlan> plansByPartitionId = new java.util.LinkedHashMap<>();
        List<PartitionCompileTrace> traces = new ArrayList<>();
        Set<PartitionValueRef> requiredMaterialized = requiredMaterializedValueRefs(forwardOutput, gradientBindings);
        for (PlanningJob job : jobs) {
            PartitionPlanningResult planning = selectPlanner(job.strategy()).plan(
                    new PartitionPlanningRequest(
                            job.strategy(),
                            job.target(),
                            planningContext,
                            job.policy(),
                            descriptors.legalityAdapterFor(job.target()),
                            job.sourcePolicy(),
                            requiredMaterialized,
                            job.cpuRegionConfig()
                    )
            );
            partitions.addAll(planning.partitions());
            plansByPartitionId.putAll(planning.plansByPartitionId());
            traces.add(planning.trace());
        }
        PartitionPlanningResult planning = new PartitionPlanningResult(
                partitions,
                plansByPartitionId,
                mergeTrace(jobs, traces)
        );
        return new Snapshot(
                planning.partitions(),
                planning.attachedPlans(),
                backendSelectionCandidates(planning),
                planning.trace()
        );
    }

    private record PlanningJob(
            PartitionTarget target,
            PartitionPlannerStrategy strategy,
            graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy policy,
            PartitionSourcePolicy sourcePolicy,
            CpuRegionConfig cpuRegionConfig
    ) {
    }

    private static List<PlanningJob> resolvePlanningJobs(
            PartitionConfig config,
            OffloadConfig offloadConfig,
            CpuRegionConfig cpuRegionConfig,
            List<CompiledNode> nodes
    ) {
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
                    plannerPolicyFor(config, configured == PartitionTarget.CPU ? cpuRegionConfig : null),
                    PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                    configured == PartitionTarget.CPU ? cpuRegionConfig : CpuRegionConfig.defaults()
            ));
        }
        boolean cpuSeen = false;
        boolean metalSeen = false;
        boolean cudaSeen = false;
        for (CompiledNode node : nodes) {
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
            PartitionPlannerStrategy acceleratorStrategy = acceleratorStrategy(offloadConfig);
            if (acceleratorStrategy == null && (metalSeen || cudaSeen)) {
                acceleratorStrategy = config.plannerStrategy();
            }
            if (acceleratorStrategy != null) {
                var policy = graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config);
                if (metalSeen || (planAcceleratorOffload && cpuSeen)) {
                    jobs.add(new PlanningJob(
                            PartitionTarget.GPU_METAL,
                            acceleratorStrategy,
                            policy,
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
                            policy,
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
                    plannerPolicyFor(config, cpuRegionConfig),
                    PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                    cpuRegionConfig
            ));
        }
        return List.copyOf(jobs);
    }

    private static PartitionPlannerStrategy acceleratorStrategy(OffloadConfig offloadConfig) {
        return switch (offloadConfig.acceleratorRegionPolicy()) {
            case OFF -> null;
            case GREEDY_CLOSED_REGIONS -> PartitionPlannerStrategy.GREEDY_MAX_REGION;
            case SCORED_PROFITABLE_REGIONS -> PartitionPlannerStrategy.SCORED_CANDIDATE_SEARCH;
        };
    }

    private static graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy plannerPolicyFor(
            PartitionConfig config,
            CpuRegionConfig cpuRegionConfig
    ) {
        var base = graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(config);
        if (cpuRegionConfig == null) {
            return base;
        }
        return new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy(
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

    private static PartitionCompileTrace mergeTrace(List<PlanningJob> jobs, List<PartitionCompileTrace> traces) {
        if (traces.isEmpty()) {
            return PartitionCompileTrace.empty();
        }
        if (traces.size() == 1) {
            return traces.getFirst();
        }
        List<graph.execution.trace.PartitionDecisionTrace> decisions = new ArrayList<>();
        int considered = 0;
        int accepted = 0;
        int rejected = 0;
        for (PartitionCompileTrace trace : traces) {
            if (trace == null) {
                continue;
            }
            considered += trace.totalConsidered();
            accepted += trace.acceptedCount();
            rejected += trace.rejectedCount();
            decisions.addAll(trace.decisions());
        }
        PlanningJob first = jobs.getFirst();
        return new PartitionCompileTrace(
                first.strategy(),
                first.target(),
                considered,
                accepted,
                rejected,
                decisions
        );
    }

    private static List<BackendCandidatePartition> backendSelectionCandidates(PartitionPlanningResult planning) {
        return planning.partitions().stream()
                .map(partition -> Map.entry(partition, planning.planForPartition(partition.partitionId())))
                .filter(entry -> entry.getValue() != null)
                .filter(entry -> entry.getValue().backend() != ComputeBackend.CPU)
                .map(partition -> new BackendCandidatePartition(
                        partition.getKey(),
                        Set.of(partition.getValue().backend()),
                        partition.getValue()
                ))
                .toList();
    }

    private static Set<PartitionValueRef> requiredMaterializedValueRefs(
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings
    ) {
        LinkedHashSet<PartitionValueRef> required = new LinkedHashSet<>();
        if (forwardOutput != null && !forwardOutput.inputIds().isEmpty()) {
            required.add(PartitionValueRef.ofNode(forwardOutput.inputIds().getFirst()));
        }
        if (gradientBindings != null) {
            for (CompiledGradientBinding binding : gradientBindings.values()) {
                if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                    required.add(PartitionValueRef.ofNode(nodeBinding.nodeId()));
                }
            }
        }
        return Set.copyOf(required);
    }

    private static PartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new graph.optimizer.partition.ScoredCandidatePartitionPlanner();
            case CPU_NATURAL_EXECUTION_REGION -> new CpuNaturalExecutionRegionPlanner();
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
}
