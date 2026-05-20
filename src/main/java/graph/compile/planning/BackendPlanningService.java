package graph.compile.planning;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.BackendPlanningConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.AnchorBasedPartitionPlanner;
import graph.compile.planning.partition.CpuNaturalExecutionRegionPlanner;
import graph.compile.planning.partition.GreedyMaxRegionPartitionPlanner;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PartitionPlanner;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionPlanningContext;
import graph.compile.planning.partition.PartitionPlanningRequest;
import graph.compile.planning.partition.PartitionPlanningResult;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.partition.ScoredCandidatePartitionPlanner;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Authoritative compile-time backend ownership planner.
 */
public final class BackendPlanningService {
    private final BackendPlanningJobResolver resolver;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;

    public BackendPlanningService(
            BackendPlanningJobResolver resolver,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this.resolver = resolver == null ? new BackendPlanningJobResolver() : resolver;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
    }

    public static BackendPlanningService defaults() {
        return new BackendPlanningService(new BackendPlanningJobResolver(), BackendPartitionDescriptorRegistry.defaults());
    }

    public BackendPlanningResult plan(BackendPlanningRequest request) {
        if (request == null || request.compiledNodes().isEmpty()) {
            return BackendPlanningResult.empty();
        }
        BackendPlanningConfig config = request.config() == null ? BackendPlanningConfig.cpuOnly() : request.config();
        List<BackendPlanningJob> jobs = resolver.resolve(config, request.compiledNodes());
        List<ExplicitBackendIntent> explicitIntents = resolver.explicitIntents(request.compiledNodes());
        if (jobs.isEmpty()) {
            BackendPlanningRequirementValidator.validateRequired(config, explicitIntents, List.of());
            return new BackendPlanningResult(
                    jobs,
                    List.of(),
                    PartitionCompileTrace.empty()
            );
        }

        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                request.supportsBackward() ? RuntimeConfig.trainingDefaults() : RuntimeConfig.inferenceDefaults(),
                request.supportsBackward(),
                request.compiledNodes(),
                request.descriptorIndex(),
                buildConsumerMap(request.compiledNodes())
        );
        Set<GraphValueRef> requiredMaterialized = requiredMaterializedValueRefs(
                request.forwardOutput(),
                request.gradientBindings()
        );
        List<Partition> partitions = new ArrayList<>();
        LinkedHashMap<String, PartitionPlan> plansByPartitionId = new LinkedHashMap<>();
        List<PartitionCompileTrace> traces = new ArrayList<>();
        for (BackendPlanningJob job : jobs) {
            PartitionPlanningResult planning = selectPlanner(job.strategy()).plan(
                    new PartitionPlanningRequest(
                            job.strategy(),
                            job.target(),
                            planningContext,
                            job.policy(),
                            backendPartitionDescriptors.legalityAdapterFor(job.target()),
                            job.sourcePolicy(),
                            requiredMaterialized,
                            job.cpuRegionConfig(),
                            job.metalTransferModel()
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
        List<PlannedPartition> plannedPartitions = plannedPartitions(planning);
        BackendPlanningRequirementValidator.validateRequired(config, explicitIntents, planning.partitions());
        return new BackendPlanningResult(
                jobs,
                plannedPartitions,
                planning.trace()
        );
    }

    private static PartitionCompileTrace mergeTrace(List<BackendPlanningJob> jobs, List<PartitionCompileTrace> traces) {
        if (traces.isEmpty()) {
            return PartitionCompileTrace.empty();
        }
        if (traces.size() == 1) {
            return traces.getFirst();
        }
        List<PartitionDecisionTrace> decisions = new ArrayList<>();
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
        BackendPlanningJob first = jobs.getFirst();
        return new PartitionCompileTrace(
                first.strategy(),
                first.target(),
                considered,
                accepted,
                rejected,
                decisions
        );
    }

    private static List<PlannedPartition> plannedPartitions(PartitionPlanningResult planning) {
        return planning.partitions().stream()
                .map(partition -> Map.entry(partition, planning.planForPartition(partition.partitionId())))
                .filter(entry -> entry.getValue() != null)
                .map(partition -> new PlannedPartition(
                        partition.getKey(),
                        partition.getValue(),
                        Set.of(partition.getValue().backend())
                ))
                .toList();
    }

    private static Set<GraphValueRef> requiredMaterializedValueRefs(
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings
    ) {
        LinkedHashSet<GraphValueRef> required = new LinkedHashSet<>();
        if (forwardOutput != null && !forwardOutput.inputIds().isEmpty()) {
            required.add(GraphValueRef.node(forwardOutput.inputIds().getFirst()));
        }
        if (gradientBindings != null) {
            for (CompiledGradientBinding binding : gradientBindings.values()) {
                if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                    required.add(GraphValueRef.node(nodeBinding.nodeId()));
                }
            }
        }
        return Set.copyOf(required);
    }

    private static PartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case ANCHOR_MAX_REGION -> new AnchorBasedPartitionPlanner();
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new ScoredCandidatePartitionPlanner();
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
