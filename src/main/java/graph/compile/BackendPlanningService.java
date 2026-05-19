package graph.compile;

import backend.ComputeBackend;
import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.BackendDiscoveryMode;
import config.compile.BackendPlanningConfig;
import config.compile.BackendPlanningFailurePolicy;
import config.compile.BackendPlanningRequirementScope;
import config.compile.BackendTarget;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.AnchorBasedPartitionPlanner;
import graph.optimizer.partition.CpuNaturalExecutionRegionPlanner;
import graph.optimizer.partition.GreedyMaxRegionPartitionPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionPlanner;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionPlanningRequest;
import graph.optimizer.partition.PartitionPlanningResult;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.GraphValueRef;
import graph.optimizer.partition.PlannedPartition;
import graph.optimizer.partition.ScoredCandidatePartitionPlanner;

import java.util.ArrayList;
import java.util.EnumSet;
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
        List<BackendPlanningDiagnostic> diagnostics = new ArrayList<>();
        EnumSet<BackendTarget> explicitTargets = resolver.explicitTargets(request.compiledNodes());
        if (jobs.isEmpty()) {
            validateRequired(config, explicitTargets, List.of(), diagnostics);
            return new BackendPlanningResult(
                    jobs,
                    List.of(),
                    PartitionCompileTrace.empty(),
                    diagnostics
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
        validateRequired(config, explicitTargets, planning.partitions(), diagnostics);
        return new BackendPlanningResult(
                jobs,
                plannedPartitions,
                planning.trace(),
                diagnostics
        );
    }

    private static void validateRequired(
            BackendPlanningConfig config,
            EnumSet<BackendTarget> explicitTargets,
            List<Partition> partitions,
            List<BackendPlanningDiagnostic> diagnostics
    ) {
        if (config.failurePolicy() == BackendPlanningFailurePolicy.OPTIONAL) {
            return;
        }
        if (config.failurePolicy() == BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION) {
            Set<BackendTarget> accepted = acceptedAcceleratorTargets(partitions);
            boolean ok = config.requirementScope() == BackendPlanningRequirementScope.EACH_TARGET
                    ? accepted.containsAll(config.targets())
                    : !accepted.isEmpty();
            if (!ok) {
                diagnostics.add(new BackendPlanningDiagnostic(
                        BackendPlanningDiagnostic.Severity.ERROR,
                        "REQUIRED_ACCELERATOR_REGION_MISSING",
                        "Required accelerator backend planning produced no legal region"
                ));
                throw new IllegalStateException("Required accelerator backend planning produced no legal region");
            }
        }
        if (config.failurePolicy() == BackendPlanningFailurePolicy.REQUIRE_ALL_EXPLICIT_INTENTS) {
            Set<BackendTarget> accepted = acceptedAcceleratorTargets(partitions);
            if (!accepted.containsAll(explicitTargets)) {
                diagnostics.add(new BackendPlanningDiagnostic(
                        BackendPlanningDiagnostic.Severity.ERROR,
                        "REQUIRED_EXPLICIT_BACKEND_INTENT_MISSING",
                        "One or more explicit backend intents could not be planned"
                ));
                throw new IllegalStateException("One or more explicit backend intents could not be planned");
            }
        }
    }

    private static Set<BackendTarget> acceptedAcceleratorTargets(List<Partition> partitions) {
        EnumSet<BackendTarget> out = EnumSet.noneOf(BackendTarget.class);
        for (Partition partition : partitions == null ? List.<Partition>of() : partitions) {
            BackendTarget target = BackendTarget.fromPartitionTarget(partition.target());
            if (target != null && target.accelerator()) {
                out.add(target);
            }
        }
        return out;
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
