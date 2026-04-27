package graph.compile;

import backend.ComputeBackend;
import backend.partition.BackendPartitionDescriptorRegistry;
import config.optimizer.PartitionConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.GreedyMaxRegionPartitionPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionPlanner;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionPlanningRequest;
import graph.optimizer.partition.PartitionPlanningResult;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValueRef;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class PartitionPlanningSnapshotBuilder {
    private PartitionPlanningSnapshotBuilder() {
    }

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

        public static Snapshot empty() {
            return new Snapshot(List.of(), List.of(), List.of(), PartitionCompileTrace.empty());
        }
    }

    public static Snapshot build(
            PartitionConfig partitionConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledNode forwardOutput,
            Map<?, CompiledGradientBinding> gradientBindings
    ) {
        return build(
                partitionConfig,
                supportsBackward,
                compiledNodes,
                forwardOutput,
                gradientBindings,
                BackendPartitionDescriptorRegistry.defaults()
        );
    }

    public static Snapshot build(
            PartitionConfig partitionConfig,
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
        PartitionTarget target = resolvePartitionTarget(resolvedConfig, nodes);
        if (target.isNone()) {
            return Snapshot.empty();
        }

        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                supportsBackward ? RuntimeConfig.trainingDefaults() : RuntimeConfig.inferenceDefaults(),
                supportsBackward,
                nodes,
                buildConsumerMap(nodes)
        );
        PartitionPlanningResult planning = selectPlanner(resolvedConfig.plannerStrategy()).plan(
                new PartitionPlanningRequest(
                        resolvedConfig.plannerStrategy(),
                        target,
                        planningContext,
                        graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.PlannerPolicy.fromConfig(resolvedConfig),
                        descriptors.legalityAdapterFor(target),
                        requiredMaterializedValueRefs(forwardOutput, gradientBindings)
                )
        );
        return new Snapshot(
                planning.partitions(),
                planning.attachedPlans(),
                backendSelectionCandidates(planning),
                planning.trace()
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

    private static PartitionTarget resolvePartitionTarget(PartitionConfig config, List<CompiledNode> nodes) {
        PartitionTarget configured = config.target();
        if (configured != null && !configured.isAuto()) {
            return configured;
        }
        boolean cpuSeen = false;
        for (CompiledNode node : nodes) {
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

    private static PartitionPlanner selectPlanner(PartitionPlannerStrategy strategy) {
        PartitionPlannerStrategy resolved = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        return switch (resolved) {
            case GREEDY_MAX_REGION -> new GreedyMaxRegionPartitionPlanner();
            case SCORED_CANDIDATE_SEARCH -> new graph.optimizer.partition.ScoredCandidatePartitionPlanner();
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
