package planning.backend;

import config.compile.BackendDiscoveryMode;
import config.compile.BackendPlanningConfig;
import config.compile.BackendTarget;
import config.compile.PartitionSearchConfig;
import config.optimizer.CpuPartitionConfig;
import config.optimizer.CpuPartitionPolicy;
import graph.model.CompiledNode;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionSourcePolicy;
import planning.partition.PartitionTarget;
import planning.partition.cost.AcceleratorPartitionScoreModel;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Single resolver for backend planning jobs.
 */
public final class BackendPlanningJobResolver {

    public List<BackendPlanningJob> resolve(BackendPlanningConfig config, List<CompiledNode> nodes) {
        BackendPlanningConfig resolved = config == null ? BackendPlanningConfig.cpuOnly() : config;
        List<CompiledNode> graph = nodes == null ? List.of() : List.copyOf(nodes);
        if (graph.isEmpty()) {
            return List.of();
        }

        EnumSet<BackendTarget> explicitTargets = explicitTargets(graph);
        boolean cpuSeen = graph.stream().anyMatch(node -> node.backend() == backend.contract.ComputeBackend.CPU);
        List<BackendPlanningJob> jobs = new ArrayList<>();

        if (resolved.discoveryMode() != BackendDiscoveryMode.CPU_ONLY) {
            for (BackendTarget target : resolved.targets()) {
                if (!target.accelerator()) {
                    continue;
                }
                boolean explicitSeen = explicitTargets.contains(target);
                if (resolved.discoveryMode() == BackendDiscoveryMode.EXPLICIT && !explicitSeen) {
                    continue;
                }
                if (resolved.discoveryMode() == BackendDiscoveryMode.AUTO && !explicitSeen && !cpuSeen) {
                    continue;
                }
                jobs.add(acceleratorJob(
                        resolved,
                        target.toPartitionTarget(),
                        sourcePolicy(resolved.discoveryMode()),
                        explicitSeen ? "explicit-backend-intent" : "auto-accelerator-discovery",
                        graph
                ));
            }
        }

        if (cpuSeen && resolved.cpuPartitions().policy() != CpuPartitionPolicy.OFF) {
            jobs.add(cpuJob(resolved));
        }
        return List.copyOf(jobs);
    }

    EnumSet<BackendTarget> explicitTargets(List<CompiledNode> nodes) {
        EnumSet<BackendTarget> out = EnumSet.noneOf(BackendTarget.class);
        for (ExplicitBackendIntent intent : explicitIntents(nodes)) {
            out.add(intent.target());
        }
        return out;
    }

    List<ExplicitBackendIntent> explicitIntents(List<CompiledNode> nodes) {
        List<ExplicitBackendIntent> out = new ArrayList<>();
        if (nodes == null) {
            return out;
        }
        for (CompiledNode node : nodes) {
            BackendTarget target = BackendTarget.fromPartitionTarget(PartitionTarget.fromBackend(node.backend()));
            if (target != null && target.accelerator()) {
                out.add(new ExplicitBackendIntent(node.id(), target));
            }
        }
        return List.copyOf(out);
    }

    private BackendPlanningJob acceleratorJob(
            BackendPlanningConfig config,
            PartitionTarget target,
            PartitionSourcePolicy sourcePolicy,
            String reason,
            List<CompiledNode> nodes
    ) {
        return new BackendPlanningJob(
                target,
                config.ownershipPlanner().toPartitionPlannerStrategy(),
                acceleratorPlannerPolicy(config.search(), nodes),
                sourcePolicy,
                CpuPartitionConfig.defaults(),
                reason
        );
    }

    private BackendPlanningJob cpuJob(BackendPlanningConfig config) {
        return new BackendPlanningJob(
                PartitionTarget.CPU,
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION,
                cpuPlannerPolicy(config.search(), config.cpuPartitions()),
                PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                config.cpuPartitions(),
                "cpu-natural-partition"
        );
    }

    private static AcceleratorPartitionScoreModel.PlannerPolicy acceleratorPlannerPolicy(
            PartitionSearchConfig search,
            List<CompiledNode> nodes
    ) {
        var weights = search.scoreWeights();
        int maxSearchNodes = search.maxSearchNodes();
        if (containsBackwardNodes(nodes)) {
            maxSearchNodes = Math.max(maxSearchNodes, nodes == null ? 0 : nodes.size());
        }
        return new AcceleratorPartitionScoreModel.PlannerPolicy(
                maxSearchNodes,
                search.maxVisitedCandidates(),
                weights.nodeWeight(),
                weights.internalEdgeWeight(),
                weights.mergeNodeBonus(),
                weights.tailDepthWeight(),
                weights.externalInputPenalty(),
                weights.workWeight()
        );
    }

    private static AcceleratorPartitionScoreModel.PlannerPolicy cpuPlannerPolicy(
            PartitionSearchConfig search,
            CpuPartitionConfig cpuPartitionConfig
    ) {
        var weights = search.scoreWeights();
        return new AcceleratorPartitionScoreModel.PlannerPolicy(
                cpuPartitionConfig.maxPartitionNodes(),
                search.maxVisitedCandidates(),
                weights.nodeWeight(),
                weights.internalEdgeWeight(),
                weights.mergeNodeBonus(),
                weights.tailDepthWeight(),
                weights.externalInputPenalty(),
                weights.workWeight()
        );
    }

    private static PartitionSourcePolicy sourcePolicy(BackendDiscoveryMode mode) {
        return switch (mode) {
            case AUTO -> PartitionSourcePolicy.CPU_OR_TARGET_BACKEND;
            case EXPLICIT -> PartitionSourcePolicy.TARGET_BACKEND_ONLY;
            case CPU_ONLY -> PartitionSourcePolicy.TARGET_BACKEND_ONLY;
        };
    }

    private static boolean containsBackwardNodes(List<CompiledNode> nodes) {
        return nodes != null && nodes.stream().anyMatch(CompiledNode::backwardNode);
    }
}
