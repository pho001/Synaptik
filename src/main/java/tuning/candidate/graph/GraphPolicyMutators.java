package tuning.candidate.graph;

import config.compile.BackendPlanningConfig;
import config.compile.BackendPlanningCostConfig;
import config.compile.CompileConfig;
import config.compile.MemoryPlanningConfig;
import config.compile.PlanningCostProfile;
import config.compile.PartitionExecutionConfig;
import config.compile.PartitionOwnershipPlannerStrategy;
import config.optimizer.CseConfig;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuPartitionConfig;
import config.optimizer.MemoryConfig;
import config.compile.TransferCostPreset;
import config.optimizer.PiecewiseLoweringConfig;
import config.profile.GraphExecutionPolicy;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class GraphPolicyMutators {
    private GraphPolicyMutators() {
    }

    public record GraphPolicyVariant(
            String name,
            GraphAutotuneParameter parameter,
            GraphExecutionPolicy policy,
            Map<String, String> knobAssignments
    ) {
        public GraphPolicyVariant {
            name = name == null || name.isBlank() ? "graphPolicy=current" : name;
            Objects.requireNonNull(parameter, "parameter cannot be null");
            Objects.requireNonNull(policy, "policy cannot be null");
            knobAssignments = knobAssignments == null ? Map.of() : Map.copyOf(knobAssignments);
        }

        public GraphPolicyVariant(
                String name,
                GraphAutotuneParameter parameter,
                GraphExecutionPolicy policy
        ) {
            this(name, parameter, policy, Map.of());
        }
    }

    public static List<GraphPolicyVariant> standard(GraphExecutionPolicy base) {
        Objects.requireNonNull(base, "base cannot be null");
        CompileConfig compile = base.compile();
        return List.of(
                new GraphPolicyVariant(
                        "graphPolicy=current",
                        GraphAutotuneParameter.CURRENT_GRAPH_POLICY,
                        base
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=cpu-only+cpuPartition=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_PARTITION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuPartitions(CpuPartitionConfig.defaults()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuPartitionConfig.defaults(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=cpu-only+cpuPartition=elementwise-islands+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_PARTITION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuPartitions(CpuPartitionConfig.elementwiseIslands()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuPartitionConfig.elementwiseIslands(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=cpu-only+cpuPartition=natural+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_FUSION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuPartitions(CpuPartitionConfig.defaults()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.aggressive()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuPartitionConfig.defaults(), CpuFusionConfig.aggressive())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=auto+ownershipPlanner=anchor+cpuPartition=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.BACKEND_DISCOVERY_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                        .withOwnershipPlanner(PartitionOwnershipPlannerStrategy.ANCHOR)
                                        .withCpuPartitions(CpuPartitionConfig.defaults()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.autoAccelerator(), CpuPartitionConfig.defaults(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=auto+ownershipPlanner=scored+cpuPartition=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.OWNERSHIP_PLANNER_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                        .withOwnershipPlanner(PartitionOwnershipPlannerStrategy.SCORED)
                                        .withCpuPartitions(CpuPartitionConfig.defaults()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(
                                BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(PartitionOwnershipPlannerStrategy.SCORED),
                                CpuPartitionConfig.defaults(),
                                CpuFusionConfig.defaults())
                )
        );
    }

    public static List<GraphPolicyVariant> research(GraphExecutionPolicy base) {
        Objects.requireNonNull(base, "base cannot be null");
        CompileConfig compile = base.compile();
        return List.of(
                new GraphPolicyVariant(
                        "cse=strict",
                        GraphAutotuneParameter.RESEARCH_CSE_POLICY,
                        GraphExecutionPolicy.of(compile.withGraphOptimization(
                                compile.graphOptimization().withCse(CseConfig.strictDefaults())))
                ),
                new GraphPolicyVariant(
                        "cse=aggressive",
                        GraphAutotuneParameter.RESEARCH_CSE_POLICY,
                        GraphExecutionPolicy.of(compile.withGraphOptimization(
                                compile.graphOptimization().withCse(CseConfig.aggressiveDefaults())))
                ),
                new GraphPolicyVariant(
                        "piecewise=current",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        base
                ),
                new GraphPolicyVariant(
                        "piecewise=off",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(compile.withGraphOptimization(compile.graphOptimization().withRewrite(
                                compile.graphOptimization().rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.defaults())
                        )))
                ),
                new GraphPolicyVariant(
                        "piecewise=canonical",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(compile.withGraphOptimization(compile.graphOptimization().withRewrite(
                                compile.graphOptimization().rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.aggressiveDefaults())
                        )))
                ),
                new GraphPolicyVariant(
                        "memory=current",
                        GraphAutotuneParameter.RESEARCH_MEMORY_LIFETIME,
                        base
                ),
                new GraphPolicyVariant(
                        "memory=phase-isolated",
                        GraphAutotuneParameter.MEMORY_PLANNING_POLICY,
                        GraphExecutionPolicy.of(compile.withMemoryPlanning(
                                new MemoryPlanningConfig(true, new MemoryConfig(true, false, false, 1))))
                ),
                new GraphPolicyVariant(
                        "memory=cross-phase-lifetime",
                        GraphAutotuneParameter.MEMORY_PLANNING_POLICY,
                        GraphExecutionPolicy.of(compile.withMemoryPlanning(
                                new MemoryPlanningConfig(true, new MemoryConfig(false, true, false, 1))))
                ),
                new GraphPolicyVariant(
                        "planningCost=metal-transfer-measured+ownershipPlanner=scored",
                        GraphAutotuneParameter.PLANNING_COST_PROFILE,
                        GraphExecutionPolicy.of(compile.withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                .withOwnershipPlanner(PartitionOwnershipPlannerStrategy.SCORED)
                                .withCost(new BackendPlanningCostConfig(PlanningCostProfile.measuredTransfer())))),
                        transferKnobs(PartitionOwnershipPlannerStrategy.SCORED, TransferCostPreset.MEASURED)
                ),
                new GraphPolicyVariant(
                        "planningCost=metal-transfer-aggressive+ownershipPlanner=scored",
                        GraphAutotuneParameter.PLANNING_COST_PROFILE,
                        GraphExecutionPolicy.of(compile.withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                .withOwnershipPlanner(PartitionOwnershipPlannerStrategy.SCORED)
                                .withCost(new BackendPlanningCostConfig(PlanningCostProfile.aggressiveTransfer())))),
                        transferKnobs(PartitionOwnershipPlannerStrategy.SCORED, TransferCostPreset.AGGRESSIVE)
                ),
                new GraphPolicyVariant(
                        "research:cpuPartition=off+cpuFusion=off",
                        GraphAutotuneParameter.CPU_PARTITION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(compile.backendPlanning().withCpuPartitions(CpuPartitionConfig.off()))
                                .withPartitionExecution(PartitionExecutionConfig.disabled())),
                        cpuKnobs(CpuPartitionConfig.off(), CpuFusionConfig.off())
                ),
                new GraphPolicyVariant(
                        "research:cpuPartition=aggressive+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_PARTITION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(compile.backendPlanning().withCpuPartitions(CpuPartitionConfig.aggressive()))
                                .withPartitionExecution(compile.partitionExecution().withCpuFusion(CpuFusionConfig.aggressive()))),
                        cpuKnobs(CpuPartitionConfig.aggressive(), CpuFusionConfig.aggressive())
                )
        );
    }

    private static Map<String, String> graphKnobs(
            BackendPlanningConfig backendPlanning,
            CpuPartitionConfig cpuPartition,
            CpuFusionConfig cpuFusion
    ) {
        return Map.of(
                "compile.backendPlanning.discoveryMode", backendPlanning.discoveryMode().name(),
                "compile.backendPlanning.ownershipPlanner", backendPlanning.ownershipPlanner().name(),
                "compile.backendPlanning.cpuPartition.policy", cpuPartition.policy().name(),
                "compile.partitionExecution.cpuFusion.mode", cpuFusion.mode().name()
        );
    }

    private static Map<String, String> transferKnobs(
            PartitionOwnershipPlannerStrategy ownershipPlanner,
            TransferCostPreset transferModel
    ) {
        return Map.of(
                "compile.backendPlanning.ownershipPlanner", ownershipPlanner.name(),
                "compile.backendPlanning.cost.transferCostPreset", transferModel.name()
        );
    }

    private static Map<String, String> cpuKnobs(CpuPartitionConfig cpuPartition, CpuFusionConfig cpuFusion) {
        return Map.of(
                "compile.backendPlanning.cpuPartition.policy", cpuPartition.policy().name(),
                "compile.partitionExecution.cpuFusion.mode", cpuFusion.mode().name()
        );
    }
}
