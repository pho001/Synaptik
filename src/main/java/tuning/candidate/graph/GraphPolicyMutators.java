package tuning.candidate.graph;

import config.compile.BackendPlanningConfig;
import config.compile.BackendPlanningCostConfig;
import config.compile.CompileConfig;
import config.compile.MemoryPlanningConfig;
import config.compile.PlanningCostProfile;
import config.compile.RegionOptimizationConfig;
import config.compile.RegionOwnershipPlannerStrategy;
import config.optimizer.CseConfig;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
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
                        "backendDiscovery=cpu-only+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuRegions(CpuRegionConfig.defaults()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuRegionConfig.defaults(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=cpu-only+cpuRegion=elementwise-islands+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuRegions(CpuRegionConfig.elementwiseIslands()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuRegionConfig.elementwiseIslands(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=cpu-only+cpuRegion=natural+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_FUSION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuRegions(CpuRegionConfig.defaults()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.aggressive()))),
                        graphKnobs(BackendPlanningConfig.cpuOnly(), CpuRegionConfig.defaults(), CpuFusionConfig.aggressive())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=auto+ownershipPlanner=anchor+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.BACKEND_DISCOVERY_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                        .withOwnershipPlanner(RegionOwnershipPlannerStrategy.ANCHOR)
                                        .withCpuRegions(CpuRegionConfig.defaults()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(BackendPlanningConfig.autoAccelerator(), CpuRegionConfig.defaults(), CpuFusionConfig.defaults())
                ),
                new GraphPolicyVariant(
                        "backendDiscovery=auto+ownershipPlanner=scored+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.OWNERSHIP_PLANNER_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                        .withOwnershipPlanner(RegionOwnershipPlannerStrategy.SCORED)
                                        .withCpuRegions(CpuRegionConfig.defaults()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()))),
                        graphKnobs(
                                BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(RegionOwnershipPlannerStrategy.SCORED),
                                CpuRegionConfig.defaults(),
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
                                .withOwnershipPlanner(RegionOwnershipPlannerStrategy.SCORED)
                                .withCost(new BackendPlanningCostConfig(PlanningCostProfile.measuredTransfer())))),
                        transferKnobs(RegionOwnershipPlannerStrategy.SCORED, TransferCostPreset.MEASURED)
                ),
                new GraphPolicyVariant(
                        "planningCost=metal-transfer-aggressive+ownershipPlanner=scored",
                        GraphAutotuneParameter.PLANNING_COST_PROFILE,
                        GraphExecutionPolicy.of(compile.withBackendPlanning(BackendPlanningConfig.autoAccelerator()
                                .withOwnershipPlanner(RegionOwnershipPlannerStrategy.SCORED)
                                .withCost(new BackendPlanningCostConfig(PlanningCostProfile.aggressiveTransfer())))),
                        transferKnobs(RegionOwnershipPlannerStrategy.SCORED, TransferCostPreset.AGGRESSIVE)
                ),
                new GraphPolicyVariant(
                        "research:cpuRegion=off+cpuFusion=off",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(compile.backendPlanning().withCpuRegions(CpuRegionConfig.off()))
                                .withRegionOptimization(RegionOptimizationConfig.disabled())),
                        cpuKnobs(CpuRegionConfig.off(), CpuFusionConfig.off())
                ),
                new GraphPolicyVariant(
                        "research:cpuRegion=aggressive+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(compile
                                .withBackendPlanning(compile.backendPlanning().withCpuRegions(CpuRegionConfig.aggressive()))
                                .withRegionOptimization(compile.regionOptimization().withCpuFusion(CpuFusionConfig.aggressive()))),
                        cpuKnobs(CpuRegionConfig.aggressive(), CpuFusionConfig.aggressive())
                )
        );
    }

    private static Map<String, String> graphKnobs(
            BackendPlanningConfig backendPlanning,
            CpuRegionConfig cpuRegion,
            CpuFusionConfig cpuFusion
    ) {
        return Map.of(
                "compile.backendPlanning.discoveryMode", backendPlanning.discoveryMode().name(),
                "compile.backendPlanning.ownershipPlanner", backendPlanning.ownershipPlanner().name(),
                "compile.backendPlanning.cpuRegion.policy", cpuRegion.policy().name(),
                "compile.regionOptimization.cpuFusion.mode", cpuFusion.mode().name()
        );
    }

    private static Map<String, String> transferKnobs(
            RegionOwnershipPlannerStrategy ownershipPlanner,
            TransferCostPreset transferModel
    ) {
        return Map.of(
                "compile.backendPlanning.ownershipPlanner", ownershipPlanner.name(),
                "compile.backendPlanning.cost.transferCostPreset", transferModel.name()
        );
    }

    private static Map<String, String> cpuKnobs(CpuRegionConfig cpuRegion, CpuFusionConfig cpuFusion) {
        return Map.of(
                "compile.backendPlanning.cpuRegion.policy", cpuRegion.policy().name(),
                "compile.regionOptimization.cpuFusion.mode", cpuFusion.mode().name()
        );
    }
}
