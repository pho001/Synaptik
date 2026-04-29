package tuning.candidate.graph;

import config.optimizer.CseConfig;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.MetalTransferModel;
import config.optimizer.OffloadConfig;
import config.optimizer.PiecewiseLoweringConfig;
import config.profile.GraphExecutionPolicy;

import java.util.List;
import java.util.Objects;

public final class GraphPolicyMutators {
    private GraphPolicyMutators() {
    }

    public record GraphPolicyVariant(
            String name,
            GraphAutotuneParameter parameter,
            GraphExecutionPolicy policy
    ) {
        public GraphPolicyVariant {
            name = name == null || name.isBlank() ? "graphPolicy=current" : name;
            Objects.requireNonNull(parameter, "parameter cannot be null");
            Objects.requireNonNull(policy, "policy cannot be null");
        }
    }

    public static List<GraphPolicyVariant> standard(GraphExecutionPolicy base) {
        Objects.requireNonNull(base, "base cannot be null");
        var optimizer = base.optimizer();
        return List.of(
                new GraphPolicyVariant(
                        "graphPolicy=current",
                        GraphAutotuneParameter.CURRENT_GRAPH_POLICY,
                        base
                ),
                new GraphPolicyVariant(
                        "offload=cpu-only+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.defaults())
                                .withCpuRegion(CpuRegionConfig.defaults())
                                .withCpuFusion(CpuFusionConfig.defaults()))
                ),
                new GraphPolicyVariant(
                        "offload=cpu-only+cpuRegion=elementwise-islands+cpuFusion=balanced",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.defaults())
                                .withCpuRegion(CpuRegionConfig.elementwiseIslands())
                                .withCpuFusion(CpuFusionConfig.defaults()))
                ),
                new GraphPolicyVariant(
                        "offload=cpu-only+cpuRegion=natural+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_FUSION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.defaults())
                                .withCpuRegion(CpuRegionConfig.defaults())
                                .withCpuFusion(CpuFusionConfig.aggressive()))
                ),
                new GraphPolicyVariant(
                        "offload=accelerator-profitable+accelRegion=greedy+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.OFFLOAD_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.acceleratorGreedy())
                                .withCpuRegion(CpuRegionConfig.defaults())
                                .withCpuFusion(CpuFusionConfig.defaults()))
                ),
                new GraphPolicyVariant(
                        "offload=accelerator-profitable+accelRegion=scored+cpuRegion=natural+cpuFusion=balanced",
                        GraphAutotuneParameter.ACCELERATOR_REGION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.acceleratorScored())
                                .withCpuRegion(CpuRegionConfig.defaults())
                                .withCpuFusion(CpuFusionConfig.defaults()))
                )
        );
    }

    public static List<GraphPolicyVariant> research(GraphExecutionPolicy base) {
        Objects.requireNonNull(base, "base cannot be null");
        var optimizer = base.optimizer();
        return List.of(
                new GraphPolicyVariant(
                        "cse=strict",
                        GraphAutotuneParameter.RESEARCH_CSE_POLICY,
                        GraphExecutionPolicy.of(optimizer.withCse(CseConfig.strictDefaults()))
                ),
                new GraphPolicyVariant(
                        "cse=aggressive",
                        GraphAutotuneParameter.RESEARCH_CSE_POLICY,
                        GraphExecutionPolicy.of(optimizer.withCse(CseConfig.aggressiveDefaults()))
                ),
                new GraphPolicyVariant(
                        "piecewise=current",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        base
                ),
                new GraphPolicyVariant(
                        "piecewise=off",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(optimizer.withRewrite(
                                optimizer.rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.defaults())
                        ))
                ),
                new GraphPolicyVariant(
                        "piecewise=canonical",
                        GraphAutotuneParameter.RESEARCH_PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(optimizer.withRewrite(
                                optimizer.rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.aggressiveDefaults())
                        ))
                ),
                new GraphPolicyVariant(
                        "memory=current",
                        GraphAutotuneParameter.RESEARCH_MEMORY_LIFETIME,
                        base
                ),
                new GraphPolicyVariant(
                        "memory=phase-isolated",
                        GraphAutotuneParameter.RESEARCH_MEMORY_LIFETIME,
                        GraphExecutionPolicy.of(optimizer.withMemory(new MemoryConfig(true, false, false, 1)))
                ),
                new GraphPolicyVariant(
                        "memory=cross-phase-lifetime",
                        GraphAutotuneParameter.RESEARCH_MEMORY_LIFETIME,
                        GraphExecutionPolicy.of(optimizer.withMemory(new MemoryConfig(false, true, false, 1)))
                ),
                new GraphPolicyVariant(
                        "metalTransfer=measured+accelRegion=scored",
                        GraphAutotuneParameter.RESEARCH_METAL_TRANSFER_MODEL,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.acceleratorScored())
                                .withPartition(optimizer.partition().withMetalTransferModel(MetalTransferModel.MEASURED)))
                ),
                new GraphPolicyVariant(
                        "metalTransfer=aggressive+accelRegion=scored",
                        GraphAutotuneParameter.RESEARCH_METAL_TRANSFER_MODEL,
                        GraphExecutionPolicy.of(optimizer
                                .withOffload(OffloadConfig.acceleratorScored())
                                .withPartition(optimizer.partition().withMetalTransferModel(MetalTransferModel.AGGRESSIVE)))
                ),
                new GraphPolicyVariant(
                        "research:cpuRegion=off+cpuFusion=off",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withCpuRegion(CpuRegionConfig.off())
                                .withCpuFusion(CpuFusionConfig.off()))
                ),
                new GraphPolicyVariant(
                        "research:cpuRegion=aggressive+cpuFusion=aggressive",
                        GraphAutotuneParameter.CPU_REGION_POLICY,
                        GraphExecutionPolicy.of(optimizer
                                .withCpuRegion(CpuRegionConfig.aggressive())
                                .withCpuFusion(CpuFusionConfig.aggressive()))
                )
        );
    }
}
