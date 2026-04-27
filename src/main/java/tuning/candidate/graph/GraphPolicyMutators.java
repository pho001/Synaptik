package tuning.candidate.graph;

import config.optimizer.CseConfig;
import config.optimizer.MemoryConfig;
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
        return List.of(new GraphPolicyVariant(
                "graphPolicy=current",
                GraphAutotuneParameter.CURRENT_GRAPH_POLICY,
                base
        ));
    }

    public static List<GraphPolicyVariant> research(GraphExecutionPolicy base) {
        Objects.requireNonNull(base, "base cannot be null");
        var optimizer = base.optimizer();
        return List.of(
                new GraphPolicyVariant(
                        "cse=strict",
                        GraphAutotuneParameter.CSE_STRICT_SAFETY,
                        GraphExecutionPolicy.of(optimizer.withCse(CseConfig.strictDefaults()))
                ),
                new GraphPolicyVariant(
                        "cse=aggressive",
                        GraphAutotuneParameter.CSE_STRICT_SAFETY,
                        GraphExecutionPolicy.of(optimizer.withCse(CseConfig.aggressiveDefaults()))
                ),
                new GraphPolicyVariant(
                        "piecewise=current",
                        GraphAutotuneParameter.PIECEWISE_LOWERING,
                        base
                ),
                new GraphPolicyVariant(
                        "piecewise=off",
                        GraphAutotuneParameter.PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(optimizer.withRewrite(
                                optimizer.rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.defaults())
                        ))
                ),
                new GraphPolicyVariant(
                        "piecewise=canonical",
                        GraphAutotuneParameter.PIECEWISE_LOWERING,
                        GraphExecutionPolicy.of(optimizer.withRewrite(
                                optimizer.rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.aggressiveDefaults())
                        ))
                ),
                new GraphPolicyVariant(
                        "memory=current",
                        GraphAutotuneParameter.MEMORY_LIFETIME,
                        base
                ),
                new GraphPolicyVariant(
                        "memory=phase-isolated",
                        GraphAutotuneParameter.MEMORY_LIFETIME,
                        GraphExecutionPolicy.of(optimizer.withMemory(new MemoryConfig(true, false, false, 1)))
                ),
                new GraphPolicyVariant(
                        "memory=cross-phase-lifetime",
                        GraphAutotuneParameter.MEMORY_LIFETIME,
                        GraphExecutionPolicy.of(optimizer.withMemory(new MemoryConfig(false, true, false, 1)))
                )
        );
    }
}
