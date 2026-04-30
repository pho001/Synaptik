package tuning.candidate.graph;

import config.profile.GraphExecutionPolicy;
import tuning.candidate.RuntimeConfigOverride;

import java.util.Map;
import java.util.Objects;

/**
 * Graph autotune variant that may alter both graph policy and runtime policy.
 *
 * <p>Most graph autotune candidates mutate only optimizer policy and keep the calibrated runtime profile
 * frozen. Accelerator buffer binding is different: it is a graph/workload-specific runtime decision, so
 * the candidate needs a small runtime override while still sharing the calibrated platform thresholds.</p>
 *
 * @param name candidate name
 * @param parameter parameter family being varied
 * @param policy graph execution policy for this candidate
 * @param runtimeOverride function applied to the calibrated runtime config
 * @param graphPolicyMutated whether graph policy differs from the seed
 * @param runtimeMutated whether runtime config differs intentionally from the calibrated seed
 * @param knobAssignments tuning knob keys intentionally changed by this variant
 * @param metadata additional candidate attributes
 */
public record GraphRuntimePolicyVariant(
        String name,
        GraphAutotuneParameter parameter,
        GraphExecutionPolicy policy,
        RuntimeConfigOverride runtimeOverride,
        boolean graphPolicyMutated,
        boolean runtimeMutated,
        Map<String, String> knobAssignments,
        Map<String, String> metadata
) {
    public GraphRuntimePolicyVariant {
        name = name == null || name.isBlank() ? "graphPolicy=current" : name;
        Objects.requireNonNull(parameter, "parameter cannot be null");
        Objects.requireNonNull(policy, "policy cannot be null");
        runtimeOverride = runtimeOverride == null ? RuntimeConfigOverride.identity() : runtimeOverride;
        knobAssignments = knobAssignments == null ? Map.of() : Map.copyOf(knobAssignments);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public static GraphRuntimePolicyVariant fromGraphPolicy(
            GraphPolicyMutators.GraphPolicyVariant variant,
            GraphExecutionPolicy seed
    ) {
        Objects.requireNonNull(variant, "variant cannot be null");
        return new GraphRuntimePolicyVariant(
                variant.name(),
                variant.parameter(),
                variant.policy(),
                RuntimeConfigOverride.identity(),
                !variant.policy().equals(seed),
                false,
                variant.knobAssignments(),
                Map.of()
        );
    }
}
