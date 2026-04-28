package graph.optimizer.memory;

import java.util.Objects;

/**
 * Per-tensor policy controlling whether region memory binding may be used at runtime.
 *
 * @param regionBindingAllowed whether a tensor may bind to region-managed storage
 * @param reason diagnostic reason for the decision
 */
public record RuntimeMemoryBindingPolicy(
        boolean regionBindingAllowed,
        String reason
) {
    public static final RuntimeMemoryBindingPolicy REGION_BINDING_ALLOWED =
            new RuntimeMemoryBindingPolicy(true, "region-binding-allowed");

    public RuntimeMemoryBindingPolicy {
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates a policy that skips region binding.
     *
     * @param reason diagnostic reason
     * @return skip policy
     */
    public static RuntimeMemoryBindingPolicy skip(String reason) {
        return new RuntimeMemoryBindingPolicy(false, Objects.requireNonNull(reason, "reason cannot be null"));
    }
}
