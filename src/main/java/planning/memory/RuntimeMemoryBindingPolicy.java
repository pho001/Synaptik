package planning.memory;

import java.util.Objects;

/**
 * Per-tensor policy controlling whether partition memory binding may be used at runtime.
 *
 * @param partitionBindingAllowed whether a tensor may bind to partition-managed storage
 * @param reason diagnostic reason for the decision
 */
public record RuntimeMemoryBindingPolicy(
        boolean partitionBindingAllowed,
        String reason
) {
    public static final RuntimeMemoryBindingPolicy PARTITION_BINDING_ALLOWED =
            new RuntimeMemoryBindingPolicy(true, "partition-binding-allowed");

    public RuntimeMemoryBindingPolicy {
        reason = reason == null ? "" : reason;
    }

    /**
     * Creates a policy that skips partition binding.
     *
     * @param reason diagnostic reason
     * @return skip policy
     */
    public static RuntimeMemoryBindingPolicy skip(String reason) {
        return new RuntimeMemoryBindingPolicy(false, Objects.requireNonNull(reason, "reason cannot be null"));
    }
}
