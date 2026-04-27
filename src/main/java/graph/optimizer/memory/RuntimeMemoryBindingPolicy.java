package graph.optimizer.memory;

import java.util.Objects;

public record RuntimeMemoryBindingPolicy(
        boolean regionBindingAllowed,
        String reason
) {
    public static final RuntimeMemoryBindingPolicy REGION_BINDING_ALLOWED =
            new RuntimeMemoryBindingPolicy(true, "region-binding-allowed");

    public RuntimeMemoryBindingPolicy {
        reason = reason == null ? "" : reason;
    }

    public static RuntimeMemoryBindingPolicy skip(String reason) {
        return new RuntimeMemoryBindingPolicy(false, Objects.requireNonNull(reason, "reason cannot be null"));
    }
}
