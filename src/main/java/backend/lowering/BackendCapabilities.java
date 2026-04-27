package backend.lowering;

import backend.ComputeBackend;

import java.util.Set;

public record BackendCapabilities(
        Set<ComputeBackend> availableBackends
) {
    public BackendCapabilities {
        availableBackends = Set.copyOf(availableBackends == null ? Set.of() : availableBackends);
    }

    public static BackendCapabilities none() {
        return new BackendCapabilities(Set.of());
    }

    public boolean supports(ComputeBackend backend) {
        return backend != null && availableBackends.contains(backend);
    }
}
