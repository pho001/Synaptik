package graph.execution.state;

import backend.memory.ExecutionResource;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Run-scoped owned native/backend resources.
 */
final class RuntimeResourceRegistry {
    private final List<ExecutionResource> executionResources = new ArrayList<>();

    void registerResource(ExecutionResource resource) {
        executionResources.add(Objects.requireNonNull(resource, "resource cannot be null"));
    }

    void closeResources() {
        RuntimeException closeFailure = null;
        for (int i = executionResources.size() - 1; i >= 0; i--) {
            try {
                executionResources.get(i).close();
            } catch (RuntimeException ex) {
                if (closeFailure == null) {
                    closeFailure = new RuntimeException("One or more execution resources failed to close.");
                }
                closeFailure.addSuppressed(ex);
            }
        }
        executionResources.clear();
        if (closeFailure != null) {
            throw closeFailure;
        }
    }
}
