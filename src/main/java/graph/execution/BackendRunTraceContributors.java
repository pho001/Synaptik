package graph.execution;

import java.util.LinkedHashMap;
import java.util.List;

final class BackendRunTraceContributors {
    private static final List<BackendRunTraceContributor> DEFAULTS = List.of(
            new CpuRunTraceContributor(),
            new AcceleratorRunTraceContributor(),
            new MetalRunTraceContributor(),
            new CudaRunTraceContributor(),
            new StorageRunTraceContributor()
    );

    private BackendRunTraceContributors() {
    }

    static void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        for (BackendRunTraceContributor contributor : DEFAULTS) {
            contributor.contribute(context, attrs);
        }
    }
}
