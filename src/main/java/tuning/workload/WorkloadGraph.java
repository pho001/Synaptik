package tuning.workload;

import graph.compile.intent.BackendIntentPlan;
import tensor.Tensor;

import java.util.Objects;

/**
 * Instantiated tensor graph plus compile-local metadata needed to compile it.
 *
 * @param root executable root tensor
 * @param backendIntentPlan compile-local backend intent plan
 */
public record WorkloadGraph(
        Tensor root,
        BackendIntentPlan backendIntentPlan
) {
    public WorkloadGraph {
        Objects.requireNonNull(root, "root cannot be null");
        backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
    }

    /**
     * Creates a workload graph without explicit backend intent.
     *
     * @param root executable root tensor
     * @return workload graph
     */
    public static WorkloadGraph of(Tensor root) {
        return new WorkloadGraph(root, BackendIntentPlan.empty());
    }
}
