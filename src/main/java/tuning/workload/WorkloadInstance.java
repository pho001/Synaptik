package tuning.workload;

import graph.compile.intent.BackendIntentPlan;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

/**
 * Candidate-bound workload graph with validation metadata.
 */
public interface WorkloadInstance {
    /**
     * @return root tensor that should be compiled and executed
     */
    Tensor root();

    /**
     * @return target output or gradients to validate
     */
    ValidationTarget validationTarget();

    /**
     * @return reference data or baseline profile used by validation
     */
    ValidationReference reference();

    /**
     * @return metadata used by reports and persistence fingerprints
     */
    WorkloadMetadata metadata();

    /**
     * @return compile-local backend intent plan for this instantiated graph
     */
    default BackendIntentPlan backendIntentPlan() {
        return BackendIntentPlan.empty();
    }
}
