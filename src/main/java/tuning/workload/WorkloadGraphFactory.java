package tuning.workload;

/**
 * Creates an instantiated workload graph for one execution profile.
 */
@FunctionalInterface
public interface WorkloadGraphFactory {
    /**
     * @param environment workload environment
     * @return instantiated workload graph
     */
    WorkloadGraph create(WorkloadEnvironment environment);
}
