package tuning.workload;

/**
 * Reusable workload specification for tuning flows.
 *
 * <p>A spec is a factory, not the executable graph itself. Sessions call
 * {@link #instantiate(WorkloadEnvironment)} separately for each candidate so the
 * workload can bind to the candidate's execution profile.</p>
 */
public interface WorkloadSpec {
    /**
     * @return stable workload name used in reports and persistence fingerprints
     */
    String name();

    /**
     * @return workload family used for preset and search defaults
     */
    WorkloadKind kind();

    /**
     * Creates a workload instance for one candidate environment.
     *
     * @param environment candidate execution environment
     * @return executable workload instance
     */
    WorkloadInstance instantiate(WorkloadEnvironment environment);
}
