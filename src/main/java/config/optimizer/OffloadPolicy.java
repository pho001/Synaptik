package config.optimizer;

/**
 * Graph-level policy for accelerator/offload region planning.
 */
public enum OffloadPolicy {
    /**
     * Do not create accelerator ownership regions. CPU planning may still create CPU execution regions.
     */
    CPU_ONLY,

    /**
     * Allow accelerator ownership regions when the target backend is available and planner policy accepts them.
     */
    ACCELERATOR_IF_PROFITABLE
}
