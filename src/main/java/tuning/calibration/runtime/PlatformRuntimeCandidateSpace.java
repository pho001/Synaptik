package tuning.calibration.runtime;

import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Generates runtime-profile candidates for one calibration step.
 *
 * <p>Implementations may inspect the workload shape to produce targeted runtime
 * candidates. The returned list should be deterministic for the same seed
 * profile and workload so calibration results are reproducible.</p>
 */
public interface PlatformRuntimeCandidateSpace {
    /**
     * Generates runtime candidates for the supplied workload.
     *
     * @param workload calibration workload
     * @return candidates to measure; never mutate the returned list after exposure
     */
    List<RuntimeProfileCandidate> generate(WorkloadSpec workload);
}
