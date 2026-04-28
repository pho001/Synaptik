package tuning.measure;

import tuning.candidate.Candidate;
import tuning.workload.WorkloadInstance;

/**
 * Measures one candidate against an already-instantiated workload.
 *
 * <p>Implementations typically compile and prepare the workload under the
 * candidate's execution profile, then execute it according to
 * {@link MeasurementPolicy}. Engines may have side effects inherent to
 * execution, such as warming caches or initializing backends, but should not
 * persist results.</p>
 */
public interface MeasurementEngine {
    /**
     * Measures a candidate workload.
     *
     * @param candidate candidate profile to execute; must not be {@code null}
     * @param workload workload instance created for that candidate; must not be {@code null}
     * @param policy measurement controls; must not be {@code null}
     * @return measurement result with traces and steady-state statistics
     */
    MeasurementResult measure(Candidate candidate, WorkloadInstance workload, MeasurementPolicy policy);
}
