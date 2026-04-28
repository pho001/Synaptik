package tuning.validate;

import tuning.candidate.Candidate;
import tuning.workload.WorkloadSpec;
import tuning.workload.WorkloadInstance;

/**
 * Validates one candidate workload before measurement.
 *
 * <p>Engines compare the workload's validation target against the reference
 * described by the workload instance and policy. They should return
 * {@link ValidationResult#skipped()} when validation is disabled instead of
 * throwing.</p>
 */
public interface ValidationEngine {
    /**
     * Validates a candidate workload.
     *
     * @param candidate candidate being validated
     * @param workloadSpec workload specification that produced the instance
     * @param workload instantiated workload
     * @param policy validation controls
     * @return validation status and metrics
     */
    ValidationResult validate(Candidate candidate, WorkloadSpec workloadSpec, WorkloadInstance workload, ValidationPolicy policy);
}
