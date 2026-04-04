package tuning.validate;

import tuning.candidate.Candidate;
import tuning.workload.WorkloadSpec;
import tuning.workload.WorkloadInstance;

public interface ValidationEngine {
    ValidationResult validate(Candidate candidate, WorkloadSpec workloadSpec, WorkloadInstance workload, ValidationPolicy policy);
}
