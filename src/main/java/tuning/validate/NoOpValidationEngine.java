package tuning.validate;

import tuning.candidate.Candidate;
import tuning.workload.WorkloadInstance;
import tuning.workload.WorkloadSpec;

public final class NoOpValidationEngine implements ValidationEngine {
    @Override
    public ValidationResult validate(Candidate candidate, WorkloadSpec workloadSpec, WorkloadInstance workload, ValidationPolicy policy) {
        return ValidationResult.skipped();
    }
}
