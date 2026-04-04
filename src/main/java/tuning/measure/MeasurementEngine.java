package tuning.measure;

import tuning.candidate.Candidate;
import tuning.workload.WorkloadInstance;

public interface MeasurementEngine {
    MeasurementResult measure(Candidate candidate, WorkloadInstance workload, MeasurementPolicy policy);
}
