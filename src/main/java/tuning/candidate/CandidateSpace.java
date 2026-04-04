package tuning.candidate;

import tuning.workload.WorkloadSpec;

import java.util.List;

public interface CandidateSpace {
    List<Candidate> generate(WorkloadSpec workload);
}
