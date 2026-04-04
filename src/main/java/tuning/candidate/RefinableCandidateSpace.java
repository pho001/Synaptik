package tuning.candidate;

import tuning.workload.WorkloadSpec;

import java.util.List;

public interface RefinableCandidateSpace extends CandidateSpace {
    List<Candidate> neighbors(Candidate candidate, WorkloadSpec workload);
}
