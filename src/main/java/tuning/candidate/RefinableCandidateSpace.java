package tuning.candidate;

import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Candidate space that can provide local neighbors for refinement search.
 */
public interface RefinableCandidateSpace extends CandidateSpace {
    /**
     * Returns candidates adjacent to an already evaluated candidate.
     *
     * @param candidate evaluated candidate
     * @param workload workload being tuned
     * @return neighboring candidates for refinement
     */
    List<Candidate> neighbors(Candidate candidate, WorkloadSpec workload);
}
