package tuning.candidate;

import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Generates candidate profiles for a workload.
 *
 * <p>Candidate spaces are searched by autotune sessions. Benchmark sessions do
 * not use this interface because benchmarks operate on explicit caller-supplied
 * entries.</p>
 */
public interface CandidateSpace {
    /**
     * Generates candidates for the supplied workload.
     *
     * @param workload workload whose shape/metadata may influence candidates
     * @return deterministic list of candidates to search
     */
    List<Candidate> generate(WorkloadSpec workload);
}
