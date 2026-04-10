package tuning.session;

import tuning.workload.WorkloadSpec;

import java.util.List;

public interface PlatformRuntimeCandidateSpace {
    List<RuntimeProfileCandidate> generate(WorkloadSpec workload);
}
