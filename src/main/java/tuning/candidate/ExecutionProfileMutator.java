package tuning.candidate;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadSpec;

import java.util.List;

public interface ExecutionProfileMutator {
    List<ExecutionProfileVariant> variants(ExecutionProfile baseProfile, WorkloadSpec workload);
}
