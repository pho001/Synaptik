package tuning.session;

import config.profile.PlatformRuntimeProfile;
import tuning.workload.WorkloadSpec;

import java.util.List;

public interface PlatformRuntimeProfileMutator {
    List<RuntimeProfileCandidate> variants(PlatformRuntimeProfile baseProfile, WorkloadSpec workload);
}
