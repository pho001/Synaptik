package tuning.calibration.runtime;

import config.profile.PlatformRuntimeProfile;
import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Expands a platform runtime profile into variants for a calibration workload.
 *
 * <p>Mutators should treat the supplied base profile as immutable and return new
 * {@link RuntimeProfileCandidate} instances. Null or empty results conventionally
 * mean the mutator has no applicable variants for the workload.</p>
 */
public interface PlatformRuntimeProfileMutator {
    /**
     * Produces variants of the base runtime profile.
     *
     * @param baseProfile current runtime profile selected by earlier steps
     * @param workload workload being calibrated
     * @return runtime-profile candidates for this mutator
     */
    List<RuntimeProfileCandidate> variants(PlatformRuntimeProfile baseProfile, WorkloadSpec workload);
}
