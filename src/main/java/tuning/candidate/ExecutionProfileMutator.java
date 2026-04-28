package tuning.candidate;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Expands an execution profile into profile variants for a candidate grid.
 *
 * <p>Mutators should leave the supplied profile unchanged and return new variant
 * profiles. Returning {@code null} or an empty list means the mutator has no
 * applicable variants for the workload.</p>
 */
public interface ExecutionProfileMutator {
    /**
     * Produces variants of an execution profile.
     *
     * @param baseProfile profile to mutate
     * @param workload workload being tuned
     * @return profile variants with suffixes used in candidate names
     */
    List<ExecutionProfileVariant> variants(ExecutionProfile baseProfile, WorkloadSpec workload);
}
