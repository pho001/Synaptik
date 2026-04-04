package tuning.candidate;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class StaticProfileCandidateSpace implements CandidateSpace {
    private final List<ExecutionProfile> profiles;

    public StaticProfileCandidateSpace(List<ExecutionProfile> profiles) {
        this.profiles = profiles == null ? List.of() : List.copyOf(profiles);
    }

    @Override
    public List<Candidate> generate(WorkloadSpec workload) {
        Objects.requireNonNull(workload, "workload cannot be null");
        List<Candidate> out = new ArrayList<>(profiles.size());
        for (ExecutionProfile profile : profiles) {
            out.add(new Candidate(profile.candidateName(), profile));
        }
        return List.copyOf(out);
    }
}
