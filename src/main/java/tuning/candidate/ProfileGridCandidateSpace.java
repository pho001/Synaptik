package tuning.candidate;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class ProfileGridCandidateSpace implements CandidateSpace {
    private final ExecutionProfile baseProfile;
    private final List<ExecutionProfileMutator> mutators;

    public ProfileGridCandidateSpace(
            ExecutionProfile baseProfile,
            List<ExecutionProfileMutator> mutators
    ) {
        this.baseProfile = Objects.requireNonNull(baseProfile, "baseProfile cannot be null");
        this.mutators = mutators == null ? List.of() : List.copyOf(mutators);
    }

    @Override
    public List<Candidate> generate(WorkloadSpec workload) {
        Objects.requireNonNull(workload, "workload cannot be null");
        List<ExecutionProfileVariant> current = List.of(new ExecutionProfileVariant(baseProfile.candidateName(), baseProfile));
        for (ExecutionProfileMutator mutator : mutators) {
            List<ExecutionProfileVariant> next = new ArrayList<>();
            for (ExecutionProfileVariant variant : current) {
                List<ExecutionProfileVariant> expanded = mutator.variants(variant.profile(), workload);
                if (expanded == null || expanded.isEmpty()) {
                    next.add(variant);
                    continue;
                }
                for (ExecutionProfileVariant child : expanded) {
                    String suffix = variant.suffix() + "+" + child.suffix();
                    ExecutionProfile profile = child.profile();
                    String candidateName = profile.candidateName() == null || profile.candidateName().isBlank()
                            ? suffix
                            : suffix + ":" + profile.candidateName();
                    next.add(new ExecutionProfileVariant(
                            suffix,
                            new ExecutionProfile(
                                    profile.profileName(),
                                    candidateName,
                                    profile.dataType(),
                                    profile.mode(),
                                    profile.optimizer(),
                                    profile.runtime(),
                                    profile.workload()
                            )
                    ));
                }
            }
            current = List.copyOf(next);
        }

        List<Candidate> out = new ArrayList<>(current.size());
        for (ExecutionProfileVariant variant : current) {
            out.add(new Candidate(variant.suffix(), variant.profile()));
        }
        return List.copyOf(out);
    }
}
