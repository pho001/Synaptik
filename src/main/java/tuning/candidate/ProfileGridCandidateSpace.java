package tuning.candidate;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class ProfileGridCandidateSpace implements RefinableCandidateSpace {
    private final ExecutionProfile baseProfile;
    private final List<ExecutionProfileMutator> mutators;

    public ProfileGridCandidateSpace(
            ExecutionProfile baseProfile,
            List<ExecutionProfileMutator> mutators
    ) {
        this.baseProfile = Objects.requireNonNull(baseProfile, "baseProfile cannot be null");
        this.mutators = mutators == null ? List.of() : List.copyOf(mutators);
    }

    public ExecutionProfile baseProfile() {
        return baseProfile;
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

    @Override
    public List<Candidate> neighbors(Candidate candidate, WorkloadSpec workload) {
        Objects.requireNonNull(candidate, "candidate cannot be null");
        Objects.requireNonNull(workload, "workload cannot be null");
        List<Candidate> neighbors = new ArrayList<>();
        String self = ExecutableProfileFingerprint.of(candidate);
        java.util.LinkedHashMap<String, Candidate> dedup = new java.util.LinkedHashMap<>();
        for (ExecutionProfileMutator mutator : mutators) {
            List<ExecutionProfileVariant> variants = mutator.variants(candidate.profile(), workload);
            if (variants == null) {
                continue;
            }
            for (ExecutionProfileVariant variant : variants) {
                Candidate next = new Candidate(variant.suffix(), variant.profile());
                String fp = ExecutableProfileFingerprint.of(next);
                if (self.equals(fp)) {
                    continue;
                }
                dedup.putIfAbsent(fp, next);
            }
        }
        neighbors.addAll(dedup.values());
        return List.copyOf(neighbors);
    }
}
