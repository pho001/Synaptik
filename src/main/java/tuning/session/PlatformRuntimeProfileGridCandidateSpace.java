package tuning.session;

import config.profile.PlatformRuntimeProfile;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class PlatformRuntimeProfileGridCandidateSpace implements PlatformRuntimeCandidateSpace {
    private final PlatformRuntimeProfile baseProfile;
    private final List<PlatformRuntimeProfileMutator> mutators;

    public PlatformRuntimeProfileGridCandidateSpace(
            PlatformRuntimeProfile baseProfile,
            List<PlatformRuntimeProfileMutator> mutators
    ) {
        this.baseProfile = Objects.requireNonNull(baseProfile, "baseProfile cannot be null");
        this.mutators = mutators == null ? List.of() : List.copyOf(mutators);
    }

    @Override
    public List<RuntimeProfileCandidate> generate(WorkloadSpec workload) {
        Objects.requireNonNull(workload, "workload cannot be null");
        List<RuntimeProfileCandidate> current = List.of(new RuntimeProfileCandidate("base", baseProfile, Map.of()));
        for (PlatformRuntimeProfileMutator mutator : mutators) {
            List<RuntimeProfileCandidate> next = new ArrayList<>();
            for (RuntimeProfileCandidate variant : current) {
                List<RuntimeProfileCandidate> expanded = mutator.variants(variant.runtimeProfile(), workload);
                if (expanded == null || expanded.isEmpty()) {
                    next.add(variant);
                    continue;
                }
                for (RuntimeProfileCandidate child : expanded) {
                    String suffix = variant.name() + "+" + child.name();
                    Map<String, String> merged = new LinkedHashMap<>(variant.knobAssignments());
                    merged.putAll(child.knobAssignments());
                    next.add(new RuntimeProfileCandidate(suffix, child.runtimeProfile(), merged));
                }
            }
            current = List.copyOf(next);
        }
        return current;
    }
}
