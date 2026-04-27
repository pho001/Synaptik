package tuning.candidate.graph;

import backend.runtime.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;
import tuning.candidate.CandidateSpace;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

public final class GraphAutotuneCandidateSpace implements CandidateSpace {
    private final String profileName;
    private final DataType dataType;
    private final ExecutionMode executionMode;
    private final PlatformRuntimeProfile runtimeProfile;
    private final GraphExecutionPolicy graphPolicy;
    private final GraphAutotuneMode mode;

    public GraphAutotuneCandidateSpace(
            String profileName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy,
            GraphAutotuneMode mode
    ) {
        this.profileName = profileName == null || profileName.isBlank() ? "graph-autotune" : profileName;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode cannot be null");
        this.runtimeProfile = Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        this.graphPolicy = Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        this.mode = mode == null ? GraphAutotuneMode.STANDARD : mode;
    }

    @Override
    public List<Candidate> generate(WorkloadSpec workload) {
        List<GraphPolicyMutators.GraphPolicyVariant> variants = mode == GraphAutotuneMode.RESEARCH
                ? GraphPolicyMutators.research(graphPolicy)
                : GraphPolicyMutators.standard(graphPolicy);
        return variants.stream().map(this::candidate).toList();
    }

    private Candidate candidate(GraphPolicyMutators.GraphPolicyVariant variant) {
        boolean standard = mode == GraphAutotuneMode.STANDARD;
        var profile = GraphPolicyCandidateAssembler.assemble(
                profileName,
                variant.name(),
                dataType,
                executionMode,
                runtimeProfile,
                variant.policy()
        );
        CandidateMetadata metadata = standard
                ? CandidateMetadata.graphStandard("current")
                : CandidateMetadata.graphResearch(
                        variant.parameter().name(),
                        variant.name(),
                        !variant.policy().equals(graphPolicy)
                );
        return new Candidate(
                variant.name(),
                profile,
                standard ? CandidateKind.GRAPH_STANDARD : CandidateKind.GRAPH_RESEARCH,
                metadata.withAttribute("graphParameter", variant.parameter().name())
        );
    }
}
