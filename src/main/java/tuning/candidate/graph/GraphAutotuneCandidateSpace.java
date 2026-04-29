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

/**
 * Candidate space used by graph autotune.
 *
 * <p>Generated candidates all share the supplied dtype, execution mode, and
 * platform runtime profile. Only the graph execution policy varies. Standard
 * mode emits production-eligible candidates; research mode emits exploratory
 * metadata so persistence consumers can distinguish them.</p>
 */
public final class GraphAutotuneCandidateSpace implements CandidateSpace {
    private final String profileName;
    private final DataType dataType;
    private final ExecutionMode executionMode;
    private final PlatformRuntimeProfile runtimeProfile;
    private final GraphExecutionPolicy graphPolicy;
    private final GraphAutotuneMode mode;

    /**
     * Creates a graph autotune candidate space.
     *
     * @param profileName profile namespace for generated execution profiles
     * @param dataType dtype for generated execution profiles
     * @param executionMode execution mode for generated execution profiles
     * @param runtimeProfile fixed runtime profile
     * @param graphPolicy seed graph policy to vary
     * @param mode graph autotune mode
     */
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

    /**
     * Generates graph-policy candidates. The current implementation does not
     * inspect the workload, but the parameter is retained for the candidate-space
     * contract.
     *
     * @param workload workload being tuned
     * @return graph-policy candidates for this mode
     */
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
        boolean graphPolicyMutated = !variant.policy().equals(graphPolicy);
        CandidateMetadata metadata = standard
                ? new CandidateMetadata(
                        "graph-autotune",
                        "1",
                        variant.parameter().name(),
                        variant.name(),
                        "STANDARD",
                        true,
                        graphPolicyMutated,
                        true,
                        java.util.Map.of()
                )
                : CandidateMetadata.graphResearch(
                        variant.parameter().name(),
                        variant.name(),
                        graphPolicyMutated
                );
        var optimizer = variant.policy().optimizer();
        return new Candidate(
                variant.name(),
                profile,
                standard ? CandidateKind.GRAPH_STANDARD : CandidateKind.GRAPH_RESEARCH,
                metadata.withAttribute("graphParameter", variant.parameter().name())
                        .withAttribute("offloadPolicy", optimizer.offload().policy().name())
                        .withAttribute("acceleratorRegionPolicy", optimizer.offload().acceleratorRegionPolicy().name())
                        .withAttribute("cpuRegionPolicy", optimizer.cpuRegion().policy().name())
                        .withAttribute("cpuFusionPolicy", optimizer.cpuFusion().mode().name())
                        .withAttribute("productionEligible", Boolean.toString(standard))
        );
    }
}
