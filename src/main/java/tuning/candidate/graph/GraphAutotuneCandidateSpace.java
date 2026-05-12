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
import tuning.ownership.TuningKnobOwnership;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

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
        List<GraphPolicyMutators.GraphPolicyVariant> graphVariants = mode == GraphAutotuneMode.RESEARCH
                ? GraphPolicyMutators.research(graphPolicy)
                : GraphPolicyMutators.standard(graphPolicy);
        List<GraphRuntimePolicyVariant> variants = new ArrayList<>();
        graphVariants.stream()
                .map(variant -> GraphRuntimePolicyVariant.fromGraphPolicy(variant, graphPolicy))
                .forEach(variants::add);
        return variants.stream().map(this::candidate).toList();
    }

    private Candidate candidate(GraphRuntimePolicyVariant variant) {
        TuningKnobOwnership.validateGraphWorkload(variant.knobAssignments(), variant.name());
        boolean standard = mode == GraphAutotuneMode.STANDARD;
        var profile = GraphPolicyCandidateAssembler.assemble(
                profileName,
                variant.name(),
                dataType,
                executionMode,
                runtimeProfile,
                variant.policy(),
                variant.runtimeOverride()
        );
        CandidateMetadata metadata = standard
                ? standardMetadata(variant)
                : CandidateMetadata.graphResearch(
                        variant.parameter().name(),
                        variant.name(),
                        variant.graphPolicyMutated()
                );
        var compile = variant.policy().compile();
        var backendPlanning = compile.backendPlanning();
        var regionOptimization = compile.regionOptimization();
        CandidateMetadata enriched = metadata.withAttribute("graphParameter", variant.parameter().name())
                .withAttribute("knobOwner", "GRAPH_WORKLOAD")
                .withAttribute("backendDiscoveryMode", backendPlanning.discoveryMode().name())
                .withAttribute("backendFailurePolicy", backendPlanning.failurePolicy().name())
                .withAttribute("ownershipPlanner", backendPlanning.ownershipPlanner().name())
                .withAttribute("metalTransferModel", backendPlanning.cost().planningCostProfile().metalTransferModel().name())
                .withAttribute("cpuRegionPolicy", backendPlanning.cpuRegions().policy().name())
                .withAttribute("cpuFusionPolicy", regionOptimization.cpuFusion().mode().name())
                .withAttribute("productionEligible", Boolean.toString(standard));
        if (!variant.knobAssignments().isEmpty()) {
            enriched = enriched.withAttribute("knobAssignments", variant.knobAssignments().keySet().stream()
                    .sorted()
                    .collect(Collectors.joining(",")));
        }
        for (var entry : variant.metadata().entrySet()) {
            enriched = enriched.withAttribute(entry.getKey(), entry.getValue());
        }
        return new Candidate(
                variant.name(),
                profile,
                standard ? CandidateKind.GRAPH_STANDARD : CandidateKind.GRAPH_RESEARCH,
                enriched
        );
    }

    private static CandidateMetadata standardMetadata(GraphRuntimePolicyVariant variant) {
        return new CandidateMetadata(
                "graph-autotune",
                "1",
                variant.parameter().name(),
                variant.name(),
                "STANDARD",
                !variant.runtimeMutated(),
                variant.graphPolicyMutated(),
                true,
                java.util.Map.of()
        );
    }

}
