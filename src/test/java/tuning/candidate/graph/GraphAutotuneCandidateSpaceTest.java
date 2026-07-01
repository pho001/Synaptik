package tuning.candidate.graph;

import runtime.contract.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.candidate.CandidateKind;
import tuning.ownership.TuningKnobOwner;
import tuning.ownership.TuningKnobOwnership;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraphAutotuneCandidateSpaceTest {
    @Test
    void standardCandidatesExposeProductionGraphPolicies() {
        var candidates = new GraphAutotuneCandidateSpace(
                "abc",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile(),
                GraphExecutionPolicy.trainingDefaults(),
                GraphAutotuneMode.STANDARD
        ).generate(null);

        assertTrue(candidates.size() > 1);
        assertTrue(candidates.stream().allMatch(candidate -> candidate.kind() == CandidateKind.GRAPH_STANDARD));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.metadata().productionEligible()));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "CPU_PARTITION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "CPU_FUSION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "BACKEND_DISCOVERY_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "OWNERSHIP_PLANNER_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.metadata().runtimeFrozen()));
        assertTrue(candidates.stream().allMatch(candidate ->
                "GRAPH_WORKLOAD".equals(candidate.metadata().attributes().get("knobOwner"))));
    }

    @Test
    void standardGraphAutotuneCandidatesOnlyMutateGraphOwnedKnobs() {
        var candidates = new GraphAutotuneCandidateSpace(
                "abc",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile(),
                GraphExecutionPolicy.trainingDefaults(),
                GraphAutotuneMode.STANDARD
        ).generate(null);

        for (var candidate : candidates) {
            String assignments = candidate.metadata().attributes().getOrDefault("knobAssignments", "");
            if (assignments.isBlank()) {
                continue;
            }
            for (String knob : assignments.split(",")) {
                assertEquals(TuningKnobOwner.GRAPH_WORKLOAD, TuningKnobOwnership.ownerOf(knob));
            }
        }
    }

    @Test
    void graphAutotuneMarksBackendPlanningAsGraphOwned() {
        var candidate = new GraphAutotuneCandidateSpace(
                "abc",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile(),
                GraphExecutionPolicy.trainingDefaults(),
                GraphAutotuneMode.STANDARD
        ).generate(null).stream()
                .filter(generated -> generated.name().contains("backendDiscovery=auto"))
                .findFirst()
                .orElseThrow();

        assertEquals("GRAPH_WORKLOAD", candidate.metadata().attributes().get("knobOwner"));
        assertTrue(candidate.metadata().attributes().get("knobAssignments")
                .contains("compile.backendPlanning.discoveryMode"));
    }

    @Test
    void researchCandidatesKeepResearchKnobsMarkedNonProduction() {
        PlatformRuntimeProfile runtime = runtimeProfile();
        var candidates = new GraphAutotuneCandidateSpace(
                "abc",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                runtime,
                GraphExecutionPolicy.trainingDefaults(),
                GraphAutotuneMode.RESEARCH
        ).generate(null);

        assertFalse(candidates.isEmpty());
        assertTrue(candidates.stream().allMatch(candidate -> candidate.kind() == CandidateKind.GRAPH_RESEARCH));
        assertTrue(candidates.stream().allMatch(candidate -> !candidate.metadata().productionEligible()));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "RESEARCH_CSE_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "MEMORY_PLANNING_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "PLANNING_COST_PROFILE".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "AGGRESSIVE".equals(candidate.metadata().attributes().get("transferCostPreset"))));
        assertEquals(runtime.toRuntimeConfig().blas(), candidates.getFirst().profile().runtime().blas());
        assertEquals(
                runtime.toRuntimeConfig().cpuKernelConfig().cheapVectorMinSize(),
                candidates.getFirst().profile().runtime().cpuKernelConfig().cheapVectorMinSize()
        );
    }

    private static PlatformRuntimeProfile runtimeProfile() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults()
        );
        return PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed);
    }
}
