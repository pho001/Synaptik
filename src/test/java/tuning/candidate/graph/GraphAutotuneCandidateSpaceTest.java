package tuning.candidate.graph;

import backend.runtime.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.candidate.CandidateKind;

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
                "CPU_REGION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "CPU_FUSION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "OFFLOAD_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "ACCELERATOR_BUFFER_MODE".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream()
                .filter(candidate -> "ACCELERATOR_BUFFER_MODE".equals(candidate.metadata().attributes().get("graphParameter")))
                .allMatch(candidate -> "accelerator-buffer".equals(candidate.metadata().parameterFamily())));
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
                "RESEARCH_MEMORY_LIFETIME".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "RESEARCH_METAL_TRANSFER_MODEL".equals(candidate.metadata().attributes().get("graphParameter"))));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "AGGRESSIVE".equals(candidate.metadata().attributes().get("metalTransferModel"))));
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
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults()
        );
        return PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed);
    }
}
