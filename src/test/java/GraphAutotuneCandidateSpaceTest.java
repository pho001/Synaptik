import runtime.contract.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.candidate.CandidateKind;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.ownership.TuningKnobOwner;
import tuning.ownership.TuningKnobOwnership;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GraphAutotuneCandidateSpaceTest {
    @Test
    void standardModeGeneratesProductionGraphPolicyCandidates() {
        var seed = seedProfile();
        var space = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.STANDARD
        );

        var candidates = space.generate(workload());

        assertTrue(candidates.size() > 1);
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.name().equals("graphPolicy=current")));
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.name().contains("backendDiscovery=auto")));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.kind() == CandidateKind.GRAPH_STANDARD));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.metadata().runtimeFrozen()));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.metadata().productionEligible()));
        assertTrue(candidates.stream()
                .filter(candidate -> candidate.name().equals("graphPolicy=current"))
                .allMatch(candidate -> !candidate.metadata().graphPolicyMutated()));
        assertTrue(candidates.stream()
                .filter(candidate -> !candidate.name().equals("graphPolicy=current"))
                .anyMatch(candidate -> candidate.metadata().graphPolicyMutated()));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "CPU_PARTITION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
    }

    @Test
    void standardModeKeepsRuntimeFixed() {
        var seed = seedProfile();
        var runtime = PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed);
        var candidates = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                runtime,
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.STANDARD
        ).generate(workload());

        var auto = candidates.stream()
                .filter(candidate -> candidate.name().contains("backendDiscovery=auto"))
                .findFirst()
                .orElseThrow();

        assertEquals("BACKEND_DISCOVERY_POLICY", auto.metadata().parameterFamily());
        assertEquals("GRAPH_WORKLOAD", auto.metadata().attributes().get("knobOwner"));
        assertTrue(auto.metadata().attributes().get("knobAssignments")
                .contains("compile.backendPlanning.discoveryMode"));
        assertEquals(runtime.toRuntimeConfig().accelerator(), auto.profile().runtime().accelerator());
        assertEquals(runtime.toRuntimeConfig().blas(), auto.profile().runtime().blas());
        assertEquals("AUTO", auto.profile().compile().backendPlanning().discoveryMode().name());
    }

    @Test
    void standardModeDoesNotMutateRuntimeConfig() {
        var seed = seedProfile();
        var runtime = PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed);
        var policy = GraphExecutionPolicy.fromExecutionProfile(seed);
        var candidate = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                runtime,
                policy,
                GraphAutotuneMode.STANDARD
        ).generate(workload()).stream()
                .filter(generated -> generated.name().equals("graphPolicy=current"))
                .findFirst()
                .orElseThrow();

        var expectedRuntime = runtime.toRuntimeConfig();
        assertEquals(expectedRuntime.blas(), candidate.profile().runtime().blas());
        assertEquals(expectedRuntime.conv2d(), candidate.profile().runtime().conv2d());
        assertEquals(expectedRuntime.accelerator(), candidate.profile().runtime().accelerator());
        assertEquals(
                expectedRuntime.cpuKernelConfig().cheapVectorMinSize(),
                candidate.profile().runtime().cpuKernelConfig().cheapVectorMinSize()
        );
        assertEquals(policy.compile(), candidate.profile().compile());
    }

    @Test
    void standardGraphAutotuneCandidatesOnlyMutateGraphOwnedKnobs() {
        var seed = seedProfile();
        var candidates = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.STANDARD
        ).generate(workload());

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
        var seed = seedProfile();
        var candidate = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.STANDARD
        ).generate(workload()).stream()
                .filter(generated -> generated.name().contains("backendDiscovery=auto"))
                .findFirst()
                .orElseThrow();

        assertEquals("GRAPH_WORKLOAD", candidate.metadata().attributes().get("knobOwner"));
        assertTrue(candidate.metadata().attributes().get("knobAssignments")
                .contains("compile.backendPlanning.discoveryMode"));
        assertEquals(
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed).toRuntimeConfig().accelerator(),
                candidate.profile().runtime().accelerator()
        );
    }

    @Test
    void researchModeIsExplicitAndNotProductionEligible() {
        var seed = seedProfile();
        var candidates = new GraphAutotuneCandidateSpace(
                "graph-research",
                seed.dataType(),
                seed.mode(),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.RESEARCH
        ).generate(workload());

        assertTrue(candidates.size() > 1);
        assertTrue(candidates.stream().allMatch(candidate -> candidate.kind() == CandidateKind.GRAPH_RESEARCH));
        assertTrue(candidates.stream().allMatch(candidate -> !candidate.metadata().productionEligible()));
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.name().equals("cse=aggressive")));
        assertTrue(candidates.stream()
                .filter(candidate -> candidate.name().equals("piecewise=current"))
                .allMatch(candidate -> !candidate.metadata().graphPolicyMutated()));
        assertTrue(candidates.stream().noneMatch(candidate -> candidate.name().contains("stageOrder")));
        assertTrue(candidates.stream().noneMatch(candidate -> candidate.name().contains("conv2dLowering")));
    }

    private static config.profile.ExecutionProfile seedProfile() {
        return new config.profile.ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults()
        );
    }

    private static TensorRootWorkloadSpec workload() {
        return new TensorRootWorkloadSpec(
                "graph_autotune",
                WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        );
    }
}
