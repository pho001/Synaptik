import backend.runtime.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.candidate.CandidateKind;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GraphAutotuneCandidateSpaceTest {
    @Test
    void standardModeGeneratesOnlyCurrentGraphPolicyCandidate() {
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

        assertEquals(1, candidates.size());
        var candidate = candidates.getFirst();
        assertEquals("graphPolicy=current", candidate.name());
        assertEquals(CandidateKind.GRAPH_STANDARD, candidate.kind());
        assertTrue(candidate.metadata().runtimeFrozen());
        assertFalse(candidate.metadata().graphPolicyMutated());
        assertTrue(candidate.metadata().productionEligible());
    }

    @Test
    void standardModeDoesNotMutateRuntimeOrOptimizerConfig() {
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
        ).generate(workload()).getFirst();

        var expectedRuntime = runtime.toRuntimeConfig();
        assertEquals(expectedRuntime.blas(), candidate.profile().runtime().blas());
        assertEquals(expectedRuntime.conv2d(), candidate.profile().runtime().conv2d());
        assertEquals(expectedRuntime.accelerator(), candidate.profile().runtime().accelerator());
        assertEquals(
                expectedRuntime.cpuKernelConfig().cheapVectorMinSize(),
                candidate.profile().runtime().cpuKernelConfig().cheapVectorMinSize()
        );
        assertEquals(policy.optimizer(), candidate.profile().optimizer());
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
        assertTrue(candidates.stream().noneMatch(candidate -> candidate.name().contains("partition")));
    }

    private static config.profile.ExecutionProfile seedProfile() {
        return new config.profile.ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
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
