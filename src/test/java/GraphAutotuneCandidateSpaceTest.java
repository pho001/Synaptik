import backend.runtime.ExecutionMode;
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
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.name().equals("acceleratorBuffer=off")));
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.name().equals("acceleratorBuffer=auto")));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.kind() == CandidateKind.GRAPH_STANDARD));
        assertTrue(candidates.stream()
                .filter(candidate -> !candidate.name().startsWith("acceleratorBuffer="))
                .allMatch(candidate -> candidate.metadata().runtimeFrozen()));
        assertTrue(candidates.stream()
                .filter(candidate -> candidate.name().startsWith("acceleratorBuffer="))
                .allMatch(candidate -> !candidate.metadata().runtimeFrozen()));
        assertTrue(candidates.stream().allMatch(candidate -> candidate.metadata().productionEligible()));
        assertTrue(candidates.stream()
                .filter(candidate -> candidate.name().equals("graphPolicy=current"))
                .allMatch(candidate -> !candidate.metadata().graphPolicyMutated()));
        assertTrue(candidates.stream()
                .filter(candidate -> !candidate.name().equals("graphPolicy=current"))
                .anyMatch(candidate -> candidate.metadata().graphPolicyMutated()));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "CPU_REGION_POLICY".equals(candidate.metadata().attributes().get("graphParameter"))));
    }

    @Test
    void standardModeGeneratesAcceleratorBufferRuntimeCandidates() {
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

        var off = candidates.stream()
                .filter(candidate -> candidate.name().equals("acceleratorBuffer=off"))
                .findFirst()
                .orElseThrow();
        var auto = candidates.stream()
                .filter(candidate -> candidate.name().equals("acceleratorBuffer=auto"))
                .findFirst()
                .orElseThrow();

        assertEquals("accelerator-buffer", off.metadata().parameterFamily());
        assertEquals("buffer-off", off.metadata().parameterVariant());
        assertEquals("OFF", off.metadata().attributes().get("acceleratorBufferBindingMode"));
        assertEquals("GRAPH_WORKLOAD", off.metadata().attributes().get("knobOwner"));
        assertTrue(off.metadata().attributes().get("knobAssignments")
                .contains("runtime.accelerator.metal.buffer.bindingMode"));
        assertEquals(config.runtime.AcceleratorBufferBindingMode.OFF,
                off.profile().runtime().accelerator().metal().buffer().bindingMode());
        assertEquals(config.runtime.AcceleratorBufferBindingMode.AUTO,
                auto.profile().runtime().accelerator().metal().buffer().bindingMode());
        assertEquals(runtime.toRuntimeConfig().blas(), off.profile().runtime().blas());
        assertEquals(GraphExecutionPolicy.fromExecutionProfile(seed).optimizer(), off.profile().optimizer());
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
        assertEquals(policy.optimizer(), candidate.profile().optimizer());
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
    void graphAutotuneMarksAcceleratorBufferModeAsGraphOwned() {
        var seed = seedProfile();
        var candidate = new GraphAutotuneCandidateSpace(
                "graph-standard",
                seed.dataType(),
                seed.mode(),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                GraphAutotuneMode.STANDARD
        ).generate(workload()).stream()
                .filter(generated -> generated.name().equals("acceleratorBuffer=auto"))
                .findFirst()
                .orElseThrow();

        assertEquals("GRAPH_WORKLOAD", candidate.metadata().attributes().get("knobOwner"));
        assertTrue(candidate.metadata().attributes().get("knobAssignments")
                .contains("runtime.accelerator.cuda.buffer.bindingMode"));
        assertTrue(candidate.metadata().attributes().get("knobAssignments")
                .contains("runtime.accelerator.opencl.buffer.bindingMode"));
        assertTrue(candidate.metadata().attributes().get("knobAssignments")
                .contains("runtime.accelerator.metal.buffer.bindingMode"));
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
