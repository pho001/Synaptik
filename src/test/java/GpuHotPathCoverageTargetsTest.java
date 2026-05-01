import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuHotPathCoverageTarget;
import tuning.benchmark.report.GpuHotPathCoverageTargets;
import tuning.workload.StandardWorkloads;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuHotPathCoverageTargetsTest {
    @Test
    void defaultTargetsNameTransformerMlpConvAndNormalizationHotPaths() {
        List<String> names = GpuHotPathCoverageTargets.defaultWorkloadNames();

        assertEquals(List.of(
                "transformer_block_hot_path",
                "mlp_classifier_small",
                "conv2d_resnet_3x3",
                "layer_norm_small"
        ), names);
    }

    @Test
    void defaultTargetsExistInStandardWorkloadCatalog() {
        for (String name : GpuHotPathCoverageTargets.defaultWorkloadNames()) {
            assertEquals(name, StandardWorkloads.defaultCatalog().require(name).name());
        }
    }

    @Test
    void benchmarkSuiteCanBeBuiltForDefaultTargets() {
        BenchmarkSuiteRequest request = GpuHotPathCoverageTargets.benchmarkSuite(List.of(
                BenchmarkEntry.candidate("phase14-target-coverage", profile())
        ));
        List<String> names = request.workloads().stream().map(tuning.workload.WorkloadSpec::name).toList();

        assertEquals(4, request.workloads().size());
        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(names.contains("layer_norm_small"));
        assertEquals("phase14-target-coverage", request.entries().getFirst().name());
    }

    @Test
    void phaseNineteenHotPathTargetsIncludeMultiOpEvidenceTargets() {
        List<String> names = GpuHotPathCoverageTargets.defaultWorkloadNames();

        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(names.contains("layer_norm_small"));
    }

    @Test
    void targetsMapToDownstreamRequirementFamilies() {
        List<GpuHotPathCoverageTarget> targets = GpuHotPathCoverageTargets.defaults();
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUDAG")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUSTORAGE")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUNORM")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUFUSEX")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUMULTI")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUHARDEN")));
        assertEquals(19, targets.getFirst().ownerPhase());
        assertEquals(17, targets.getLast().ownerPhase());
    }

    @Test
    void phaseTwentyTargetsHaveHardeningPolicies() {
        List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();

        assertEquals(4, expectations.size());
        assertTrue(expectations.stream().allMatch(expectation -> "GPU_METAL".equals(expectation.backend())));
        assertTrue(expectations.stream().allMatch(expectation -> expectation.policy() != null));
        assertTrue(GpuHotPathCoverageTargets.defaults().stream()
                .allMatch(target -> target.requirementFamilies().contains("GPUHARDEN")));
    }

    @Test
    void phaseTwentyTargetPoliciesUsePhaseFourteenWorkloadNames() {
        List<String> expectationNames = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .map(GpuCoverageHotPathExpectation::workloadName)
                .toList();

        assertEquals(GpuHotPathCoverageTargets.defaultWorkloadNames(), expectationNames);
        assertEquals(List.of(
                "transformer_block_hot_path",
                "mlp_classifier_small",
                "conv2d_resnet_3x3",
                "layer_norm_small"
        ), expectationNames);
    }

    private static ExecutionProfile profile() {
        return new ExecutionProfile(
                "phase14-target-profile",
                "phase14-target-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );
    }
}
