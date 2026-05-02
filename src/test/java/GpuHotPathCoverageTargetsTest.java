import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuHotPathCoverageTarget;
import tuning.benchmark.report.GpuHotPathCoverageTargets;
import tuning.benchmark.report.GpuTargetCoverageTruth;
import tuning.benchmark.report.GpuTargetExecutionStatus;
import tuning.workload.StandardWorkloads;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuHotPathCoverageTargetsTest {
    @Test
    void defaultTargetsNameV14GpuCoverageClosureHotPaths() {
        List<String> names = GpuHotPathCoverageTargets.defaultWorkloadNames();

        assertEquals(List.of(
                "reduction_chain_small",
                "transformer_block_hot_path",
                "mlp_classifier_small",
                "mlp_classifier_small_bf16",
                "conv2d_resnet_3x3",
                "max_pool2d_small",
                "layer_norm_small",
                "layer_norm_small_bf16",
                "rms_norm_small",
                "rms_norm_small_bf16",
                "reduction_chain_small_bf16",
                "cross_entropy_small",
                "bool_compare_where_small",
                "gather_take_small",
                "layout_broadcast_repair_small"
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

        assertEquals(15, request.workloads().size());
        assertTrue(names.contains("reduction_chain_small"));
        assertTrue(names.contains("reduction_chain_small_bf16"));
        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("mlp_classifier_small_bf16"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(names.contains("max_pool2d_small"));
        assertTrue(names.contains("layer_norm_small"));
        assertTrue(names.contains("layer_norm_small_bf16"));
        assertTrue(names.contains("rms_norm_small"));
        assertTrue(names.contains("rms_norm_small_bf16"));
        assertTrue(names.contains("cross_entropy_small"));
        assertTrue(names.contains("bool_compare_where_small"));
        assertTrue(names.contains("gather_take_small"));
        assertTrue(names.contains("layout_broadcast_repair_small"));
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
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPURED")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUNORMX")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUSDPA")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPULOSSIDX")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUCONVBOOL")));
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALINTIDX"))
                .count());
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALLAYOUT"))
                .count());
        assertEquals(4, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALBF16"))
                .count());
        assertTrue(targets.stream().allMatch(target -> target.requirementFamilies().contains("GPUNATIVE")));
        assertTrue(targets.stream().allMatch(target -> target.requirementFamilies().contains("GPUCLOSE")));
        assertEquals(23, targets.getFirst().ownerPhase());
        assertEquals(33, targets.getLast().ownerPhase());
    }

    @Test
    void phaseTwentyTargetsHaveHardeningPolicies() {
        List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();

        assertEquals(15, expectations.size());
        assertTrue(expectations.stream().allMatch(expectation -> "GPU_METAL".equals(expectation.backend())));
        assertTrue(expectations.stream().allMatch(expectation -> expectation.policy() != null));
        assertTrue(GpuHotPathCoverageTargets.defaults().stream()
                .allMatch(target -> target.requirementFamilies().contains("GPUCLOSE")));
    }

    @Test
    void phaseTwentyTargetPoliciesUsePhaseFourteenWorkloadNames() {
        List<String> expectationNames = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .map(GpuCoverageHotPathExpectation::workloadName)
                .toList();

        assertEquals(GpuHotPathCoverageTargets.defaultWorkloadNames(), expectationNames);
        assertEquals(List.of(
                "reduction_chain_small",
                "transformer_block_hot_path",
                "mlp_classifier_small",
                "mlp_classifier_small_bf16",
                "conv2d_resnet_3x3",
                "max_pool2d_small",
                "layer_norm_small",
                "layer_norm_small_bf16",
                "rms_norm_small",
                "rms_norm_small_bf16",
                "reduction_chain_small_bf16",
                "cross_entropy_small",
                "bool_compare_where_small",
                "gather_take_small",
                "layout_broadcast_repair_small"
        ), expectationNames);
    }

    @Test
    void phaseTwentyFourNormalizationTargetsRequireSupportedGpuCoverage() {
        List<GpuCoverageHotPathExpectation> normalization = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(expectation -> expectation.workloadName().equals("layer_norm_small")
                        || expectation.workloadName().equals("rms_norm_small"))
                .toList();

        assertEquals(2, normalization.size());
        assertTrue(normalization.stream().allMatch(GpuCoverageHotPathExpectation::nativeEvidenceRequired));
        assertTrue(normalization.stream().allMatch(expectation -> expectation.expectedVisibleReasons().isEmpty()));
        assertTrue(normalization.stream().allMatch(expectation -> expectation.policy().minLoweredPrimitiveCount() >= 5));
        assertTrue(normalization.stream().allMatch(expectation -> expectation.policy().requireNativeBufferBinding()));
    }

    @Test
    void phaseTwentyFiveTransformerExpectationsReflectMetalSupportAndCudaFallback() {
        GpuCoverageHotPathExpectation metal = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(expectation -> expectation.workloadName().equals("transformer_block_hot_path"))
                .findFirst()
                .orElseThrow();
        GpuCoverageHotPathExpectation cuda = GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA")
                .stream()
                .filter(expectation -> expectation.workloadName().equals("transformer_block_hot_path"))
                .findFirst()
                .orElseThrow();

        assertTrue(metal.nativeEvidenceRequired());
        assertTrue(metal.expectedVisibleReasons().isEmpty());
        assertTrue(metal.policy().requireNativeBufferBinding());
        assertEquals(0, metal.policy().maxFallbackCount());
        assertEquals(0, metal.policy().maxTensorArrayStepCount());
        assertTrue(metal.policy().minLoweredPrimitiveCount() >= 1);

        assertEquals("GPU_CUDA", cuda.backend());
        assertTrue(!cuda.nativeEvidenceRequired());
        assertTrue(cuda.expectedVisibleReasons().contains("CAPABILITY_MISSING"));
        assertTrue(cuda.expectedVisibleReasons().contains("SCALED_DOT_PRODUCT_ATTENTION"));
    }

    @Test
    void phaseTwentyEightReductionTargetsRequireNativeCoverageForBothAccelerators() {
        for (String backend : List.of("GPU_METAL", "GPU_CUDA")) {
            GpuCoverageHotPathExpectation reduction = GpuHotPathCoverageTargets.expectationsForBackend(backend)
                    .stream()
                    .filter(expectation -> expectation.workloadName().equals("reduction_chain_small"))
                    .findFirst()
                    .orElseThrow();
            GpuCoverageGatePolicy policy = reduction.policy();

            assertTrue(reduction.nativeEvidenceRequired());
            assertTrue(reduction.expectedVisibleReasons().isEmpty());
            assertTrue(policy.requireNativeBufferBinding());
            assertEquals(0, policy.maxCpuMaterializationCount());
            assertEquals(0, policy.maxFallbackCount());
            assertEquals(0, policy.maxTensorArrayStepCount());
            assertTrue(policy.minMaxSelectedRegionLength() >= 1);
            assertTrue(policy.minLoweredPrimitiveCount() >= 1);
        }
    }

    @Test
    void phaseTwentyEightSupportedTargetsHaveHardNativePolicies() {
        Map<String, GpuCoverageHotPathExpectation> metal = expectationsByName("GPU_METAL");
        Map<String, GpuCoverageHotPathExpectation> cuda = expectationsByName("GPU_CUDA");

        assertHardNativePolicy(metal.get("reduction_chain_small"));
        assertHardNativePolicy(cuda.get("reduction_chain_small"));
        assertHardNativePolicy(metal.get("layer_norm_small"));
        assertHardNativePolicy(cuda.get("layer_norm_small"));
        assertHardNativePolicy(metal.get("rms_norm_small"));
        assertHardNativePolicy(cuda.get("rms_norm_small"));
        assertHardNativePolicy(metal.get("transformer_block_hot_path"));
        assertHardNativePolicy(metal.get("mlp_classifier_small"));
        assertHardNativePolicy(metal.get("mlp_classifier_small_bf16"));
        assertHardNativePolicy(metal.get("layer_norm_small_bf16"));
        assertHardNativePolicy(metal.get("rms_norm_small_bf16"));
        assertHardNativePolicy(metal.get("reduction_chain_small_bf16"));
        assertHardNativePolicy(metal.get("bool_compare_where_small"));
        assertHardNativePolicy(metal.get("gather_take_small"));
        GpuCoverageHotPathExpectation layoutRepair = metal.get("layout_broadcast_repair_small");
        assertTrue(layoutRepair.nativeEvidenceRequired());
        assertTrue(layoutRepair.expectedVisibleReasons().isEmpty());
        assertTrue(layoutRepair.policy().requireNativeBufferBinding());
        assertEquals(1, layoutRepair.policy().maxCpuMaterializationCount());
        assertEquals(0, layoutRepair.policy().maxFallbackCount());
        assertEquals(0, layoutRepair.policy().maxTensorArrayStepCount());
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "bool_compare_where_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "gather_take_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "layout_broadcast_repair_small");

        assertEquals(
                GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                GpuTargetCoverageTruth.rowsFor(backend.ComputeBackend.GPU_METAL).stream()
                        .filter(row -> row.opType() == operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)
                        .findFirst()
                        .orElseThrow()
                        .executionStatus()
        );
    }

    @Test
    void phaseTwentyEightUnsupportedTargetsRequireVisibleReasons() {
        List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();

        assertVisibleBlocker(expectations, "conv2d_resnet_3x3");
        assertVisibleBlocker(expectations, "max_pool2d_small");
        assertVisibleBlocker(expectations, "cross_entropy_small");

        GpuCoverageHotPathExpectation cudaSdpa = expectationsByName("GPU_CUDA").get("transformer_block_hot_path");
        assertTrue(!cudaSdpa.nativeEvidenceRequired());
        assertTrue(cudaSdpa.expectedVisibleReasons().contains("CAPABILITY_MISSING"));
    }

    @Test
    void phaseThirtyBf16TargetsUseMetalOnlyHardNativePolicies() {
        Map<String, GpuCoverageHotPathExpectation> metal = expectationsByName("GPU_METAL");
        Map<String, GpuCoverageHotPathExpectation> cuda = expectationsByName("GPU_CUDA");

        for (String workload : List.of(
                "mlp_classifier_small_bf16",
                "layer_norm_small_bf16",
                "rms_norm_small_bf16",
                "reduction_chain_small_bf16"
        )) {
            assertHardNativePolicy(metal.get(workload));
            assertTrue(metal.get(workload).nativeEvidenceRequired(), workload);
            assertTrue(!cuda.get(workload).nativeEvidenceRequired(), workload);
            assertEquals(0, cuda.get(workload).policy().minLoweredPrimitiveCount(), workload);
        }
    }

    private static Map<String, GpuCoverageHotPathExpectation> expectationsByName(String backend) {
        return GpuHotPathCoverageTargets.expectationsForBackend(backend)
                .stream()
                .collect(Collectors.toMap(GpuCoverageHotPathExpectation::workloadName, expectation -> expectation));
    }

    private static void assertHardNativePolicy(GpuCoverageHotPathExpectation expectation) {
        assertTrue(expectation.nativeEvidenceRequired());
        assertTrue(expectation.expectedVisibleReasons().isEmpty());
        assertTrue(expectation.policy().requireNativeBufferBinding());
        assertEquals(0, expectation.policy().maxCpuMaterializationCount());
        assertEquals(0, expectation.policy().maxFallbackCount());
        assertEquals(0, expectation.policy().maxTensorArrayStepCount());
    }

    private static void assertVisibleBlocker(
            List<GpuCoverageHotPathExpectation> expectations,
            String workloadName
    ) {
        GpuCoverageHotPathExpectation expectation = expectations.stream()
                .filter(item -> item.workloadName().equals(workloadName))
                .findFirst()
                .orElseThrow();
        assertTrue(!expectation.nativeEvidenceRequired());
        assertTrue(!expectation.expectedVisibleReasons().isEmpty());
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
