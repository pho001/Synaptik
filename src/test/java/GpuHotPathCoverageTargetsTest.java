import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuHotPathCoverageTarget;
import tuning.benchmark.report.CudaHotPathBlockerClass;
import tuning.benchmark.report.CudaHotPathBlockerPolicy;
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
                "avg_pool2d_small",
                "layer_norm_small",
                "layer_norm_small_bf16",
                "rms_norm_small",
                "rms_norm_small_bf16",
                "reduction_chain_small_bf16",
                "dense_loss_small",
                "cross_entropy_small",
                "training_transformer_block_hot_path",
                "training_dense_loss_small",
                "training_reduction_chain_small",
                "training_layer_norm_small",
                "training_cross_entropy_small",
                "bool_compare_where_small",
                "gather_take_small",
                "scatter_index_gradient_small",
                "layout_broadcast_repair_small",
                "masked_sdpa_small"
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

        assertEquals(24, request.workloads().size());
        assertTrue(names.contains("reduction_chain_small"));
        assertTrue(names.contains("reduction_chain_small_bf16"));
        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("mlp_classifier_small_bf16"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(names.contains("max_pool2d_small"));
        assertTrue(names.contains("avg_pool2d_small"));
        assertTrue(names.contains("layer_norm_small"));
        assertTrue(names.contains("layer_norm_small_bf16"));
        assertTrue(names.contains("rms_norm_small"));
        assertTrue(names.contains("rms_norm_small_bf16"));
        assertTrue(names.contains("dense_loss_small"));
        assertTrue(names.contains("cross_entropy_small"));
        assertTrue(names.contains("training_transformer_block_hot_path"));
        assertTrue(names.contains("training_dense_loss_small"));
        assertTrue(names.contains("training_reduction_chain_small"));
        assertTrue(names.contains("training_layer_norm_small"));
        assertTrue(names.contains("training_cross_entropy_small"));
        assertTrue(names.contains("bool_compare_where_small"));
        assertTrue(names.contains("gather_take_small"));
        assertTrue(names.contains("scatter_index_gradient_small"));
        assertTrue(names.contains("layout_broadcast_repair_small"));
        assertTrue(names.contains("masked_sdpa_small"));
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
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("METALLOSS")));
        assertTrue(targets.stream().anyMatch(target -> target.requirementFamilies().contains("GPUCONVBOOL")));
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALINTIDX"))
                .count());
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALSCATTER"))
                .count());
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALLAYOUT"))
                .count());
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALSDPAMASK"))
                .count());
        assertEquals(4, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALBF16"))
                .count());
        assertEquals(6, targets.stream()
                .filter(target -> target.requirementFamilies().contains("CUDADTYPE"))
                .count());
        assertEquals(1, targets.stream()
                .filter(target -> target.requirementFamilies().contains("CUDAINDEX"))
                .count());
        assertEquals(5, targets.stream()
                .filter(target -> target.requirementFamilies().contains("CUDANN"))
                .count());
        assertEquals(6, targets.stream()
                .filter(target -> target.requirementFamilies().contains("CUDATRAIN"))
                .count());
        assertTrue(targets.stream().allMatch(target -> target.requirementFamilies().contains("GPUNATIVE")));
        assertTrue(targets.stream().allMatch(target -> target.requirementFamilies().contains("GPUCLOSE")));
        assertEquals(23, targets.getFirst().ownerPhase());
        assertEquals(34, targets.getLast().ownerPhase());
        assertEquals(37, targets.stream()
                .filter(target -> target.workloadName().equals("dense_loss_small"))
                .findFirst()
                .orElseThrow()
                .ownerPhase());
        assertEquals(36, targets.stream()
                .filter(target -> target.workloadName().equals("scatter_index_gradient_small"))
                .findFirst()
                .orElseThrow()
                .ownerPhase());
        assertEquals(3, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALCONVPOOL"))
                .count());
        assertEquals(5, targets.stream()
                .filter(target -> target.requirementFamilies().contains("METALTRAIN"))
                .count());
    }

    @Test
    void phaseTwentyTargetsHaveHardeningPolicies() {
        List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();

        assertEquals(24, expectations.size());
        assertTrue(expectations.stream().allMatch(expectation -> "GPU_METAL".equals(expectation.backend())));
        assertTrue(expectations.stream().allMatch(expectation -> expectation.policy() != null));
        assertTrue(GpuHotPathCoverageTargets.defaults().stream()
                .allMatch(target -> target.requirementFamilies().contains("GPUCLOSE")));
    }

    @Test
    void cudaHotPathBlockerPolicyNamesV16Targets() {
        assertEquals(CudaHotPathBlockerClass.V16_BLOCKER,
                CudaHotPathBlockerPolicy.classify("masked_sdpa_small"));
        assertEquals(CudaHotPathBlockerClass.V16_BLOCKER,
                CudaHotPathBlockerPolicy.classify("conv2d_resnet_3x3"));
        assertEquals(CudaHotPathBlockerClass.V16_BLOCKER,
                CudaHotPathBlockerPolicy.classify("dense_loss_small"));
        assertEquals(CudaHotPathBlockerClass.V16_BLOCKER,
                CudaHotPathBlockerPolicy.classify("gather_take_small"));
        assertEquals(CudaHotPathBlockerClass.V16_BLOCKER,
                CudaHotPathBlockerPolicy.classify("bool_compare_where_small"));
        assertEquals(CudaHotPathBlockerClass.REQUIRES_NATIVE_EVIDENCE,
                CudaHotPathBlockerPolicy.classify("scatter_index_gradient_small"));
        assertEquals(CudaHotPathBlockerClass.ACCEPTED_CAPABILITY_GAP,
                CudaHotPathBlockerPolicy.classify("layer_norm_small_bf16"));
        assertTrue(CudaHotPathBlockerPolicy.v16BlockerTargets().contains("masked_sdpa_small"));
        assertTrue(CudaHotPathBlockerPolicy.v16BlockerTargets().contains("conv2d_resnet_3x3"));
        assertTrue(CudaHotPathBlockerPolicy.v16BlockerTargets().contains("dense_loss_small"));
        assertTrue(CudaHotPathBlockerPolicy.v16BlockerTargets().contains("gather_take_small"));
        assertTrue(CudaHotPathBlockerPolicy.v16BlockerTargets().contains("bool_compare_where_small"));
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
                "avg_pool2d_small",
                "layer_norm_small",
                "layer_norm_small_bf16",
                "rms_norm_small",
                "rms_norm_small_bf16",
                "reduction_chain_small_bf16",
                "dense_loss_small",
                "cross_entropy_small",
                "training_transformer_block_hot_path",
                "training_dense_loss_small",
                "training_reduction_chain_small",
                "training_layer_norm_small",
                "training_cross_entropy_small",
                "bool_compare_where_small",
                "gather_take_small",
                "scatter_index_gradient_small",
                "layout_broadcast_repair_small",
                "masked_sdpa_small"
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
    void phaseFortyTwoCudaNnTargetsStayVisibleBlockersUntilNativeEvidenceExists() {
        Map<String, GpuCoverageHotPathExpectation> cuda = expectationsByName("GPU_CUDA");

        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "masked_sdpa_small");
        assertTrue(cuda.get("masked_sdpa_small").expectedVisibleReasons().contains("masked_sdpa_small"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "conv2d_resnet_3x3");
        assertTrue(cuda.get("conv2d_resnet_3x3").expectedVisibleReasons().contains("CONV_POOL"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "max_pool2d_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "avg_pool2d_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "dense_loss_small");
        assertTrue(cuda.get("dense_loss_small").expectedVisibleReasons().contains("DAG_PRIMITIVE_UNSUPPORTED"));
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
        assertHardNativePolicy(metal.get("masked_sdpa_small"));
        assertHardNativePolicy(metal.get("conv2d_resnet_3x3"));
        assertHardNativePolicy(metal.get("max_pool2d_small"));
        assertHardNativePolicy(metal.get("avg_pool2d_small"));
        assertHardNativePolicy(metal.get("dense_loss_small"));
        GpuCoverageHotPathExpectation layoutRepair = metal.get("layout_broadcast_repair_small");
        assertTrue(layoutRepair.nativeEvidenceRequired());
        assertTrue(layoutRepair.expectedVisibleReasons().isEmpty());
        assertTrue(layoutRepair.policy().requireNativeBufferBinding());
        assertEquals(1, layoutRepair.policy().maxCpuMaterializationCount());
        assertEquals(0, layoutRepair.policy().maxFallbackCount());
        assertEquals(0, layoutRepair.policy().maxTensorArrayStepCount());
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "bool_compare_where_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "gather_take_small");
        assertHardNativePolicy(metal.get("scatter_index_gradient_small"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "scatter_index_gradient_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "layout_broadcast_repair_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "masked_sdpa_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "dense_loss_small");

        assertEquals(
                GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                GpuTargetCoverageTruth.rowsFor(backend.contract.ComputeBackend.GPU_METAL).stream()
                        .filter(row -> row.opType() == operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)
                        .findFirst()
                        .orElseThrow()
                        .executionStatus()
        );
    }

    @Test
    void phaseTwentyEightUnsupportedTargetsRequireVisibleReasons() {
        List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();

        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "conv2d_resnet_3x3");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "max_pool2d_small");
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "avg_pool2d_small");
        assertHardNativePolicy(expectationsByName("GPU_METAL").get("cross_entropy_small"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "cross_entropy_small");
        assertHardNativePolicy(expectationsByName("GPU_METAL").get("scatter_index_gradient_small"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "scatter_index_gradient_small");

        GpuCoverageHotPathExpectation cudaSdpa = expectationsByName("GPU_CUDA").get("transformer_block_hot_path");
        assertTrue(!cudaSdpa.nativeEvidenceRequired());
        assertTrue(cudaSdpa.expectedVisibleReasons().contains("CAPABILITY_MISSING"));
    }

    @Test
    void phaseThirtySevenLossTargetsSeparateDenseNativeFromIndexBlocker() {
        Map<String, GpuCoverageHotPathExpectation> metal = expectationsByName("GPU_METAL");
        Map<String, GpuCoverageHotPathExpectation> cuda = expectationsByName("GPU_CUDA");

        assertHardNativePolicy(metal.get("dense_loss_small"));
        assertTrue(metal.get("dense_loss_small").policy().minLoweredPrimitiveCount() >= 3);

        assertHardNativePolicy(metal.get("cross_entropy_small"));
        assertTrue(metal.get("cross_entropy_small").expectedVisibleReasons().isEmpty());
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "dense_loss_small");
        assertTrue(cuda.get("dense_loss_small").expectedVisibleReasons().contains("DAG_PRIMITIVE_UNSUPPORTED"));

        Map<operations.Operation.OpType, GpuTargetCoverageTruth.Row> rows = GpuTargetCoverageTruth.rowsFor(backend.contract.ComputeBackend.GPU_METAL)
                .stream()
                .collect(Collectors.toMap(GpuTargetCoverageTruth.Row::opType, row -> row));
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.NLL_LOSS).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.SCATTER_ADD).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.GATHER_GRAD).executionStatus());
        assertEquals(GpuTargetExecutionStatus.NATIVE_EXECUTABLE, rows.get(operations.Operation.OpType.TAKE_ALONG_AXIS_GRAD).executionStatus());
    }

    @Test
    void phaseThirtyEightTrainingTargetsSeparatePublicationFromInternalMaterialization() {
        Map<String, GpuCoverageHotPathExpectation> metal = expectationsByName("GPU_METAL");

        for (String workload : List.of(
                "training_transformer_block_hot_path",
                "training_dense_loss_small",
                "training_reduction_chain_small",
                "training_layer_norm_small",
                "training_cross_entropy_small"
        )) {
            GpuCoverageHotPathExpectation expectation = metal.get(workload);
            assertTrue(expectation.nativeEvidenceRequired(), workload);
            assertTrue(expectation.expectedVisibleReasons().isEmpty(), workload);
            assertTrue(expectation.policy().requireNativeBufferBinding(), workload);
            assertEquals(0, expectation.policy().maxInternalCpuMaterializationCount(), workload);
            assertTrue(expectation.policy().maxGradientPublicationMaterializationCount() > 0, workload);
            assertEquals(
                    expectation.policy().maxGradientPublicationMaterializationCount(),
                    expectation.policy().maxCpuMaterializationCount(),
                workload
            );
        }
    }

    @Test
    void phaseFortyThreeCudaTrainingTargetsSeparateSupportedRowsFromVisibleBlockers() {
        Map<String, GpuCoverageHotPathExpectation> cuda = expectationsByName("GPU_CUDA");

        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "training_transformer_block_hot_path");
        assertTrue(cuda.get("training_transformer_block_hot_path").expectedVisibleReasons()
                .contains("SCALED_DOT_PRODUCT_ATTENTION_BACKWARD"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "training_dense_loss_small");
        assertTrue(cuda.get("training_dense_loss_small").expectedVisibleReasons()
                .contains("DAG_PRIMITIVE_UNSUPPORTED"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "training_cross_entropy_small");
        assertTrue(cuda.get("training_cross_entropy_small").expectedVisibleReasons()
                .contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertVisibleBlocker(GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA"), "scatter_index_gradient_small");

        assertTrue(cuda.get("training_reduction_chain_small").nativeEvidenceRequired());
        assertTrue(cuda.get("training_reduction_chain_small").expectedVisibleReasons().isEmpty());
        assertTrue(cuda.get("training_reduction_chain_small").policy().requireNativeBufferBinding());
        assertEquals(0, cuda.get("training_reduction_chain_small").policy().maxInternalCpuMaterializationCount());
        assertTrue(cuda.get("training_reduction_chain_small").policy().maxGradientPublicationMaterializationCount() > 0);
        assertTrue(cuda.get("training_layer_norm_small").nativeEvidenceRequired());
        assertTrue(cuda.get("training_layer_norm_small").expectedVisibleReasons().isEmpty());
        assertTrue(cuda.get("training_layer_norm_small").policy().requireNativeBufferBinding());
        assertEquals(0, cuda.get("training_layer_norm_small").policy().maxInternalCpuMaterializationCount());
        assertTrue(cuda.get("training_layer_norm_small").policy().maxGradientPublicationMaterializationCount() > 0);
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );
    }
}
