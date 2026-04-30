import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.optimizer.AlgebraicRewriteConfig;
import config.optimizer.AcceleratorRegionPolicy;
import config.optimizer.CseConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.CpuFusionCheapProducerPolicy;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuFusionFanoutPolicy;
import config.optimizer.CpuFusionLayoutPolicy;
import config.optimizer.CpuFusionMode;
import config.optimizer.CpuRegionBoundaryPolicy;
import config.optimizer.CpuRegionConfig;
import config.optimizer.CpuRegionFanoutPolicy;
import config.optimizer.CpuRegionPolicy;
import config.optimizer.FuseConfig;
import config.optimizer.LinearLoweringConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.MetalTransferModel;
import config.optimizer.OffloadConfig;
import config.optimizer.OffloadPolicy;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.optimizer.PartitionConfig;
import config.optimizer.PiecewiseLoweringConfig;
import config.optimizer.RewriteConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileIO;
import config.profile.WorkloadProfile;
import config.runtime.ApproximationConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class ExecutionProfileIoTest {
    @Test
    void executionProfileRoundTripsStageOrderAndRuntimeConfig() throws Exception {
        ExecutionProfile expected = new ExecutionProfile(
                "profile-roundtrip",
                "AUTO_TEST",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                new OptimizerConfig(
                        List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
                        new RewriteConfig(
                                new AlgebraicRewriteConfig(true),
                                new LinearLoweringConfig(false),
                                new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS),
                                new PiecewiseLoweringConfig(true, true, true)
                        ),
                        CseConfig.aggressiveDefaults(),
                        FuseConfig.inferenceDefaults(),
                        new MemoryConfig(false, false, true, 8),
                        new PartitionConfig(
                                9,
                                77,
                                11.0,
                                22.0,
                                33.0,
                                44.0,
                                55.0,
                                66.0,
                                graph.optimizer.partition.PartitionPlannerStrategy.SCORED_CANDIDATE_SEARCH,
                                graph.optimizer.partition.PartitionTarget.GPU_METAL,
                                MetalTransferModel.MEASURED
                        ),
                        new OffloadConfig(
                                OffloadPolicy.ACCELERATOR_IF_PROFITABLE,
                                AcceleratorRegionPolicy.SCORED_PROFITABLE_REGIONS
                        ),
                        new CpuRegionConfig(
                                CpuRegionPolicy.AGGRESSIVE_CPU_REGIONS,
                                123,
                                CpuRegionFanoutPolicy.INCLUDE_AND_SPLIT_EXECUTION_UNITS,
                                CpuRegionBoundaryPolicy.INCLUDE_SAFE_LAYOUT_PASSTHROUGH
                        ),
                        new CpuFusionConfig(
                                CpuFusionMode.LOCAL_AGGRESSIVE,
                                31,
                                CpuFusionFanoutPolicy.MATERIALIZE_AND_CONTINUE,
                                CpuFusionLayoutPolicy.ALIAS_VIEW_PASSTHROUGH,
                                CpuFusionCheapProducerPolicy.INLINE_CHEAP_SHARED
                        )
                ),
                new RuntimeConfig(
                        new KernelTuningConfig(
                                new CpuKernelConfig(
                                        4, 32, 32, 32,
                                        256, 256, 256, 256, 256,
                                        50_000, 50_000, 50_000, 50_000, 50_000,
                                        16_384,
                                        5, 3, 2,
                                        2_048, 4_096, 8_192, 32_768,
                                        1, 1, 1, 1,
                                        config.backend.SumAccuracyMode.FAST,
                                        2_000_000, AttentionMatMulPolicy.FORCE_ON, CpuMatMulMicroKernel.F32_4X2),
                                CudaKernelConfig.defaultsInference(),
                                OpenClKernelConfig.defaultsInference()
                        ),
                        new ApproximationConfig(ApproxMode.OFF, false),
                        new BlasConfig(BlasProvider.NONE, 2_000_000L, true, 3.0d, false, 12.0d, false, 0),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true),
                        new AcceleratorConfig(
                                new AcceleratorBackendConfig(
                                        false,
                                        true,
                                        111L,
                                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, false, 1_111L)
                                ),
                                new AcceleratorBackendConfig(
                                        true,
                                        false,
                                        222L,
                                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.AUTO, true, 2_222L)
                                ),
                                new AcceleratorBackendConfig(
                                        true,
                                        true,
                                        333L,
                                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, false, 3_333L)
                                )
                        )
                ),
                new WorkloadProfile(config.profile.WorkloadKind.TRANSFORMER_HOT_PATH, 4, 8, 64, 32, 32, 256, true)
        );

        Path path = Files.createTempFile("execution-profile-", ".json");
        ExecutionProfileIO.saveExecutionProfile(path, expected);
        ExecutionProfile actual = ExecutionProfileIO.loadExecutionProfileOrDefault(path, ExecutionProfileIoTest.defaultProfile());

        assertEquals(expected.profileName(), actual.profileName());
        assertEquals(expected.candidateName(), actual.candidateName());
        assertEquals(expected.dataType(), actual.dataType());
        assertEquals(expected.mode(), actual.mode());
        assertEquals(expected.optimizer().stageOrder(), actual.optimizer().stageOrder());
        assertEquals(expected.optimizer().rewrite(), actual.optimizer().rewrite());
        assertFalse(actual.optimizer().cse().strictSafety());
        assertEquals(expected.optimizer().memory(), actual.optimizer().memory());
        assertEquals(expected.optimizer().partition(), actual.optimizer().partition());
        assertEquals(expected.optimizer().offload(), actual.optimizer().offload());
        assertEquals(expected.optimizer().cpuRegion(), actual.optimizer().cpuRegion());
        assertEquals(expected.optimizer().cpuFusion(), actual.optimizer().cpuFusion());
        assertEquals(expected.runtime().kernel().cpu().cheapVectorMinSize(), actual.runtime().kernel().cpu().cheapVectorMinSize());
        assertEquals(expected.runtime().kernel().cpu().transcendentalVectorMinSize(), actual.runtime().kernel().cpu().transcendentalVectorMinSize());
        assertEquals(expected.runtime().kernel().cpu().reductionVectorMinSize(), actual.runtime().kernel().cpu().reductionVectorMinSize());
        assertEquals(expected.runtime().kernel().cpu().contiguousMaterializeThreshold(), actual.runtime().kernel().cpu().contiguousMaterializeThreshold());
        assertEquals(expected.runtime().kernel().cpu().lowCostTargetChunksPerWorker(), actual.runtime().kernel().cpu().lowCostTargetChunksPerWorker());
        assertEquals(expected.runtime().kernel().cpu().mediumCostTargetChunksPerWorker(), actual.runtime().kernel().cpu().mediumCostTargetChunksPerWorker());
        assertEquals(expected.runtime().kernel().cpu().highCostTargetChunksPerWorker(), actual.runtime().kernel().cpu().highCostTargetChunksPerWorker());
        assertEquals(expected.runtime().kernel().cpu().minScalarChunkSize(), actual.runtime().kernel().cpu().minScalarChunkSize());
        assertEquals(expected.runtime().kernel().cpu().minVectorChunkSize(), actual.runtime().kernel().cpu().minVectorChunkSize());
        assertEquals(expected.runtime().kernel().cpu().minReductionChunkSize(), actual.runtime().kernel().cpu().minReductionChunkSize());
        assertEquals(expected.runtime().kernel().cpu().commonPoolLowCostMaxWorkPerWorker(), actual.runtime().kernel().cpu().commonPoolLowCostMaxWorkPerWorker());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulPolicy(), actual.runtime().kernel().cpu().attentionMatMulPolicy());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulTileM(), actual.runtime().kernel().cpu().attentionMatMulTileM());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulTileN(), actual.runtime().kernel().cpu().attentionMatMulTileN());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulTileK(), actual.runtime().kernel().cpu().attentionMatMulTileK());
        assertEquals(expected.runtime().kernel().cpu().matMulMicroKernel(), actual.runtime().kernel().cpu().matMulMicroKernel());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulMicroKernel(), actual.runtime().kernel().cpu().attentionMatMulMicroKernel());
        assertEquals(expected.runtime().blas().provider(), actual.runtime().blas().provider());
        assertEquals(expected.runtime().blas().threads(), actual.runtime().blas().threads());
        assertEquals(expected.runtime().blas().f32RequireMgeK(), actual.runtime().blas().f32RequireMgeK());
        assertEquals(expected.runtime().blas().f32MaxNOverK(), actual.runtime().blas().f32MaxNOverK());
        assertEquals(expected.runtime().blas().f32WideRequireMgeK(), actual.runtime().blas().f32WideRequireMgeK());
        assertEquals(expected.runtime().blas().f32WideMaxNOverK(), actual.runtime().blas().f32WideMaxNOverK());
        assertEquals(expected.runtime().approximation().approxMode(), actual.runtime().approximation().approxMode());
        assertEquals(expected.runtime().fused(), actual.runtime().fused());
        assertEquals(expected.runtime().accelerator(), actual.runtime().accelerator());
        assertEquals(expected.workload(), actual.workload());
    }

    @Test
    void legacyExecutionProfilesWithoutAcceleratorBufferKeysUseDefaults() {
        ExecutionProfile fallback = defaultProfile();
        String json = """
                {
                  "dataType": "FLOAT32",
                  "mode": "FORWARD",
                  "profileName": "legacy",
                  "candidateName": "legacy",
                  "runtime": {
                    "accelerator": {
                      "metalEnabled": true,
                      "metalRequireRuntimeAvailability": true,
                      "metalMinimumEstimatedWork": 123
                    }
                  }
                }
                """;

        ExecutionProfile actual = ExecutionProfileIO.fromJsonOrDefault(json, fallback);

        assertEquals(123L, actual.runtime().accelerator().metal().minimumEstimatedWork());
        assertEquals(AcceleratorBufferBindingMode.AUTO, actual.runtime().accelerator().metal().buffer().bindingMode());
        assertEquals(0L, actual.runtime().accelerator().metal().buffer().minimumEstimatedWork());
        assertEquals(true, actual.runtime().accelerator().metal().buffer().allowPreparedInputMaterialization());
    }

    @Test
    void legacyExecutionProfileInvalidBufferModeStillUsesDefaultUntilStrictExecutionProfileIoExists() {
        ExecutionProfile fallback = defaultProfile();
        String json = """
                {
                  "dataType": "FLOAT32",
                  "mode": "FORWARD",
                  "profileName": "legacy",
                  "candidateName": "legacy",
                  "runtime": {
                    "accelerator": {
                      "metalBufferBindingMode": "BROKEN"
                    }
                  }
                }
                """;

        ExecutionProfile actual = ExecutionProfileIO.fromJsonOrDefault(json, fallback);

        assertEquals(
                fallback.runtime().accelerator().metal().buffer().bindingMode(),
                actual.runtime().accelerator().metal().buffer().bindingMode()
        );
    }

    @Test
    void oldProfilesFallbackAttentionMatmulMicroKernelToGenericMatmulMicroKernel() {
        ExecutionProfile fallback = defaultProfile();
        String json = """
                {
                  "dataType": "FLOAT32",
                  "mode": "FORWARD_BACKWARD",
                  "profileName": "legacy",
                  "candidateName": "legacy",
                  "runtime": {
                    "kernel": {
                      "cpu": {
                        "cpuMatMulMicroKernel": "F32_4X2"
                      }
                    }
                  }
                }
                """;

        ExecutionProfile actual = ExecutionProfileIO.fromJsonOrDefault(json, fallback);

        assertEquals(CpuMatMulMicroKernel.F32_4X2, actual.runtime().kernel().cpu().matMulMicroKernel());
        assertEquals(CpuMatMulMicroKernel.F32_4X2, actual.runtime().kernel().cpu().attentionMatMulMicroKernel());
        assertEquals(actual.runtime().kernel().cpu().matMulTileM(), actual.runtime().kernel().cpu().attentionMatMulTileM());
        assertEquals(actual.runtime().kernel().cpu().matMulTileN(), actual.runtime().kernel().cpu().attentionMatMulTileN());
        assertEquals(actual.runtime().kernel().cpu().matMulTileK(), actual.runtime().kernel().cpu().attentionMatMulTileK());
    }

    private static ExecutionProfile defaultProfile() {
        return new ExecutionProfile(
                "default",
                "default",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                RuntimeConfig.trainingDefaults()
        );
    }
}
