import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.explicit.ExplicitProfileMutators;
import tuning.calibration.runtime.PlatformRuntimeProfileGridCandidateSpace;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;
import tuning.workload.StandardWorkloads;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ProfileGridCandidateSpaceTest {
    @Test
    void profileGridGeneratesCartesianProfileVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "grid",
                "grid",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                List.of(
                        ExplicitProfileMutators.conv2dLoweringModes(List.of(
                                config.optimizer.Conv2dLoweringMode.HEURISTIC,
                                config.optimizer.Conv2dLoweringMode.OFF
                        )),
                        ExplicitProfileMutators.blasThreads(List.of(0, 1, 2))
                )
        );

        var candidates = space.generate(StandardWorkloads.conv2d(
                "conv",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        ));

        assertEquals(6, candidates.size());
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("conv2dLowering=OFF")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasThreads=AUTO")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasThreads=1")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasThreads=2")));
    }

    @Test
    void transformerHotPathMutatorsProduceAttentionAndBlasVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "transformer-grid",
                "transformer-grid",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                new config.runtime.RuntimeConfig(
                        config.backend.CpuKernelConfig.defaultsInference(),
                        config.runtime.ApproximationConfig.defaults(),
                        new config.runtime.BlasConfig(
                                backend.blas.BlasProvider.OPENBLAS_FFM,
                                2_000_000L,
                                true,
                                3.0d,
                                false,
                                0
                        )
                ),
                StandardWorkloads.transformerHotPathDefaults()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                ExplicitProfileMutators.transformerHotPathMutators()
        );

        var candidates = space.generate(StandardWorkloads.transformerHotPath("transformer_hot_path"));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("attentionMatMul=FORCE_ON")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasProvider=OPENBLAS_FFM")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("vectorThresholds=")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("fusedFallback=")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=openblas-native")));
    }

    @Test
    void cpuRuntimePolicyMutatorProducesNativeAndOpenBlasStorageVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "runtime-policy-grid",
                "runtime-policy-grid",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.cpuRuntimePolicyVariants())
        ).generate(StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=cpu-array")
                && c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_ARRAY
                && c.profile().runtime().blas().provider() == backend.blas.BlasProvider.NONE));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=cpu-native-require")
                && c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_NATIVE
                && c.profile().runtime().nativeCpuFailurePolicy() == config.runtime.NativeCpuFailurePolicy.REQUIRE_NATIVE));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=cpu-native-auto")
                && c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.AUTO
                && c.profile().runtime().nativeCpuFailurePolicy() == config.runtime.NativeCpuFailurePolicy.FALLBACK_TO_ARRAY));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=openblas-array-copy")
                && c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_ARRAY
                && c.profile().runtime().blas().provider() == backend.blas.BlasProvider.OPENBLAS_FFM
                && c.profile().runtime().blas().storageMode() == config.runtime.BlasStorageMode.CPU_ARRAY));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("runtime=openblas-native")
                && c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_NATIVE
                && c.profile().runtime().blas().provider() == backend.blas.BlasProvider.OPENBLAS_FFM
                && c.profile().runtime().blas().storageMode() == config.runtime.BlasStorageMode.CPU_NATIVE));
    }

    @Test
    void explicitOpenBlasRouteThreadMutatorGeneratesRouteSpecificVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "openblas-route-thread-grid",
                "openblas-route-thread-grid",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                new config.runtime.RuntimeConfig(
                        config.backend.CpuKernelConfig.defaultsInference(),
                        config.runtime.ApproximationConfig.defaults(),
                        new config.runtime.BlasConfig(
                                backend.blas.BlasProvider.OPENBLAS_FFM,
                                1_000_000L,
                                true,
                                3.0d,
                                true,
                                3.0d,
                                config.runtime.BlasStorageMode.AUTO,
                                false,
                                0,
                                0,
                                0
                        )),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.openBlasRouteThreads(List.of(0, 1), List.of(0, 4)))
        ).generate(StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertEquals(4, candidates.size());
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("openBlasRouteThreads=1/4")
                && c.profile().runtime().blas().openBlasArrayCopyThreads() == 1
                && c.profile().runtime().blas().openBlasNativeSegmentThreads() == 4));
    }

    @Test
    void blasAndKernelMutatorsPreserveNativeCpuRuntimePolicy() {
        ExecutionProfile base = new ExecutionProfile(
                "native-policy-preservation",
                "native-policy-preservation",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults()
                        .withCpuStorageProfile(config.runtime.CpuStorageProfile.CPU_NATIVE)
                        .withNativeCpuFailurePolicy(config.runtime.NativeCpuFailurePolicy.REQUIRE_NATIVE),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(
                        ExplicitProfileMutators.matmulBlasProviders(
                                List.of(backend.blas.BlasProvider.OPENBLAS_FFM),
                                List.of(1_000_000L)
                        ),
                        ExplicitProfileMutators.blasThreads(List.of(0)),
                        ExplicitProfileMutators.parallelThresholds(List.of(4_096), List.of(2_048), List.of(8_192))
                )
        ).generate(StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertTrue(candidates.stream().allMatch(c ->
                c.profile().runtime().cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_NATIVE));
        assertTrue(candidates.stream().allMatch(c ->
                c.profile().runtime().nativeCpuFailurePolicy() == config.runtime.NativeCpuFailurePolicy.REQUIRE_NATIVE));
    }

    @Test
    void mlpNormalizationAndLossMutatorsAlsoProduceFusedPolicyVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "policy-grid",
                "policy-grid",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var mlpCandidates = new ProfileGridCandidateSpace(base, ExplicitProfileMutators.mlpWorkloadMutators())
                .generate(StandardWorkloads.mlpClassification("mlp_test", 8, 16, 24, 12, 4, tensor.loss.LossReduction.MEAN));
        var normCandidates = new ProfileGridCandidateSpace(base, ExplicitProfileMutators.normalizationWorkloadMutators())
                .generate(StandardWorkloads.normalization(
                        "norm_test",
                        tuning.workload.NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
                        2, 16, 4, 1, 1e-5
                ));
        var lossCandidates = new ProfileGridCandidateSpace(base, ExplicitProfileMutators.lossWorkloadMutators())
                .generate(StandardWorkloads.indexedLoss(
                        "loss_test",
                        tuning.workload.LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES,
                        4, 8, tensor.loss.LossReduction.MEAN
                ));
        var genericCandidates = new ProfileGridCandidateSpace(base, ExplicitProfileMutators.genericWorkloadMutators())
                .generate(new tuning.workload.TensorRootWorkloadSpec(
                        "generic",
                        tuning.workload.WorkloadKind.GENERIC,
                        environment -> tensor.Tensor.scalar(1.0).add(tensor.Tensor.scalar(2.0))
                ));

        assertTrue(mlpCandidates.stream().anyMatch(c -> c.name().contains("fusedFallback=")));
        assertTrue(normCandidates.stream().anyMatch(c -> c.name().contains("fusedFallback=")));
        assertTrue(lossCandidates.stream().anyMatch(c -> c.name().contains("fusedFallback=")));
        assertTrue(genericCandidates.stream().anyMatch(c -> c.name().contains("fusedFallback=")));
    }

    @Test
    void advancedSchedulerPoliciesAreAvailableOnlyViaExplicitOptInMutator() {
        ExecutionProfile base = new ExecutionProfile(
                "advanced-scheduler",
                "advanced-scheduler",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.advancedSchedulerPolicies(
                        List.of(4, 6),
                        List.of(2),
                        List.of(1),
                        List.of(2048),
                        List.of(4096, 8192),
                        List.of(16384),
                        List.of(16384, 32768)
                ))
        ).generate(StandardWorkloads.transformerHotPath("transformer_hot_path"));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("scheduler=")));
        assertEquals(8, candidates.size());
    }

    @Test
    void matmulBlasShapeHeuristicsProduceExpectedVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "matmul-heuristics",
                "matmul-heuristics",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.matmulBlasShapeHeuristics(
                        List.of(true, false),
                        List.of(2.0, 4.0)
                ))
        ).generate(StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasShape=true:2.0")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasShape=false:4.0")));
    }

    @Test
    void parallelThresholdMutatorProducesExpectedVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "parallel-thresholds",
                "parallel-thresholds",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var candidates = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.parallelThresholds(
                        List.of(4_096, 16_384),
                        List.of(2_048),
                        List.of(8_192)
                ))
        ).generate(new tuning.workload.TensorRootWorkloadSpec(
                "generic",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0).add(tensor.Tensor.scalar(2.0))
        ));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("parallelThresholds=4096/2048/8192")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("parallelThresholds=16384/2048/8192")));
    }

    @Test
    void metalSelectionMutatorProducesVariantsAndSurvivesLaterRuntimeMutators() {
        ExecutionProfile base = new ExecutionProfile(
                "metal-selection",
                "metal-selection",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        PlatformRuntimeProfile runtimeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "platform",
                "hardware",
                "TEST",
                base
        );
        var candidates = new PlatformRuntimeProfileGridCandidateSpace(
                runtimeProfile,
                List.of(
                        PlatformRuntimeProfileMutators.metalSelectionPolicies(
                                List.of(true, false),
                                List.of(true),
                                List.of(0L, 1024L)
                        ),
                        PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000))
                )
        ).generate(StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertEquals(4, candidates.size());
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("metalSelection=true/true/1024")));
        assertTrue(candidates.stream().allMatch(c -> c.name().contains("matmulParallel=")));
        assertTrue(candidates.stream().anyMatch(c -> c.runtimeProfile().accelerator().metal().enabled()));
        assertTrue(candidates.stream().anyMatch(c -> !c.runtimeProfile().accelerator().metal().enabled()));
        assertTrue(candidates.stream().allMatch(c -> c.runtimeProfile().accelerator().metal().requireRuntimeAvailability()));
        assertTrue(candidates.stream().anyMatch(c -> c.runtimeProfile().accelerator().metal().minimumEstimatedWork() == 1024L));
    }

    @Test
    void matmulSpecificMutatorsDoNotExpandGenericWorkload() {
        ExecutionProfile base = new ExecutionProfile(
                "generic-grid",
                "generic-grid",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                List.of(
                        ExplicitProfileMutators.matmulBlasProviders(
                                List.of(backend.blas.BlasProvider.NONE, backend.blas.BlasProvider.OPENBLAS_FFM),
                                List.of(1_000_000L, 2_000_000L)
                        ),
                        ExplicitProfileMutators.attentionMatMulPolicies(List.of(
                                config.backend.AttentionMatMulPolicy.AUTO,
                                config.backend.AttentionMatMulPolicy.FORCE_ON
                        ))
                )
        );

        var candidates = space.generate(new tuning.workload.TensorRootWorkloadSpec(
                "generic",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0).add(tensor.Tensor.scalar(2.0))
        ));

        assertEquals(1, candidates.size());
    }

}
