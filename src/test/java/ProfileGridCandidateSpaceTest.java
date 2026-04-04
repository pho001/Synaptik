import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
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
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                List.of(
                        ProfileMutators.conv2dLoweringModes(List.of(
                                config.optimizer.Conv2dLoweringMode.HEURISTIC,
                                config.optimizer.Conv2dLoweringMode.OFF
                        )),
                        ProfileMutators.blasThreadPolicies(
                                List.of(backend.blas.BlasThreadPolicy.AUTO, backend.blas.BlasThreadPolicy.FIXED),
                                List.of(1, 2)
                        )
                )
        );

        var candidates = space.generate(StandardWorkloads.conv2d(
                "conv",
                1, 8, 8, 8, 8, 3, 3,
                tensor.Conv2dOptions.defaults().withPadding(1, 1),
                true
        ));

        assertEquals(6, candidates.size());
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("conv2dLowering=OFF")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasThread=FIXED:2")));
    }

    @Test
    void transformerHotPathMutatorsProduceAttentionAndBlasVariants() {
        ExecutionProfile base = new ExecutionProfile(
                "transformer-grid",
                "transformer-grid",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                new config.runtime.RuntimeConfig(
                        config.backend.CpuKernelConfig.defaultsInference(),
                        config.runtime.ApproximationConfig.defaults(),
                        new config.runtime.BlasConfig(
                                backend.blas.BlasProvider.OPENBLAS_FFM,
                                2_000_000L,
                                true,
                                3.0d,
                                false,
                                backend.blas.BlasThreadPolicy.AUTO,
                                0
                        )
                ),
                StandardWorkloads.transformerHotPathDefaults()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                ProfileMutators.transformerHotPathMutators()
        );

        var candidates = space.generate(StandardWorkloads.transformerHotPath("transformer_hot_path"));

        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("attentionMatMul=FORCE_ON")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("blasProvider=OPENBLAS_FFM")));
        assertTrue(candidates.stream().anyMatch(c -> c.name().contains("vectorPolicies=")));
    }

    @Test
    void matmulSpecificMutatorsDoNotExpandGenericWorkload() {
        ExecutionProfile base = new ExecutionProfile(
                "generic-grid",
                "generic-grid",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        ProfileGridCandidateSpace space = new ProfileGridCandidateSpace(
                base,
                List.of(
                        ProfileMutators.matmulBlasProviders(
                                List.of(backend.blas.BlasProvider.NONE, backend.blas.BlasProvider.OPENBLAS_FFM),
                                List.of(1_000_000L, 2_000_000L)
                        ),
                        ProfileMutators.attentionMatMulPolicies(List.of(
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
