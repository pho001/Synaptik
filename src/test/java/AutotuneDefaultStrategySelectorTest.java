import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.ListCandidateSpace;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.session.AutotuneDefaultStrategySelector;
import tuning.session.AutotuneRequest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AutotuneDefaultStrategySelectorTest {
    @Test
    void selectorUsesExhaustiveForSmallNonRefinableSpace() {
        var request = new AutotuneRequest(
                new tuning.workload.TensorRootWorkloadSpec(
                        "small",
                        tuning.workload.WorkloadKind.GENERIC,
                        environment -> tensor.Tensor.scalar(1.0)
                ),
                new ListCandidateSpace(List.of(candidate("a"), candidate("b"))),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(8, 2, 3, false),
                tuning.store.PersistencePolicy.disabled()
        );

        assertEquals("ExhaustiveSearchStrategy", AutotuneDefaultStrategySelector.select(request).getClass().getSimpleName());
    }

    @Test
    void selectorUsesTreeBeamForModerateRefinableSpace() {
        var base = profile("base");
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ProfileMutators.conv2dLoweringModes(List.of(
                        config.optimizer.Conv2dLoweringMode.HEURISTIC,
                        config.optimizer.Conv2dLoweringMode.OFF,
                        config.optimizer.Conv2dLoweringMode.ALWAYS
                )))
        );
        var request = new AutotuneRequest(
                tuning.workload.StandardWorkloads.conv2d(
                        "conv",
                        1, 8, 8, 8, 8, 3, 3,
                        tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                        true
                ),
                space,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(8, 2, 3, false),
                tuning.store.PersistencePolicy.disabled()
        );

        assertEquals("TreeBeamSearchStrategy", AutotuneDefaultStrategySelector.select(request).getClass().getSimpleName());
    }

    @Test
    void selectorUsesBranchAndBoundForLargerRefinableSpace() {
        var base = new ExecutionProfile(
                "transformer",
                "transformer",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                new config.runtime.RuntimeConfig(
                        new config.backend.CpuKernelConfig(4, 32, 32, 32, 256, 50_000, 1_000_000_000),
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
                tuning.workload.StandardWorkloads.transformerHotPathDefaults()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                tuning.candidate.ProfileMutators.transformerHotPathMutators()
        );
        var request = new AutotuneRequest(
                tuning.workload.StandardWorkloads.transformerHotPath("transformer_hot_path"),
                space,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(32, 2, 4, false),
                tuning.store.PersistencePolicy.disabled()
        );

        assertEquals("BranchAndBoundSearchStrategy", AutotuneDefaultStrategySelector.select(request).getClass().getSimpleName());
    }

    private static Candidate candidate(String name) {
        return new Candidate(name, profile(name));
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
