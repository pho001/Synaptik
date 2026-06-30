import runtime.contract.ExecutionMode;
import backend.blas.BlasProvider;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tuning.calibration.runtime.PlatformRuntimeProfileGridCandidateSpace;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformRuntimeProfileMutatorsTest {
    @Test
    void openBlasRouteThreadMutatorGeneratesRouteSpecificRuntimeCandidates() {
        PlatformRuntimeProfile runtimeProfile = platformRuntimeProfileWithOpenBlasRouteThreads(0, 0);

        var candidates = new PlatformRuntimeProfileGridCandidateSpace(
                runtimeProfile,
                List.of(PlatformRuntimeProfileMutators.openBlasRouteThreads(
                        List.of(0, 1),
                        List.of(0, 4)
                ))
        ).generate(tuning.workload.StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertEquals(4, candidates.size());
        assertTrue(candidates.stream().anyMatch(candidate ->
                candidate.name().contains("openBlasRouteThreads=1/4")
                        && "1".equals(candidate.knobAssignments().get("runtime.blas.openBlasArrayCopyThreads"))
                        && "4".equals(candidate.knobAssignments().get("runtime.blas.openBlasNativeSegmentThreads"))
                        && candidate.runtimeProfile().matmul().openBlasArrayCopyThreads() == 1
                        && candidate.runtimeProfile().matmul().openBlasNativeSegmentThreads() == 4
        ));
    }

    @Test
    void matmulMutatorsPreserveRouteSpecificOpenBlasThreadValues() {
        PlatformRuntimeProfile runtimeProfile = platformRuntimeProfileWithOpenBlasRouteThreads(3, 5);

        var candidates = new PlatformRuntimeProfileGridCandidateSpace(
                runtimeProfile,
                List.of(
                        PlatformRuntimeProfileMutators.matmulBlasProviders(
                                List.of(BlasProvider.OPENBLAS_FFM),
                                List.of(1_000_000L)
                        ),
                        PlatformRuntimeProfileMutators.matmulShapeHeuristics(List.of(false), List.of(4.0d)),
                        PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000))
                )
        ).generate(tuning.workload.StandardWorkloads.matmul("matmul", 1, 64, 64, 64));

        assertTrue(candidates.stream().allMatch(candidate ->
                candidate.runtimeProfile().matmul().openBlasArrayCopyThreads() == 3
                        && candidate.runtimeProfile().matmul().openBlasNativeSegmentThreads() == 5
        ));
    }

    @Test
    void elementwiseDispatchMutatorAssignsNativeCheapThresholdForProfileDtype() {
        ExecutionProfile base = new ExecutionProfile(
                "elementwise-dispatch",
                "elementwise-dispatch",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults().withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE),
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
                List.of(PlatformRuntimeProfileMutators.elementwiseDispatchThresholds(
                        List.of(128, 256),
                        List.of(64),
                        List.of(8_192),
                        List.of(4_096)
                ))
        ).generate(new tuning.workload.TensorRootWorkloadSpec(
                "generic",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0).add(tensor.Tensor.scalar(2.0))
        ));

        assertEquals(2, candidates.size());
        assertTrue(candidates.stream().anyMatch(candidate ->
                candidate.name().contains("elementwiseDispatch=128/128/")
                        && "128".equals(candidate.knobAssignments().get("cpu.cheapVectorMinSize"))
                        && "128".equals(candidate.knobAssignments().get("cpu.nativeF32CheapVectorMinSize"))
                        && !candidate.knobAssignments().containsKey("cpu.nativeF64CheapVectorMinSize")
                        && candidate.runtimeProfile().elementwiseDispatch().nativeF32CheapVectorMinSize() == 128
                        && candidate.runtimeProfile().elementwiseDispatch().nativeF64CheapVectorMinSize()
                        == runtimeProfile.elementwiseDispatch().nativeF64CheapVectorMinSize()
        ));
    }

    @Test
    void elementwiseDispatchMutatorKeepsNativeCheapThresholdForArrayProfile() {
        ExecutionProfile base = new ExecutionProfile(
                "elementwise-dispatch-array",
                "elementwise-dispatch-array",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults().withCpuStorageProfile(CpuStorageProfile.CPU_ARRAY),
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
                List.of(PlatformRuntimeProfileMutators.elementwiseDispatchThresholds(
                        List.of(128),
                        List.of(64),
                        List.of(8_192),
                        List.of(4_096)
                ))
        ).generate(new tuning.workload.TensorRootWorkloadSpec(
                "generic",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0).add(tensor.Tensor.scalar(2.0))
        ));

        assertEquals(1, candidates.size());
        var candidate = candidates.getFirst();
        assertEquals("128", candidate.knobAssignments().get("cpu.cheapVectorMinSize"));
        assertTrue(!candidate.knobAssignments().containsKey("cpu.nativeF32CheapVectorMinSize"));
        assertEquals(
                runtimeProfile.elementwiseDispatch().nativeF32CheapVectorMinSize(),
                candidate.runtimeProfile().elementwiseDispatch().nativeF32CheapVectorMinSize()
        );
    }

    private static PlatformRuntimeProfile platformRuntimeProfileWithOpenBlasRouteThreads(
            int arrayCopyThreads,
            int nativeSegmentThreads
    ) {
        ExecutionProfile base = new ExecutionProfile(
                "route-threads",
                "route-threads",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                new RuntimeConfig(config.backend.CpuKernelConfig.defaultsInference(), config.runtime.ApproximationConfig.defaults(), new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1_000_000L,
                        true,
                        3.0d,
                        true,
                        3.0d,
                        config.runtime.BlasStorageMode.AUTO,
                        false,
                        0,
                        arrayCopyThreads,
                        nativeSegmentThreads
                )),
                WorkloadProfile.none()
        );
        return PlatformRuntimeProfile.fromExecutionProfile(
                "platform",
                "hardware",
                "TEST",
                base
        );
    }
}
