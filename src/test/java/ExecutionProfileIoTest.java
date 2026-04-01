import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import benchmark.OptimizerProfileIO;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
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
                        List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM),
                        CseConfig.aggressiveDefaults(),
                        FuseConfig.inferenceDefaults()
                ),
                new RuntimeConfig(
                        new KernelTuningConfig(
                                new CpuKernelConfig(4, 32, 32, 32, 256, 50_000, 0, 2, 2_048, 16_384),
                                CudaKernelConfig.defaultsInference(),
                                OpenClKernelConfig.defaultsInference()
                        ),
                        new ApproximationConfig(ApproxMode.OFF, false),
                        new BlasConfig(BlasProvider.NONE, 2_000_000L, true, 3.0d, false)
                )
        );

        Path path = Files.createTempFile("execution-profile-", ".json");
        OptimizerProfileIO.saveExecutionProfile(path, expected);
        ExecutionProfile actual = OptimizerProfileIO.loadExecutionProfileOrDefault(path, ExecutionProfileIoTest.defaultProfile());

        assertEquals(expected.profileName(), actual.profileName());
        assertEquals(expected.candidateName(), actual.candidateName());
        assertEquals(expected.dataType(), actual.dataType());
        assertEquals(expected.mode(), actual.mode());
        assertEquals(expected.optimizer().stageOrder(), actual.optimizer().stageOrder());
        assertFalse(actual.optimizer().cse().strictSafety());
        assertEquals(expected.runtime().kernel().cpu().vectorMinSize(), actual.runtime().kernel().cpu().vectorMinSize());
        assertEquals(expected.runtime().kernel().cpu().contiguousMaterializeThreshold(), actual.runtime().kernel().cpu().contiguousMaterializeThreshold());
        assertEquals(expected.runtime().blas().provider(), actual.runtime().blas().provider());
        assertEquals(expected.runtime().approximation().approxMode(), actual.runtime().approximation().approxMode());
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
