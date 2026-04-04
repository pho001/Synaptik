import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
import backend.runtime.ExecutionMode;
import benchmark.OptimizerProfileIO;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.AttentionMatMulPolicy;
import config.optimizer.CseConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.FuseConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.optimizer.RewriteConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
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
                        new RewriteConfig(new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS)),
                        CseConfig.aggressiveDefaults(),
                        FuseConfig.inferenceDefaults(),
                        new MemoryConfig(false, false, true, 8)
                ),
                new RuntimeConfig(
                        new KernelTuningConfig(
                                new CpuKernelConfig(4, 32, 32, 32, 256, 50_000, 0, 2, 2_048, 16_384,
                                        config.backend.SumAccuracyMode.FAST, 2.0d,
                                        config.backend.VectorPolicy.AUTO, config.backend.VectorPolicy.AUTO, config.backend.VectorPolicy.AUTO,
                                        2_000_000, AttentionMatMulPolicy.FORCE_ON),
                                CudaKernelConfig.defaultsInference(),
                                OpenClKernelConfig.defaultsInference()
                        ),
                        new ApproximationConfig(ApproxMode.OFF, false),
                        new BlasConfig(BlasProvider.NONE, 2_000_000L, true, 3.0d, false, BlasThreadPolicy.FIXED, 2)
                ),
                new WorkloadProfile(config.profile.WorkloadKind.TRANSFORMER_HOT_PATH, 4, 8, 64, 32, 32, 256, true)
        );

        Path path = Files.createTempFile("execution-profile-", ".json");
        OptimizerProfileIO.saveExecutionProfile(path, expected);
        ExecutionProfile actual = OptimizerProfileIO.loadExecutionProfileOrDefault(path, ExecutionProfileIoTest.defaultProfile());

        assertEquals(expected.profileName(), actual.profileName());
        assertEquals(expected.candidateName(), actual.candidateName());
        assertEquals(expected.dataType(), actual.dataType());
        assertEquals(expected.mode(), actual.mode());
        assertEquals(expected.optimizer().stageOrder(), actual.optimizer().stageOrder());
        assertEquals(expected.optimizer().rewrite(), actual.optimizer().rewrite());
        assertFalse(actual.optimizer().cse().strictSafety());
        assertEquals(expected.optimizer().memory(), actual.optimizer().memory());
        assertEquals(expected.runtime().kernel().cpu().vectorMinSize(), actual.runtime().kernel().cpu().vectorMinSize());
        assertEquals(expected.runtime().kernel().cpu().contiguousMaterializeThreshold(), actual.runtime().kernel().cpu().contiguousMaterializeThreshold());
        assertEquals(expected.runtime().kernel().cpu().attentionMatMulPolicy(), actual.runtime().kernel().cpu().attentionMatMulPolicy());
        assertEquals(expected.runtime().blas().provider(), actual.runtime().blas().provider());
        assertEquals(expected.runtime().blas().threadPolicy(), actual.runtime().blas().threadPolicy());
        assertEquals(expected.runtime().blas().threads(), actual.runtime().blas().threads());
        assertEquals(expected.runtime().approximation().approxMode(), actual.runtime().approximation().approxMode());
        assertEquals(expected.workload(), actual.workload());
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
