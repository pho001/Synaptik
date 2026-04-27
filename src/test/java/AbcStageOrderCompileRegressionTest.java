import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.workload.AbcSequenceMatmulWorkloadSpec;
import tuning.workload.WorkloadEnvironment;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

public class AbcStageOrderCompileRegressionTest {
    @Test
    void abcTrainingGraphCompilesAndRunsWithCseBeforeArFuseMemForF64() {
        assertDoesNotThrow(() -> compileAndRun(DataType.FLOAT64));
    }

    @Test
    void abcTrainingGraphCompilesAndRunsWithCseBeforeArFuseMemForF32() {
        assertDoesNotThrow(() -> compileAndRun(DataType.FLOAT32));
    }

    private static void compileAndRun(DataType dataType) {
        var profile = new config.profile.ExecutionProfile(
                "abc-stage-regression",
                "abc-stage-regression",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults().withStageOrder(List.of(
                        OptimizerStage.CSE,
                        OptimizerStage.AR,
                        OptimizerStage.PART,
                        OptimizerStage.FUSE,
                        OptimizerStage.MEM
                )),
                RuntimeConfig.trainingDefaults(),
                config.profile.WorkloadProfile.none()
        );

        var workload = new AbcSequenceMatmulWorkloadSpec("abc_stage_regression", 32, 128);
        var instance = workload.instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(instance.root(), profile.optimizer());
        compiled.prepare(profile.runtime()).execute(profile.mode());
    }
}
