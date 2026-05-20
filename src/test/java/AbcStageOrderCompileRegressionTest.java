import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.workload.AbcSequenceMatmulWorkloadSpec;
import tuning.workload.WorkloadEnvironment;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

public class AbcStageOrderCompileRegressionTest {
    @Test
    void abcTrainingGraphCompilesAndRunsWithSimplificationFixpointForF64() {
        assertDoesNotThrow(() -> compileAndRun(DataType.FLOAT64));
    }

    @Test
    void abcTrainingGraphCompilesAndRunsWithSimplificationFixpointForF32() {
        assertDoesNotThrow(() -> compileAndRun(DataType.FLOAT32));
    }

    private static void compileAndRun(DataType dataType) {
        var profile = new config.profile.ExecutionProfile(
                "abc-stage-regression",
                "abc-stage-regression",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                RuntimeConfig.trainingDefaults(),
                config.profile.WorkloadProfile.none()
        );

        var workload = new AbcSequenceMatmulWorkloadSpec("abc_stage_regression", 32, 128);
        var instance = workload.instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(instance.root(), profile.compile());
        compiled.prepare(profile.runtime()).execute(profile.mode());
    }
}
