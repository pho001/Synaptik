import backend.runtime.ExecutionMode;
import benchmark.scenario.PreparedHotPathScenario;
import benchmark.scenario.TransformerHotPathScenarioFactory;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransformerHotPathScenarioFactoryTest {
    @Test
    void createsAndExecutesTransformerHotPathScenarios() {
        ExecutionProfile profile = new ExecutionProfile(
                "transformer-hotpath-test",
                "transformer-hotpath-test",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.inferenceDefaults(),
                new WorkloadProfile(config.profile.WorkloadKind.TRANSFORMER_HOT_PATH, 2, 2, 8, 4, 4, 16, true)
        );

        List<PreparedHotPathScenario> scenarios = TransformerHotPathScenarioFactory.create(profile);
        assertEquals(8, scenarios.size());

        for (PreparedHotPathScenario scenario : scenarios) {
            scenario.run();
            assertTrue(Double.isFinite(scenario.sink()), "scenario " + scenario.name() + " should produce finite scalar sink");
        }
    }
}
