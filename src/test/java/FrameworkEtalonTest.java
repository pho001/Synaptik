import org.junit.jupiter.api.Test;
import tuning.etalon.FrameworkEtalon;
import tuning.preset.TuningPreset;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FrameworkEtalonTest {
    @Test
    void inferenceEtalonBuildsNonEmptySuite() {
        var request = FrameworkEtalon.inferenceSuite(TuningPreset.QUICK);
        assertFalse(request.workloads().isEmpty());
        assertFalse(request.entries().isEmpty());
        assertTrue(request.entries().stream().allMatch(c -> c.profile().mode() == runtime.contract.ExecutionMode.FORWARD));
    }

    @Test
    void trainingEtalonBuildsNonEmptySuite() {
        var request = FrameworkEtalon.trainingSuite(TuningPreset.QUICK);
        assertFalse(request.workloads().isEmpty());
        assertFalse(request.entries().isEmpty());
        assertTrue(request.entries().stream().allMatch(c -> c.profile().mode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD));
    }

    @Test
    void inferenceRegressionEtalonUsesMoreConservativeMeasurement() {
        var request = FrameworkEtalon.inferenceRegressionSuite();
        assertFalse(request.workloads().isEmpty());
        assertFalse(request.entries().isEmpty());
        assertEquals(8, request.measurement().warmupIters());
        assertEquals(8, request.measurement().measureIters());
        assertEquals(3, request.measurement().repeats());
    }
}
