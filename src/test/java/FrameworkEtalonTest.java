import org.junit.jupiter.api.Test;
import tuning.etalon.FrameworkEtalon;
import tuning.session.TuningPreset;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FrameworkEtalonTest {
    @Test
    void inferenceEtalonBuildsNonEmptySuite() {
        var request = FrameworkEtalon.inferenceSuite(TuningPreset.QUICK);
        assertFalse(request.workloads().isEmpty());
        assertFalse(request.entries().isEmpty());
        assertTrue(request.entries().stream().allMatch(c -> c.profile().mode() == backend.runtime.ExecutionMode.FORWARD));
    }

    @Test
    void trainingEtalonBuildsNonEmptySuite() {
        var request = FrameworkEtalon.trainingSuite(TuningPreset.QUICK);
        assertFalse(request.workloads().isEmpty());
        assertFalse(request.entries().isEmpty());
        assertTrue(request.entries().stream().allMatch(c -> c.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD));
    }
}
