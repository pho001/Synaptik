import org.junit.jupiter.api.Test;
import tuning.calibration.progress.EtaEstimator;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CalibrationEtaEstimatorTest {
    @Test
    void formatsDurationsForProgressPanel() {
        assertEquals("00:00", EtaEstimator.format(Duration.ZERO));
        assertEquals("01:05", EtaEstimator.format(Duration.ofSeconds(65)));
        assertEquals("01:01:05", EtaEstimator.format(Duration.ofSeconds(3665)));
    }

    @Test
    void remainingIsZeroUntilProgressExists() {
        EtaEstimator estimator = new EtaEstimator();

        assertEquals(Duration.ZERO, estimator.remaining(0, 10));
        assertEquals(Duration.ZERO, estimator.remaining(10, 10));
        assertTrue(!estimator.elapsed().isNegative());
    }
}
