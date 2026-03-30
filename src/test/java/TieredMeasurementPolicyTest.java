import benchmark.measure.MeasurementPolicy;
import benchmark.measure.MeasurementTier;
import benchmark.measure.TieredMeasurementPolicy;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TieredMeasurementPolicyTest {

    @Test
    void scoutFuseGetsExtraPrewarm() {
        MeasurementPolicy policy = TieredMeasurementPolicy.forTier(
                MeasurementTier.SCOUT,
                true,
                1,
                3,
                8
        );

        assertEquals(8, policy.extraPrewarmIters());
        assertEquals(1, policy.warmupIters());
        assertEquals(3, policy.measureIters());
    }

    @Test
    void nonScoutOrNonFuseDoesNotGetExtraPrewarm() {
        MeasurementPolicy scoutNoFuse = TieredMeasurementPolicy.forTier(
                MeasurementTier.SCOUT,
                false,
                1,
                3,
                8
        );
        MeasurementPolicy prescreenFuse = TieredMeasurementPolicy.forTier(
                MeasurementTier.PRESCREEN,
                true,
                1,
                3,
                8
        );

        assertEquals(0, scoutNoFuse.extraPrewarmIters());
        assertEquals(0, prescreenFuse.extraPrewarmIters());
    }
}
