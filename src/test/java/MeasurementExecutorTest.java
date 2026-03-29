import Benchmark.measure.MeasurementExecutor;
import Benchmark.measure.MeasurementPolicy;
import Benchmark.measure.NanoClock;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class MeasurementExecutorTest {

    @Test
    void measureAverageMsRunsExtraPrewarmWarmupAndMeasureIterations() {
        AtomicInteger calls = new AtomicInteger();
        MeasurementPolicy policy = new MeasurementPolicy(2, 3, 4);
        NanoClock clock = new ScriptedClock(0L, 8_000_000L);

        double avgMs = MeasurementExecutor.measureAverageMs(policy, calls::incrementAndGet, clock);

        assertEquals(9, calls.get(), "extra prewarm + warmup + measure calls");
        assertEquals(2.0, avgMs, 1e-12, "8 ms / 4 iterations");
    }

    @Test
    void policyRejectsInvalidMeasureIterations() {
        assertThrows(IllegalArgumentException.class, () -> new MeasurementPolicy(0, 0, 0));
    }

    private static final class ScriptedClock implements NanoClock {
        private final long[] values;
        private int index;

        private ScriptedClock(long... values) {
            this.values = values.clone();
        }

        @Override
        public long nanoTime() {
            if (index >= values.length) {
                return values[values.length - 1];
            }
            return values[index++];
        }
    }
}
