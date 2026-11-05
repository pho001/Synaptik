package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TrainingProgressStatsTest {
    @Test
    void completedPayloadAndStepsUseTheSameIntervalIncludingIdleTime() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        stats.step(12, 2, 4_000_000);
        clock.set(500_000_000);
        var first = stats.snapshot();
        assertEquals(1, first.steps());
        assertEquals(2, first.examples());
        assertEquals(4_000_000, first.bytes());
        assertEquals(2, first.stepsPerSecond());
        assertEquals(64, first.payloadMbitPerSecond());
        clock.set(1_000_000_000);
        var idle = stats.snapshot();
        assertEquals(0, idle.stepsPerSecond());
        assertEquals(0, idle.payloadMbitPerSecond());
        assertArrayEquals(new double[] {12}, idle.losses());
        stats.step(8, 3, 1_000_000);
        clock.set(2_000_000_000);
        var next = stats.snapshot();
        assertEquals(1, next.stepsPerSecond());
        assertEquals(8, next.payloadMbitPerSecond());
        assertEquals(5, next.examples());
        assertEquals(5_000_000, next.bytes());
        assertArrayEquals(new double[] {12, 8}, next.losses());
    }

    @Test
    void zeroTimeSamplesDoNotLoseSuccessfulPayloadOrInventInfiniteRates() {
        AtomicLong clock = new AtomicLong(10);
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        stats.step(0, 1, 12);
        var zero = stats.snapshot();
        assertTrue(Double.isNaN(zero.stepsPerSecond()));
        assertTrue(Double.isNaN(zero.payloadMbitPerSecond()));
        clock.addAndGet(1_000_000_000);
        var measured = stats.snapshot();
        assertEquals(1, measured.stepsPerSecond());
        assertEquals(0.000096, measured.payloadMbitPerSecond(), 1e-15);
        assertEquals(1_000_000_000, measured.elapsedNanos());
    }

    @Test
    void epochsHaveActualDurationsRatesAndChangingParameterMeans() {
        AtomicLong clock = new AtomicLong(1_000);
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        stats.step(10, 1, 100);
        clock.addAndGet(1_000_000_000);
        stats.step(2, 1, 300);
        stats.endEpoch();
        clock.addAndGet(9_000_000_000L);
        var ended = stats.snapshot();
        assertEquals(1_000_000_000, ended.epochNanos());
        assertEquals(10_000_000_000L, ended.elapsedNanos());
        var epoch = ended.recentEpochs()[0];
        assertEquals(1, epoch.number());
        assertEquals(2, epoch.steps());
        assertEquals(400, epoch.bytes());
        assertEquals(6, epoch.meanLoss());
        assertEquals(2, TrainingProgressStats.rate(epoch.steps(), epoch.nanos()));
        assertEquals(0.0032, TrainingProgressStats.mbitRate(epoch.bytes(), epoch.nanos()), 1e-15);
        stats.beginEpoch(2);
        clock.addAndGet(500_000_000);
        stats.step(-3, 2, 900);
        var current = stats.snapshot();
        assertEquals(500_000_000, current.epochNanos());
        assertEquals(1, current.epochSteps());
        assertEquals(-3, current.epochMean());
        assertEquals(3, current.steps());
    }

    @Test
    void successfulNonfiniteObjectivesRemainCountedAndSampledWithoutReplacement() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        double[] values = {0, -2, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN};
        for (double value : values) {
            stats.step(value, 2, 24);
            clock.incrementAndGet();
            var current = stats.snapshot();
            assertEquals(value, current.loss());
        }
        stats.endEpoch();
        var snapshot = stats.snapshot();
        assertEquals(5, snapshot.steps());
        assertEquals(10, snapshot.examples());
        assertEquals(120, snapshot.bytes());
        assertArrayEquals(values, snapshot.losses());
        assertTrue(Double.isNaN(snapshot.epochMean()));
        assertTrue(Double.isNaN(snapshot.recentEpochs()[0].meanLoss()));
    }

    @Test
    void boundedChronologicalRingsKeepOnlyActualSamplesAndRecentEpochs() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        int samples = TrainingProgressStats.LOSS_CAPACITY + 17;
        for (int epoch = 1; epoch <= samples; epoch++) {
            stats.beginEpoch(epoch);
            stats.step(epoch, 1, 4);
            clock.incrementAndGet();
            stats.endEpoch();
            stats.snapshot();
        }
        var snapshot = stats.snapshot();
        assertEquals(TrainingProgressStats.LOSS_CAPACITY, snapshot.losses().length);
        assertEquals(18, snapshot.losses()[0]);
        assertEquals(samples, snapshot.losses()[snapshot.losses().length - 1]);
        assertEquals(TrainingProgressStats.EPOCH_CAPACITY, snapshot.recentEpochs().length);
        assertEquals(samples - TrainingProgressStats.EPOCH_CAPACITY + 1,
                snapshot.recentEpochs()[0].number());
        assertEquals(samples, snapshot.recentEpochs()[snapshot.recentEpochs().length - 1].number());
        snapshot.losses()[0] = -999;
        assertEquals(18, stats.snapshot().losses()[0]);
    }

    @Test
    void finiteExtremeMeansDoNotOverflowAndNanoTimeWrapIsMeasured() {
        AtomicLong clock = new AtomicLong(Long.MAX_VALUE - 10);
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        stats.step(Double.MAX_VALUE, 1, 4);
        stats.step(-Double.MAX_VALUE, 1, 4);
        clock.addAndGet(30);
        var snapshot = stats.snapshot();
        assertEquals(30, snapshot.elapsedNanos());
        assertEquals(0, snapshot.epochMean());
        assertFalse(Double.isInfinite(snapshot.stepsPerSecond()));
    }

    @Test
    void invalidEventsDoNotPartiallyMutateSuccessfulCounters() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        assertThrows(IllegalStateException.class, () -> stats.step(1, 1, 4));
        assertThrows(IllegalStateException.class, () -> stats.beginEpoch(0));
        stats.beginEpoch(1);
        assertThrows(IllegalArgumentException.class, () -> stats.step(1, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> stats.step(1, 1, -1));
        stats.step(1, 1, Long.MAX_VALUE);
        assertThrows(ArithmeticException.class, () -> stats.step(2, 1, 1));
        var snapshot = stats.snapshot();
        assertEquals(1, snapshot.steps());
        assertEquals(1, snapshot.examples());
        assertEquals(Long.MAX_VALUE, snapshot.bytes());
        assertEquals(1, snapshot.loss());
        assertThrows(IllegalStateException.class, () -> stats.beginEpoch(2));
        stats.endEpoch();
        assertThrows(IllegalStateException.class, stats::endEpoch);
    }
}
