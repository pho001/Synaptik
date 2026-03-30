import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.measure.BenchmarkScenarioSource;
import benchmark.measure.BroadcastScenarioSource;
import benchmark.measure.CandidateMeasurementCachePort;
import benchmark.measure.CandidateMeasurementHarness;
import benchmark.measure.CandidateMeasurementResult;
import benchmark.measure.MeasuredBenchmarkScenario;
import benchmark.measure.MeasuredBroadcastScenario;
import benchmark.measure.NanoClock;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

public class CandidateMeasurementHarnessTest {

    @Test
    void harnessMeasuresAllThreePathsAndAppliesScoutFusePrewarm() {
        FakeBenchmarkScenario forward = new FakeBenchmarkScenario(11);
        FakeBenchmarkScenario train = new FakeBenchmarkScenario(17);
        FakeBroadcastScenario broadcast = new FakeBroadcastScenario();

        AtomicInteger benchmarkCreates = new AtomicInteger();
        BenchmarkScenarioSource benchmarkSource = (a, b, c, candidate, requiresGrad, graphBlocks) -> {
            benchmarkCreates.incrementAndGet();
            return requiresGrad ? train : forward;
        };
        AtomicInteger broadcastCreates = new AtomicInteger();
        BroadcastScenarioSource broadcastSource = (a, b, c, candidate) -> {
            broadcastCreates.incrementAndGet();
            return broadcast;
        };

        CandidateMeasurementHarness harness = new CandidateMeasurementHarness(
                benchmarkSource,
                broadcastSource,
                6,
                8,
                new ScriptedClock(
                        0L, 20_000_000L,
                        20_000_000L, 56_000_000L,
                        56_000_000L, 68_000_000L
                )
        );

        OptimizerCandidate candidate = new OptimizerCandidate("FUSE_TEST", List.of(OptimizationStage.FUSE), TuningKnobs.trainingDefaults());
        CandidateMeasurementResult result = harness.measure(
                candidate,
                new double[]{1, 2, 3},
                new double[]{4, 5, 6},
                new double[]{7, 8, 9},
                new double[]{1, 2},
                new double[]{3, 4},
                new double[]{5, 6},
                1,
                2,
                "SCOUT",
                null
        );

        assertEquals(2, benchmarkCreates.get());
        assertEquals(1, broadcastCreates.get());
        assertEquals(11, result.graphInfSize());
        assertEquals(17, result.graphTrnSize());
        assertEquals(10.0, result.forwardMs(), 1e-12);
        assertEquals(18.0, result.trainMs(), 1e-12);
        assertEquals(6.0, result.broadcastMs(), 1e-12);
        assertEquals(11, forward.computeCalls.get(), "extraPrewarm + warmup + measure");
        assertEquals(11, train.computeCalls.get(), "extraPrewarm + warmup + measure");
        assertEquals(11, broadcast.computeCalls.get(), "extraPrewarm + warmup + measure");
        assertEquals(1, forward.setTrainingModeOffCalls.get());
        assertEquals(1, train.setTrainingModeOnCalls.get());
    }

    @Test
    void harnessReturnsCachedResultWithoutBuildingScenarios() {
        AtomicInteger benchmarkCreates = new AtomicInteger();
        AtomicInteger broadcastCreates = new AtomicInteger();
        BenchmarkScenarioSource benchmarkSource = (a, b, c, candidate, requiresGrad, graphBlocks) -> {
            benchmarkCreates.incrementAndGet();
            return new FakeBenchmarkScenario(1);
        };
        BroadcastScenarioSource broadcastSource = (a, b, c, candidate) -> {
            broadcastCreates.incrementAndGet();
            return new FakeBroadcastScenario();
        };
        CandidateMeasurementHarness harness = new CandidateMeasurementHarness(
                benchmarkSource,
                broadcastSource,
                6,
                8,
                new ScriptedClock(0L, 1L, 1L, 2L, 2L, 3L)
        );
        OptimizerCandidate candidate = new OptimizerCandidate("CACHE_TEST", List.of(), TuningKnobs.trainingDefaults());
        CandidateMeasurementResult cached = new CandidateMeasurementResult(candidate, 3, 5, 1.0, 2.0, 3.0);
        FakeCache cache = new FakeCache(cached);

        CandidateMeasurementResult result = harness.measure(
                candidate,
                new double[]{1},
                new double[]{2},
                new double[]{3},
                new double[]{4},
                new double[]{5},
                new double[]{6},
                1,
                2,
                "PRESCREEN",
                cache
        );

        assertSame(cached, result);
        assertEquals(0, benchmarkCreates.get());
        assertEquals(0, broadcastCreates.get());
        assertEquals(1, cache.getCalls.get());
        assertEquals(0, cache.putCalls.get());
    }

    private static final class FakeBenchmarkScenario implements MeasuredBenchmarkScenario {
        private final int graphSize;
        private final AtomicInteger computeCalls = new AtomicInteger();
        private final AtomicInteger setTrainingModeOnCalls = new AtomicInteger();
        private final AtomicInteger setTrainingModeOffCalls = new AtomicInteger();

        private FakeBenchmarkScenario(int graphSize) {
            this.graphSize = graphSize;
        }

        @Override
        public int graphSize() {
            return graphSize;
        }

        @Override
        public void setTrainingMode(boolean trainingMode) {
            if (trainingMode) {
                setTrainingModeOnCalls.incrementAndGet();
            } else {
                setTrainingModeOffCalls.incrementAndGet();
            }
        }

        @Override
        public void compute() {
            computeCalls.incrementAndGet();
        }
    }

    private static final class FakeBroadcastScenario implements MeasuredBroadcastScenario {
        private final AtomicInteger computeCalls = new AtomicInteger();

        @Override
        public void compute() {
            computeCalls.incrementAndGet();
        }
    }

    private static final class FakeCache implements CandidateMeasurementCachePort {
        private final CandidateMeasurementResult cached;
        private final AtomicInteger getCalls = new AtomicInteger();
        private final AtomicInteger putCalls = new AtomicInteger();

        private FakeCache(CandidateMeasurementResult cached) {
            this.cached = cached;
        }

        @Override
        public CandidateMeasurementResult get(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters) {
            getCalls.incrementAndGet();
            return cached;
        }

        @Override
        public void put(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters, CandidateMeasurementResult result) {
            putCalls.incrementAndGet();
        }
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
