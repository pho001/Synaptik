import Benchmark.OptimizationStage;
import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;
import Benchmark.autotune.AutoTuneBestResults;
import Benchmark.autotune.AutoTuneFinalizationConfig;
import Benchmark.autotune.AutoTuneFinalizationResult;
import Benchmark.autotune.AutoTuneProfilePersistenceResult;
import Benchmark.autotune.AutoTuneResult;
import Benchmark.autotune.AutoTuneSessionConfig;
import Benchmark.autotune.AutoTuneSessionResult;
import Benchmark.autotune.AutoTuneSessionRunner;
import Benchmark.autotune.CandidatePerf;
import Benchmark.autotune.CoarseKnobSignature;
import Benchmark.autotune.CorrectnessVerdict;
import Benchmark.autotune.Phase1CandidateResult;
import Benchmark.autotune.Phase1Step;
import Benchmark.autotune.RefineConfig;
import Benchmark.autotune.RefineProgressUpdate;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

public class AutoTuneSessionRunnerTest {

    @Test
    void safetySweepStopsAfterPhase1AndSavesHistoryWhenStateful() throws IOException {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        List<Phase1Step> steps = new ArrayList<>();
        AtomicInteger historySaves = new AtomicInteger();
        AtomicInteger persistenceCalls = new AtomicInteger();

        AutoTuneSessionResult result = AutoTuneSessionRunner.run(
                List.of(a, b),
                new AutoTuneSessionConfig(
                        true,
                        false,
                        new AutoTuneFinalizationConfig(2, false, new RefineConfig(1, 1, 1))
                ),
                candidate -> candidate.name().equals("A")
                        ? new Phase1CandidateResult(
                                Phase1CandidateResult.Status.SAFE_SWEEP,
                                null,
                                new CorrectnessVerdict(true, 0.0),
                                null,
                                null
                        )
                        : new Phase1CandidateResult(
                                Phase1CandidateResult.Status.MISMATCH_SAFETY,
                                null,
                                new CorrectnessVerdict(false, 1.0),
                                null,
                                "MISMATCH_SAFETY"
                        ),
                steps::add,
                finalists -> finalists,
                (candidate, warmup, measure, tier, cache) -> {
                    throw new AssertionError("refine should not run during safety sweep");
                },
                prepared -> {
                    throw new AssertionError("finalists should not be prepared during safety sweep");
                },
                update -> {
                    throw new AssertionError("refine progress should not run during safety sweep");
                },
                (bestTraining, bestInference, validCount, mismatchCount) -> {
                    persistenceCalls.incrementAndGet();
                    return persistenceResult();
                },
                historySaves::incrementAndGet,
                new ScriptedClock(0L, 5_000_000L, 5_000_000L, 10_000_000L)
        );

        assertEquals(AutoTuneSessionResult.Status.SAFE_SWEEP_DONE, result.status());
        assertEquals(2, result.counters().processed());
        assertEquals(1, result.counters().safetySweepSafe());
        assertEquals(1, result.counters().mismatchSafety());
        assertEquals(2, steps.size());
        assertEquals(1, historySaves.get());
        assertEquals(0, persistenceCalls.get());
        assertNull(result.finalization());
        assertNull(result.persistenceResult());
    }

    @Test
    void noValidCandidateSkipsPersistenceAndHistorySave() throws IOException {
        OptimizerCandidate a = candidate("A");
        AtomicInteger historySaves = new AtomicInteger();
        AtomicInteger persistenceCalls = new AtomicInteger();

        AutoTuneSessionResult result = AutoTuneSessionRunner.run(
                List.of(a),
                new AutoTuneSessionConfig(
                        false,
                        false,
                        new AutoTuneFinalizationConfig(2, false, new RefineConfig(1, 1, 1))
                ),
                candidate -> new Phase1CandidateResult(
                        Phase1CandidateResult.Status.MISMATCH_FULL,
                        perf(candidate, 2.0, 3.0, 0.2),
                        new CorrectnessVerdict(true, 0.0),
                        new CorrectnessVerdict(false, 2.0),
                        "MISMATCH_FULL"
                ),
                step -> {},
                finalists -> finalists,
                (candidate, warmup, measure, tier, cache) -> {
                    throw new AssertionError("refine should not run without valid phase1 candidates");
                },
                prepared -> {
                    throw new AssertionError("finalists should not be prepared without valid phase1 candidates");
                },
                update -> {
                    throw new AssertionError("refine progress should not run without valid phase1 candidates");
                },
                (bestTraining, bestInference, validCount, mismatchCount) -> {
                    persistenceCalls.incrementAndGet();
                    return persistenceResult();
                },
                historySaves::incrementAndGet,
                System::nanoTime
        );

        assertEquals(AutoTuneSessionResult.Status.NO_VALID_CANDIDATE, result.status());
        assertEquals(AutoTuneFinalizationResult.Status.NO_VALID_CANDIDATE, result.finalization().status());
        assertEquals(0, historySaves.get());
        assertEquals(0, persistenceCalls.get());
        assertNull(result.bestResults());
    }

    @Test
    void emptyAfterPostcheckSavesHistoryAndSkipsPersistence() throws IOException {
        OptimizerCandidate a = candidate("A");
        AtomicInteger historySaves = new AtomicInteger();
        AtomicInteger persistenceCalls = new AtomicInteger();

        AutoTuneSessionResult result = AutoTuneSessionRunner.run(
                List.of(a),
                new AutoTuneSessionConfig(
                        false,
                        false,
                        new AutoTuneFinalizationConfig(2, true, new RefineConfig(1, 1, 1))
                ),
                candidate -> new Phase1CandidateResult(
                        Phase1CandidateResult.Status.VALID_PHASE1,
                        perf(candidate, 1.5, 2.5, 0.1),
                        new CorrectnessVerdict(true, 0.0),
                        new CorrectnessVerdict(true, 0.0),
                        null
                ),
                step -> {},
                finalists -> List.of(),
                (candidate, warmup, measure, tier, cache) -> {
                    throw new AssertionError("refine should not run when postcheck removes finalists");
                },
                prepared -> {},
                update -> {
                    throw new AssertionError("refine progress should not run when postcheck removes finalists");
                },
                (bestTraining, bestInference, validCount, mismatchCount) -> {
                    persistenceCalls.incrementAndGet();
                    return persistenceResult();
                },
                historySaves::incrementAndGet,
                System::nanoTime
        );

        assertEquals(AutoTuneSessionResult.Status.EMPTY_AFTER_POSTCHECK, result.status());
        assertEquals(AutoTuneFinalizationResult.Status.EMPTY_AFTER_POSTCHECK, result.finalization().status());
        assertEquals(1, historySaves.get());
        assertEquals(0, persistenceCalls.get());
        assertNull(result.bestResults());
    }

    @Test
    void successfulSessionPersistsBestResultsAndSavesHistory() throws IOException {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        OptimizerCandidate c = candidate("C");
        AtomicInteger historySaves = new AtomicInteger();
        List<RefineProgressUpdate> updates = new ArrayList<>();
        AtomicReference<AutoTuneResult> persistedTraining = new AtomicReference<>();
        AtomicReference<AutoTuneResult> persistedInference = new AtomicReference<>();
        AtomicReference<int[]> persistedCounts = new AtomicReference<>();
        AutoTuneProfilePersistenceResult persistenceResult = persistenceResult();

        AutoTuneSessionResult result = AutoTuneSessionRunner.run(
                List.of(a, b, c),
                new AutoTuneSessionConfig(
                        false,
                        false,
                        new AutoTuneFinalizationConfig(2, false, new RefineConfig(1, 5, 7))
                ),
                candidate -> switch (candidate.name()) {
                    case "A" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.VALID_PHASE1,
                            perf(candidate, 3.0, 5.0, 0.2),
                            new CorrectnessVerdict(true, 0.0),
                            new CorrectnessVerdict(true, 0.0),
                            null
                    );
                    case "B" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.VALID_PHASE1,
                            perf(candidate, 4.0, 2.0, 0.2),
                            new CorrectnessVerdict(true, 0.0),
                            new CorrectnessVerdict(true, 0.0),
                            null
                    );
                    default -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.MISMATCH_FULL,
                            perf(candidate, 6.0, 6.0, 0.2),
                            new CorrectnessVerdict(true, 0.0),
                            new CorrectnessVerdict(false, 1.0),
                            "MISMATCH_FULL"
                    );
                },
                step -> {},
                finalists -> finalists,
                (candidate, warmup, measure, tier, cache) -> {
                    assertEquals(5, warmup);
                    assertEquals(7, measure);
                    assertEquals("REFINE", tier);
                    return candidate.name().equals("A")
                            ? perf(candidate, 0.5, 4.0, 0.1)
                            : perf(candidate, 2.0, 0.5, 0.1);
                },
                prepared -> {},
                updates::add,
                (bestTraining, bestInference, validCount, mismatchCount) -> {
                    persistedTraining.set(bestTraining);
                    persistedInference.set(bestInference);
                    persistedCounts.set(new int[]{validCount, mismatchCount});
                    return persistenceResult;
                },
                historySaves::incrementAndGet,
                () -> 0L
        );

        assertEquals(AutoTuneSessionResult.Status.DONE, result.status());
        assertNotNull(result.finalization());
        assertEquals(AutoTuneFinalizationResult.Status.OK, result.finalization().status());
        AutoTuneBestResults best = result.bestResults();
        assertNotNull(best);
        assertEquals("B", best.training().candidate().name());
        assertEquals("A", best.inference().candidate().name());
        assertSame(best.training(), persistedTraining.get());
        assertSame(best.inference(), persistedInference.get());
        assertEquals(2, persistedCounts.get()[0]);
        assertEquals(1, persistedCounts.get()[1]);
        assertSame(persistenceResult, result.persistenceResult());
        assertEquals(1, historySaves.get());
        assertEquals(2, updates.size());
    }

    private static AutoTuneProfilePersistenceResult persistenceResult() {
        return new AutoTuneProfilePersistenceResult(9.0, 8.0, true, false, true, false);
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, double forward, double train, double broadcast) {
        return new CandidatePerf(
                candidate,
                "CSE",
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forward,
                train,
                broadcast
        );
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }

    private static final class ScriptedClock implements java.util.function.LongSupplier {
        private final long[] values;
        private int index;

        private ScriptedClock(long... values) {
            this.values = values.clone();
        }

        @Override
        public long getAsLong() {
            if (index >= values.length) {
                return values[values.length - 1];
            }
            return values[index++];
        }
    }
}
