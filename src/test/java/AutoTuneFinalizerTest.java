import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneBestResults;
import benchmark.autotune.AutoTuneFinalizationConfig;
import benchmark.autotune.AutoTuneFinalizationResult;
import benchmark.autotune.AutoTuneFinalizer;
import benchmark.autotune.AutoTuneResult;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.RefineProgressUpdate;
import benchmark.autotune.RefineConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

public class AutoTuneFinalizerTest {

    @Test
    void returnsNoValidCandidateWhenPhase1IsEmpty() {
        AtomicInteger preparedCalls = new AtomicInteger();
        AtomicInteger refineCalls = new AtomicInteger();

        AutoTuneFinalizationResult result = AutoTuneFinalizer.finalizePhase1(
                List.of(),
                new AutoTuneFinalizationConfig(2, false, new RefineConfig(1, 1, 1)),
                finalists -> finalists,
                (candidate, warmup, measure, tier, cache) -> {
                    throw new AssertionError("refine should not run");
                },
                prepared -> preparedCalls.incrementAndGet(),
                update -> refineCalls.incrementAndGet(),
                System::nanoTime
        );

        assertEquals(AutoTuneFinalizationResult.Status.NO_VALID_CANDIDATE, result.status());
        assertEquals(0, preparedCalls.get());
        assertEquals(0, refineCalls.get());
        assertNull(result.bestResults());
    }

    @Test
    void returnsEmptyAfterPostcheckWhenAllFinalistsAreDropped() {
        AutoTuneResult a = phase1("A", 1.0, 4.0);

        AtomicInteger preparedCalls = new AtomicInteger();
        AutoTuneFinalizationResult result = AutoTuneFinalizer.finalizePhase1(
                List.of(a),
                new AutoTuneFinalizationConfig(1, true, new RefineConfig(1, 1, 1)),
                finalists -> List.of(),
                (candidate, warmup, measure, tier, cache) -> {
                    throw new AssertionError("refine should not run");
                },
                prepared -> preparedCalls.incrementAndGet(),
                update -> {
                    throw new AssertionError("progress should not run");
                },
                System::nanoTime
        );

        assertEquals(AutoTuneFinalizationResult.Status.EMPTY_AFTER_POSTCHECK, result.status());
        assertEquals(1, preparedCalls.get());
        assertEquals(0, result.finalists().size());
    }

    @Test
    void refinesFinalistsAndTracksBestResults() {
        AutoTuneResult a = phase1("A", 1.0, 4.0);
        AutoTuneResult b = phase1("B", 2.0, 1.0);

        List<RefineProgressUpdate> updates = new ArrayList<>();
        AutoTuneFinalizationResult result = AutoTuneFinalizer.finalizePhase1(
                List.of(a, b),
                new AutoTuneFinalizationConfig(2, false, new RefineConfig(1, 5, 7)),
                finalists -> finalists,
                (candidate, warmup, measure, tier, cache) -> {
                    assertEquals(5, warmup);
                    assertEquals(7, measure);
                    assertEquals("REFINE", tier);
                    if (candidate.name().equals("A")) {
                        return perf(candidate, 0.5, 4.0, 0.1);
                    }
                    return perf(candidate, 2.0, 0.5, 0.1);
                },
                prepared -> {},
                updates::add,
                () -> 0L
        );

        assertEquals(AutoTuneFinalizationResult.Status.OK, result.status());
        assertIterableEquals(List.of("A", "B"), result.finalists().stream().map(OptimizerCandidate::name).toList());
        AutoTuneBestResults best = result.bestResults();
        assertEquals("B", best.training().candidate().name());
        assertEquals("A", best.inference().candidate().name());
        assertEquals(2, updates.size());
        assertSame(best.training(), updates.get(1).bestTraining());
    }

    private static AutoTuneResult phase1(String name, double trainingScore, double inferenceScore) {
        OptimizerCandidate candidate = new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
        double forward = inferenceScore;
        double train = (trainingScore - (0.35 * forward) - (0.0005 * 20)) / 0.50;
        return new AutoTuneResult(candidate, 10, 20, forward, train, 0.1, trainingScore);
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
}
