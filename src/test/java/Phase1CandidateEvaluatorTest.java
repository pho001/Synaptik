import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.CorrectnessVerdict;
import benchmark.autotune.Phase1CandidateEvaluator;
import benchmark.autotune.Phase1CandidateResult;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

public class Phase1CandidateEvaluatorTest {

    @Test
    void skipsUnsafeHistoryWhenRescanDisabled() {
        AtomicInteger resets = new AtomicInteger();
        Phase1CandidateResult result = Phase1CandidateEvaluator.evaluate(
                candidate("C"),
                true,
                false,
                true,
                false,
                resets::incrementAndGet,
                c -> {
                    throw new AssertionError("safety should not run");
                },
                c -> {
                    throw new AssertionError("measure should not run");
                },
                (c, perf) -> {
                    throw new AssertionError("full should not run");
                }
        );

        assertEquals(Phase1CandidateResult.Status.SKIPPED_UNSAFE_HISTORY, result.status());
        assertEquals(0, resets.get());
    }

    @Test
    void returnsMismatchSafetyWithoutMeasuring() {
        AtomicInteger resets = new AtomicInteger();
        AtomicInteger measures = new AtomicInteger();
        Phase1CandidateResult result = Phase1CandidateEvaluator.evaluate(
                candidate("C"),
                false,
                false,
                true,
                false,
                resets::incrementAndGet,
                c -> new CorrectnessVerdict(false, 1.25),
                c -> {
                    measures.incrementAndGet();
                    return perf(c, 1.0);
                },
                (c, perf) -> new CorrectnessVerdict(true, 0.0)
        );

        assertEquals(Phase1CandidateResult.Status.MISMATCH_SAFETY, result.status());
        assertEquals(1, resets.get());
        assertEquals(0, measures.get());
        assertEquals("MISMATCH_SAFETY", result.unsafeReason());
    }

    @Test
    void returnsSafeSweepWhenOnlySafetySweepIsEnabled() {
        AtomicInteger measures = new AtomicInteger();
        Phase1CandidateResult result = Phase1CandidateEvaluator.evaluate(
                candidate("C"),
                false,
                false,
                true,
                true,
                () -> {},
                c -> new CorrectnessVerdict(true, 0.0),
                c -> {
                    measures.incrementAndGet();
                    return perf(c, 1.0);
                },
                (c, perf) -> new CorrectnessVerdict(true, 0.0)
        );

        assertEquals(Phase1CandidateResult.Status.SAFE_SWEEP, result.status());
        assertEquals(0, measures.get());
    }

    @Test
    void returnsMismatchFullAfterMeasurement() {
        CandidatePerf measured = perf(candidate("C"), 1.0);
        Phase1CandidateResult result = Phase1CandidateEvaluator.evaluate(
                candidate("C"),
                false,
                false,
                true,
                false,
                () -> {},
                c -> new CorrectnessVerdict(true, 0.0),
                c -> measured,
                (c, perf) -> new CorrectnessVerdict(false, 2.5)
        );

        assertEquals(Phase1CandidateResult.Status.MISMATCH_FULL, result.status());
        assertSame(measured, result.perf());
        assertEquals("MISMATCH_FULL", result.unsafeReason());
    }

    @Test
    void returnsValidPhase1WhenAllChecksPass() {
        CandidatePerf measured = perf(candidate("C"), 1.0);
        Phase1CandidateResult result = Phase1CandidateEvaluator.evaluate(
                candidate("C"),
                false,
                false,
                true,
                false,
                () -> {},
                c -> new CorrectnessVerdict(true, 0.0),
                c -> measured,
                (c, perf) -> new CorrectnessVerdict(true, 0.0)
        );

        assertEquals(Phase1CandidateResult.Status.VALID_PHASE1, result.status());
        assertSame(measured, result.perf());
        assertNull(result.unsafeReason());
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, double forward) {
        return new CandidatePerf(
                candidate,
                "CSE",
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forward,
                forward + 1.0,
                forward + 2.0
        );
    }
}
