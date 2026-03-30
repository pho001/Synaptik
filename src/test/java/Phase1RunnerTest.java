import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneResult;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.CorrectnessVerdict;
import benchmark.autotune.Phase1CandidateResult;
import benchmark.autotune.Phase1Counters;
import benchmark.autotune.Phase1LoopResult;
import benchmark.autotune.Phase1Runner;
import benchmark.autotune.Phase1Step;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

public class Phase1RunnerTest {

    @Test
    void runnerAggregatesCountsAndCollectsValidPhase1Rows() {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        OptimizerCandidate c = candidate("C");
        OptimizerCandidate d = candidate("D");
        OptimizerCandidate e = candidate("E");

        List<Phase1Step> steps = new ArrayList<>();
        Phase1LoopResult result = Phase1Runner.run(
                List.of(a, b, c, d, e),
                candidate -> switch (candidate.name()) {
                    case "A" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.SKIPPED_UNSAFE_HISTORY,
                            null, null, null, null
                    );
                    case "B" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.MISMATCH_SAFETY,
                            null, new CorrectnessVerdict(false, 1.0), null, "MISMATCH_SAFETY"
                    );
                    case "C" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.SAFE_SWEEP,
                            null, new CorrectnessVerdict(true, 0.0), null, null
                    );
                    case "D" -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.MISMATCH_FULL,
                            perf(candidate, 2.0), null, new CorrectnessVerdict(false, 2.0), "MISMATCH_FULL"
                    );
                    default -> new Phase1CandidateResult(
                            Phase1CandidateResult.Status.VALID_PHASE1,
                            perf(candidate, 3.0), null, new CorrectnessVerdict(true, 0.0), null
                    );
                },
                steps::add,
                new ScriptedClock(0L, 5_000_000L, 5_000_000L, 11_000_000L, 11_000_000L, 18_000_000L, 18_000_000L, 26_000_000L, 26_000_000L, 35_000_000L)
        );

        Phase1Counters counters = result.counters();
        assertEquals(5, counters.processed());
        assertEquals(1, counters.valid());
        assertEquals(2, counters.mismatch());
        assertEquals(1, counters.mismatchSafety());
        assertEquals(1, counters.mismatchFull());
        assertEquals(1, counters.skippedUnsafe());
        assertEquals(1, counters.safetySweepSafe());
        assertEquals(1, result.validPhase1().size());
        assertEquals("E", result.validPhase1().get(0).candidate().name());
        assertEquals(5, steps.size());
        assertIterableEquals(List.of("A", "B", "C", "D", "E"), steps.stream().map(s -> s.candidate().name()).toList());
    }

    @Test
    void runnerComputesRowElapsedMsFromClock() {
        OptimizerCandidate a = candidate("A");
        List<Phase1Step> steps = new ArrayList<>();

        Phase1Runner.run(
                List.of(a),
                candidate -> new Phase1CandidateResult(
                        Phase1CandidateResult.Status.VALID_PHASE1,
                        perf(candidate, 1.0), null, new CorrectnessVerdict(true, 0.0), null
                ),
                steps::add,
                new ScriptedClock(0L, 12_500_000L)
        );

        assertEquals(1, steps.size());
        assertEquals(12.5, steps.get(0).rowMs(), 1e-12);
    }

    private static CandidatePerf perf(OptimizerCandidate candidate, double forwardMs) {
        return new CandidatePerf(
                candidate,
                "CSE",
                CoarseKnobSignature.of(candidate),
                10,
                20,
                forwardMs,
                forwardMs + 1.0,
                forwardMs + 2.0
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
