package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongSupplier;

public final class Phase1Runner {
    private Phase1Runner() {}

    public static Phase1LoopResult run(
            List<OptimizerCandidate> candidates,
            Function<OptimizerCandidate, Phase1CandidateResult> evaluator,
            Consumer<Phase1Step> stepConsumer,
            LongSupplier nanoTimeSource
    ) {
        LongSupplier clock = nanoTimeSource == null ? System::nanoTime : nanoTimeSource;
        Consumer<Phase1Step> consumer = stepConsumer == null ? step -> {} : stepConsumer;

        Phase1Counters counters = Phase1Counters.zero();
        List<AutoTuneResult> validPhase1 = new ArrayList<>();

        for (OptimizerCandidate candidate : candidates) {
            long startedNs = clock.getAsLong();
            Phase1CandidateResult result = evaluator.apply(candidate);
            counters = counters.advance(result.status());
            if (result.status() == Phase1CandidateResult.Status.VALID_PHASE1 && result.perf() != null) {
                validPhase1.add(AutoTuneResult.forTraining(result.perf()));
            }
            double rowMs = (clock.getAsLong() - startedNs) / 1_000_000.0;
            consumer.accept(new Phase1Step(candidate, result, counters, rowMs));
        }

        return new Phase1LoopResult(counters, validPhase1);
    }
}
