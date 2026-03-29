package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongSupplier;

public final class AutoTuneSessionRunner {
    private AutoTuneSessionRunner() {}

    public static AutoTuneSessionResult run(
            List<OptimizerCandidate> candidates,
            AutoTuneSessionConfig config,
            Function<OptimizerCandidate, Phase1CandidateResult> phase1Evaluator,
            Consumer<Phase1Step> phase1StepListener,
            Function<List<OptimizerCandidate>, List<OptimizerCandidate>> numericsPostcheck,
            CandidatePerfSource refinePerfSource,
            Consumer<FinalistPreparationResult> finalistListener,
            Consumer<RefineProgressUpdate> refineProgressListener,
            AutoTunePersistencePort persistencePort,
            Runnable historySaver,
            LongSupplier nanoTimeSource
    ) throws IOException {
        Runnable saveHistory = historySaver == null ? () -> {} : historySaver;

        Phase1LoopResult phase1 = Phase1Runner.run(
                candidates,
                phase1Evaluator,
                phase1StepListener,
                nanoTimeSource
        );
        Phase1Counters counters = phase1.counters();

        if (config.safetySweepOnly()) {
            if (!config.safetyStateless()) {
                saveHistory.run();
            }
            return new AutoTuneSessionResult(
                    AutoTuneSessionResult.Status.SAFE_SWEEP_DONE,
                    counters,
                    phase1,
                    null,
                    null
            );
        }

        AutoTuneFinalizationResult finalization = AutoTuneFinalizer.finalizePhase1(
                phase1.validPhase1(),
                config.finalizationConfig(),
                numericsPostcheck,
                refinePerfSource,
                finalistListener,
                refineProgressListener,
                nanoTimeSource
        );

        if (finalization.status() == AutoTuneFinalizationResult.Status.NO_VALID_CANDIDATE) {
            return new AutoTuneSessionResult(
                    AutoTuneSessionResult.Status.NO_VALID_CANDIDATE,
                    counters,
                    phase1,
                    finalization,
                    null
            );
        }

        if (finalization.status() == AutoTuneFinalizationResult.Status.EMPTY_AFTER_POSTCHECK) {
            saveHistory.run();
            return new AutoTuneSessionResult(
                    AutoTuneSessionResult.Status.EMPTY_AFTER_POSTCHECK,
                    counters,
                    phase1,
                    finalization,
                    null
            );
        }

        AutoTuneBestResults best = finalization.bestResults();
        AutoTuneProfilePersistenceResult persisted = persistencePort == null
                ? null
                : persistencePort.persist(
                best.training(),
                best.inference(),
                counters.valid(),
                counters.mismatch()
        );
        saveHistory.run();
        return new AutoTuneSessionResult(
                AutoTuneSessionResult.Status.DONE,
                counters,
                phase1,
                finalization,
                persisted
        );
    }
}
