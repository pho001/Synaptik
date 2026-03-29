package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongSupplier;

public final class AutoTuneFinalizer {
    private AutoTuneFinalizer() {}

    public static AutoTuneFinalizationResult finalizePhase1(
            List<AutoTuneResult> validPhase1,
            AutoTuneFinalizationConfig config,
            Function<List<OptimizerCandidate>, List<OptimizerCandidate>> numericsPostcheck,
            CandidatePerfSource refinePerfSource,
            Consumer<FinalistPreparationResult> finalistListener,
            Consumer<RefineProgressUpdate> refineProgressListener,
            LongSupplier nanoTimeSource
    ) {
        Consumer<FinalistPreparationResult> preparedSink = finalistListener == null ? prepared -> {} : finalistListener;
        Consumer<RefineProgressUpdate> refineSink = refineProgressListener == null ? step -> {} : refineProgressListener;

        if (validPhase1.isEmpty()) {
            return new AutoTuneFinalizationResult(
                    AutoTuneFinalizationResult.Status.NO_VALID_CANDIDATE,
                    List.of(),
                    List.of(),
                    null
            );
        }

        FinalistPreparationResult prepared = FinalistPreparation.prepare(
                validPhase1,
                config.refineTopK(),
                config.numericsPostcheckEnabled(),
                numericsPostcheck
        );
        preparedSink.accept(prepared);
        if (prepared.status() == FinalistPreparationResult.Status.EMPTY_AFTER_POSTCHECK) {
            return new AutoTuneFinalizationResult(
                    AutoTuneFinalizationResult.Status.EMPTY_AFTER_POSTCHECK,
                    prepared.finalists(),
                    List.of(),
                    null
            );
        }

        List<RefinedCandidate> refinedRows = RefineRunner.refine(
                prepared.finalists(),
                config.refineConfig(),
                refinePerfSource,
                nanoTimeSource
        );
        AutoTuneBestResults best = new AutoTuneBestResults(null, null);
        for (int i = 0; i < refinedRows.size(); i++) {
            RefinedCandidate refined = refinedRows.get(i);
            best = best.update(refined.perf());
            CandidatePerf row = refined.perf();
            refineSink.accept(new RefineProgressUpdate(
                    row.candidate(),
                    i + 1,
                    refinedRows.size(),
                    best.training(),
                    best.inference(),
                    refined.elapsedMs(),
                    row.forwardMs(),
                    row.trainMs(),
                    row.broadcastMs()
            ));
        }

        return new AutoTuneFinalizationResult(
                AutoTuneFinalizationResult.Status.OK,
                prepared.finalists(),
                refinedRows,
                best
        );
    }
}
