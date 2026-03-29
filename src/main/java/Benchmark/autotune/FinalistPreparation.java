package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public final class FinalistPreparation {
    private FinalistPreparation() {}

    public static FinalistPreparationResult prepare(
            List<AutoTuneResult> validPhase1,
            int refineTopK,
            boolean numericsPostcheckEnabled,
            Function<List<OptimizerCandidate>, List<OptimizerCandidate>> numericsPostcheck
    ) {
        List<AutoTuneResult> selectedPhase1 = Phase1FinalistSelector.selectFinalists(
                validPhase1,
                AutoTuneResult::score,
                r -> r.forwardMs() + (0.0005 * r.graphInfSize()),
                AutoTuneResult::candidate,
                refineTopK
        );
        List<OptimizerCandidate> finalistsList = new ArrayList<>(selectedPhase1.size());
        for (AutoTuneResult row : selectedPhase1) {
            finalistsList.add(row.candidate());
        }

        if (numericsPostcheckEnabled) {
            finalistsList = numericsPostcheck.apply(finalistsList);
            if (finalistsList.isEmpty()) {
                return new FinalistPreparationResult(FinalistPreparationResult.Status.EMPTY_AFTER_POSTCHECK, finalistsList);
            }
        }
        return new FinalistPreparationResult(FinalistPreparationResult.Status.OK, finalistsList);
    }
}
