package benchmark.autotune;

import benchmark.OptimizerCandidate;

public final class Phase1CandidateEvaluator {
    private Phase1CandidateEvaluator() {}

    @FunctionalInterface
    public interface CorrectnessProbe {
        CorrectnessVerdict probe(OptimizerCandidate candidate);
    }

    @FunctionalInterface
    public interface MeasurementProbe {
        CandidatePerf measure(OptimizerCandidate candidate);
    }

    @FunctionalInterface
    public interface FullCorrectnessProbe {
        CorrectnessVerdict probe(OptimizerCandidate candidate, CandidatePerf perf);
    }

    public static Phase1CandidateResult evaluate(
            OptimizerCandidate candidate,
            boolean unsafeInHistory,
            boolean rescanUnsafe,
            boolean safetyPrecheckEnabled,
            boolean safetySweepOnly,
            Runnable resetRuntimeState,
            CorrectnessProbe safetyProbe,
            MeasurementProbe measurementProbe,
            FullCorrectnessProbe fullProbe
    ) {
        if (unsafeInHistory && !rescanUnsafe) {
            return new Phase1CandidateResult(
                    Phase1CandidateResult.Status.SKIPPED_UNSAFE_HISTORY,
                    null,
                    null,
                    null,
                    null
            );
        }

        if (resetRuntimeState != null) {
            resetRuntimeState.run();
        }

        CorrectnessVerdict safetyVerdict = null;
        if (safetyPrecheckEnabled) {
            safetyVerdict = safetyProbe.probe(candidate);
            if (!safetyVerdict.ok()) {
                return new Phase1CandidateResult(
                        Phase1CandidateResult.Status.MISMATCH_SAFETY,
                        null,
                        safetyVerdict,
                        null,
                        "MISMATCH_SAFETY"
                );
            }
            if (safetySweepOnly) {
                return new Phase1CandidateResult(
                        Phase1CandidateResult.Status.SAFE_SWEEP,
                        null,
                        safetyVerdict,
                        null,
                        null
                );
            }
        }

        CandidatePerf perf = measurementProbe.measure(candidate);
        CorrectnessVerdict fullVerdict = fullProbe.probe(candidate, perf);
        if (!fullVerdict.ok()) {
            return new Phase1CandidateResult(
                    Phase1CandidateResult.Status.MISMATCH_FULL,
                    perf,
                    safetyVerdict,
                    fullVerdict,
                    "MISMATCH_FULL"
            );
        }

        return new Phase1CandidateResult(
                Phase1CandidateResult.Status.VALID_PHASE1,
                perf,
                safetyVerdict,
                fullVerdict,
                null
        );
    }
}
