package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

import java.util.ArrayList;
import java.util.List;
import java.util.function.LongSupplier;

public final class RefineRunner {
    private RefineRunner() {}

    public static List<RefinedCandidate> refine(
            List<OptimizerCandidate> finalists,
            RefineConfig config,
            CandidatePerfSource perfSource,
            LongSupplier nanoTimeSource
    ) {
        LongSupplier clock = nanoTimeSource == null ? System::nanoTime : nanoTimeSource;
        List<RefinedCandidate> out = new ArrayList<>(finalists.size());
        for (OptimizerCandidate candidate : finalists) {
            long startedNs = clock.getAsLong();
            double sumFwdMs = 0.0;
            double sumTrainMs = 0.0;
            double sumBroadcastMs = 0.0;
            int graphInfSize = -1;
            int graphTrnSize = -1;
            CoarseKnobSignature coarse = null;
            String stageOrderKey = null;

            for (int r = 0; r < config.repeats(); r++) {
                CandidatePerf perf = perfSource.measure(
                        candidate,
                        config.warmupIters(),
                        config.measureIters(),
                        "REFINE",
                        null
                );
                graphInfSize = perf.graphInfSize();
                graphTrnSize = perf.graphTrnSize();
                coarse = perf.coarseKnobSignature();
                stageOrderKey = perf.stageOrderKey();
                sumFwdMs += perf.forwardMs();
                sumTrainMs += perf.trainMs();
                sumBroadcastMs += perf.broadcastMs();
            }

            out.add(new RefinedCandidate(
                    new CandidatePerf(
                            candidate,
                            stageOrderKey,
                            coarse,
                            graphInfSize,
                            graphTrnSize,
                            sumFwdMs / config.repeats(),
                            sumTrainMs / config.repeats(),
                            sumBroadcastMs / config.repeats()
                    ),
                    (clock.getAsLong() - startedNs) / 1_000_000.0
            ));
        }
        return out;
    }
}
