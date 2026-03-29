package Benchmark.autotune;

import java.io.IOException;

@FunctionalInterface
public interface AutoTunePersistencePort {
    AutoTuneProfilePersistenceResult persist(
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference,
            int validCount,
            int mismatchCount
    ) throws IOException;
}
