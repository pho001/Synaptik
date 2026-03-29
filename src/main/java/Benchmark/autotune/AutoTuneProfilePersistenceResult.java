package Benchmark.autotune;

public record AutoTuneProfilePersistenceResult(
        double previousTrainingScore,
        double previousInferenceScore,
        boolean trainingImproved,
        boolean inferenceImproved,
        boolean hwTrainingImproved,
        boolean hwInferenceImproved
) {}
