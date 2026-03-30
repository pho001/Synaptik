package benchmark.autotune;

import benchmark.OptimizerProfileIO;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class AutoTuneProfilePersistence {
    private AutoTuneProfilePersistence() {}

    public static AutoTuneProfilePersistenceResult persist(
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference,
            int validCount,
            int mismatchCount,
            Path bestTrainingPath,
            Path bestInferencePath,
            Path bestAliasPath,
            Path runtimeProfilePath,
            Path hwProfilePath,
            String hwBucket,
            int hwProfileMaxBuckets
    ) throws IOException {
        if (bestTrainingPath.getParent() != null) {
            Files.createDirectories(bestTrainingPath.getParent());
        }

        double previousTrainingScore = OptimizerProfileIO.loadScoreOrInfinity(bestTrainingPath);
        double previousInferenceScore = OptimizerProfileIO.loadScoreOrInfinity(bestInferencePath);
        boolean trainingImproved = bestTraining.score() + 1e-12 < previousTrainingScore;
        boolean inferenceImproved = bestInference.score() + 1e-12 < previousInferenceScore;

        if (trainingImproved) {
            Files.writeString(bestTrainingPath, bestTraining.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
            Files.writeString(bestAliasPath, bestTraining.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
            OptimizerProfileIO.saveKnobs(runtimeProfilePath, bestTraining.candidate().knobs(), bestTraining.candidate().name());
        }

        if (inferenceImproved) {
            Files.writeString(bestInferencePath, bestInference.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
        }

        boolean hwTrainingImproved = OptimizerProfileIO.saveHardwareProfileIfImproved(
                hwProfilePath,
                hwBucket,
                "TRAINING",
                bestTraining.candidate(),
                bestTraining.score(),
                hwProfileMaxBuckets
        );
        boolean hwInferenceImproved = OptimizerProfileIO.saveHardwareProfileIfImproved(
                hwProfilePath,
                hwBucket,
                "INFERENCE",
                bestInference.candidate(),
                bestInference.score(),
                hwProfileMaxBuckets
        );

        return new AutoTuneProfilePersistenceResult(
                previousTrainingScore,
                previousInferenceScore,
                trainingImproved,
                inferenceImproved,
                hwTrainingImproved,
                hwInferenceImproved
        );
    }
}
