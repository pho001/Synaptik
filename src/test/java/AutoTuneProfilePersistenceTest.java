import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneProfilePersistence;
import benchmark.autotune.AutoTuneProfilePersistenceResult;
import benchmark.autotune.AutoTuneResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutoTuneProfilePersistenceTest {

    @TempDir
    Path tempDir;

    @Test
    void persistWritesProfilesWhenScoresImprove() throws Exception {
        Path bestTraining = tempDir.resolve("best-training.json");
        Path bestInference = tempDir.resolve("best-inference.json");
        Path bestAlias = tempDir.resolve("best.json");
        Path runtimeProfile = tempDir.resolve("runtime.json");
        Path hwProfiles = tempDir.resolve("hw.tsv");

        AutoTuneResult training = result("TRAIN", 1.0);
        AutoTuneResult inference = result("INF", 2.0);

        AutoTuneProfilePersistenceResult persisted = AutoTuneProfilePersistence.persist(
                training,
                inference,
                10,
                0,
                bestTraining,
                bestInference,
                bestAlias,
                runtimeProfile,
                hwProfiles,
                "test-bucket",
                10
        );

        assertTrue(persisted.trainingImproved());
        assertTrue(persisted.inferenceImproved());
        assertTrue(Files.exists(bestTraining));
        assertTrue(Files.exists(bestInference));
        assertTrue(Files.exists(bestAlias));
        assertTrue(Files.exists(runtimeProfile));
        assertTrue(Files.exists(hwProfiles));
    }

    @Test
    void persistKeepsExistingProfilesWhenScoresDoNotImprove() throws Exception {
        Path bestTraining = tempDir.resolve("best-training.json");
        Path bestInference = tempDir.resolve("best-inference.json");
        Path bestAlias = tempDir.resolve("best.json");
        Path runtimeProfile = tempDir.resolve("runtime.json");
        Path hwProfiles = tempDir.resolve("hw.tsv");

        AutoTuneResult betterTraining = result("TRAIN_BEST", 1.0);
        AutoTuneResult betterInference = result("INF_BEST", 1.0);
        Files.writeString(bestTraining, betterTraining.toJson(10, 0));
        Files.writeString(bestInference, betterInference.toJson(10, 0));
        Files.writeString(bestAlias, betterTraining.toJson(10, 0));
        Files.writeString(runtimeProfile, "{existing:true}");

        AutoTuneResult worseTraining = result("TRAIN_WORSE", 2.0);
        AutoTuneResult worseInference = result("INF_WORSE", 3.0);

        AutoTuneProfilePersistenceResult persisted = AutoTuneProfilePersistence.persist(
                worseTraining,
                worseInference,
                10,
                0,
                bestTraining,
                bestInference,
                bestAlias,
                runtimeProfile,
                hwProfiles,
                "test-bucket",
                10
        );

        assertFalse(persisted.trainingImproved());
        assertFalse(persisted.inferenceImproved());
        assertTrue(Files.readString(bestTraining).contains("TRAIN_BEST"));
        assertTrue(Files.readString(bestInference).contains("INF_BEST"));
        assertTrue(Files.readString(runtimeProfile).contains("existing:true"));
    }

    private static AutoTuneResult result(String name, double score) {
        return new AutoTuneResult(
                new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults()),
                10,
                20,
                1.0,
                2.0,
                3.0,
                score
        );
    }
}
