import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.autotune.AutoTuneProgressTracker;
import benchmark.autotune.AutoTuneResult;
import tensor.DataType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutoTuneProgressTrackerTest {

    @TempDir
    Path tempDir;

    @Test
    void initializesStartedSnapshotAndWritesRowsForPhase1AndRefine() throws Exception {
        Path progressPath = tempDir.resolve("progress.json");
        Path rowsPath = tempDir.resolve("rows.tsv");

        AutoTuneProgressTracker tracker = new AutoTuneProgressTracker(
                progressPath,
                rowsPath,
                DataType.FLOAT32,
                7,
                20,
                1000,
                Long.MAX_VALUE
        );

        String started = Files.readString(progressPath);
        assertTrue(started.contains("\"phase\": \"STARTED\""));
        assertTrue(started.contains("\"dtype\": \"FLOAT32\""));
        assertTrue(Files.readString(rowsPath).startsWith("timestamp\tphase\tstatus\tdtype\tcandidateStart"));

        AutoTuneResult bestTrain = result("TRAIN_BEST", 1.0);
        AutoTuneResult bestInf = result("INF_BEST", 2.0);
        tracker.recordPhase1(
                "VALID_PHASE1",
                candidate("AUTO_1"),
                1,
                1,
                0,
                0,
                0,
                0,
                bestTrain,
                bestInf,
                5.5,
                1.1,
                2.2,
                3.3,
                10,
                20
        );
        tracker.recordRefine(candidate("AUTO_2"), 1, 2, bestTrain, bestInf, 7.0, 1.4, 2.5, 3.6);
        tracker.complete("DONE", 1, 1, 0, 0, bestTrain, bestInf);

        String done = Files.readString(progressPath);
        assertTrue(done.contains("\"phase\": \"DONE\""));
        assertTrue(done.contains("\"bestTrainingCandidate\": \"TRAIN_BEST\""));
        assertTrue(done.contains("\"bestInferenceCandidate\": \"INF_BEST\""));

        List<String> rows = Files.readAllLines(rowsPath);
        assertEquals(3, rows.size());
        assertTrue(rows.get(1).contains("\tphase1\tVALID_PHASE1\tFLOAT32\t7\t1\t20\t1\t0\t0\tAUTO_1\t5.500\t5.500\t1.100\t2.200\t3.300\t10\t20"));
        assertTrue(rows.get(2).contains("\trefine\tREFINE_DONE 1/2\tFLOAT32\t7\t1\t20\t1\t0\t0\tAUTO_2\t7.000\tn/a\t1.400\t2.500\t3.600\t-1\t-1"));
    }

    @Test
    void phase1RowsAccumulateAverageRowMs() throws Exception {
        Path progressPath = tempDir.resolve("progress.json");
        Path rowsPath = tempDir.resolve("rows.tsv");

        AutoTuneProgressTracker tracker = new AutoTuneProgressTracker(
                progressPath,
                rowsPath,
                DataType.FLOAT64,
                0,
                10,
                1000,
                Long.MAX_VALUE
        );

        tracker.recordPhase1("ROW1", candidate("A"), 1, 1, 0, 0, 0, 0, null, null, 4.0, Double.NaN, Double.NaN, Double.NaN, -1, -1);
        tracker.recordPhase1("ROW2", candidate("B"), 2, 2, 0, 0, 0, 0, null, null, 6.0, Double.NaN, Double.NaN, Double.NaN, -1, -1);

        String json = Files.readString(progressPath);
        List<String> rows = Files.readAllLines(rowsPath);
        assertTrue(json.contains("\"avgRowMs\": 5.000"));
        assertTrue(rows.get(2).contains("\tROW2\tFLOAT64\t0\t2\t10\t2\t0\t0\tB\t6.000\t5.000\tn/a\tn/a\tn/a\t-1\t-1"));
    }

    private static AutoTuneResult result(String name, double score) {
        return new AutoTuneResult(candidate(name), 10, 20, 1.0, 2.0, 3.0, score);
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }
}
