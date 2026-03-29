import Benchmark.OptimizationStage;
import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;
import Benchmark.autotune.NumericsPostcheckConfig;
import Benchmark.autotune.NumericsPostcheckDrop;
import Benchmark.autotune.NumericsPostcheckResult;
import Benchmark.autotune.NumericsPostcheckRunner;
import Benchmark.autotune.UnsafeCandidateHistory;
import Numerics.NumericsMetrics;
import Numerics.NumericsPolicy;
import Numerics.NumericsReport;
import Tensor.DataType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NumericsPostcheckRunnerTest {

    @TempDir
    Path tempDir;

    @Test
    void dropsUnsafeCandidatesMarksHistoryAndWritesReport() throws Exception {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");
        OptimizerCandidate c = candidate("C");
        UnsafeCandidateHistory history = UnsafeCandidateHistory.empty("ctx");
        List<String> probePairs = new ArrayList<>();

        NumericsPostcheckResult result = NumericsPostcheckRunner.run(
                List.of(a, b, c),
                new NumericsPostcheckConfig(
                        DataType.FLOAT32,
                        2,
                        tempDir,
                        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
                ),
                (baseline, finalist) -> {
                    probePairs.add(baseline.name() + "->" + finalist.name() + " stages=" + baseline.stageOrder().size());
                    if (finalist.name().equals("A")) {
                        return report(finalist.name(), NumericsPolicy.Verdict.unsafe("too much drift"), 2.0e-3);
                    }
                    return report(finalist.name(), NumericsPolicy.Verdict.safe("ok"), 1.0e-7);
                },
                history,
                candidate -> "fp-" + candidate.name()
        );

        assertEquals(List.of("B", "C"), result.keptCandidates().stream().map(OptimizerCandidate::name).toList());
        assertEquals(2, result.checked());
        assertEquals(1, result.markedUnsafe());
        assertEquals(1, result.droppedUnsafe().size());
        NumericsPostcheckDrop dropped = result.droppedUnsafe().get(0);
        assertEquals("A", dropped.candidateName());
        assertTrue(dropped.reason().contains("too much drift"));
        assertTrue(history.isUnsafe("fp-A"));
        assertEquals(List.of("A_NUM_NOOPT->A stages=0", "B_NUM_NOOPT->B stages=0"), probePairs);
        assertNotNull(result.reportPath());
        String report = Files.readString(result.reportPath());
        assertTrue(report.contains("candidate\tstatus\treason"));
        assertTrue(report.contains("A\tUNSAFE\ttoo much drift"));
        assertTrue(report.contains("B\tSAFE\tok"));
    }

    @Test
    void zeroTopNBypassesProbeAndLeavesReportEmpty() {
        OptimizerCandidate a = candidate("A");
        OptimizerCandidate b = candidate("B");

        NumericsPostcheckResult result = NumericsPostcheckRunner.run(
                List.of(a, b),
                new NumericsPostcheckConfig(
                        DataType.FLOAT64,
                        0,
                        tempDir,
                        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
                ),
                (baseline, finalist) -> {
                    throw new AssertionError("probe should not run when topN=0");
                },
                UnsafeCandidateHistory.empty("ctx"),
                candidate -> "fp-" + candidate.name()
        );

        assertEquals(List.of("A", "B"), result.keptCandidates().stream().map(OptimizerCandidate::name).toList());
        assertEquals(0, result.checked());
        assertEquals(0, result.markedUnsafe());
        assertTrue(result.droppedUnsafe().isEmpty());
        assertNull(result.reportPath());
    }

    private static NumericsReport report(String candidateName, NumericsPolicy.Verdict verdict, double maxAbs) {
        NumericsMetrics.SignalMetrics signal = new NumericsMetrics.SignalMetrics(maxAbs, maxAbs / 2.0, maxAbs, 3L, 1L, 2L, 8, 0);
        NumericsMetrics.AggregateMetrics aggregate = new NumericsMetrics.AggregateMetrics(maxAbs, maxAbs, 3L, 0);
        return new NumericsReport(
                "autotune-postcheck",
                candidateName + "_NUM_NOOPT",
                candidateName,
                signal,
                signal,
                signal,
                signal,
                signal,
                aggregate,
                verdict
        );
    }

    private static OptimizerCandidate candidate(String name) {
        return new OptimizerCandidate(name, List.of(OptimizationStage.CSE), TuningKnobs.trainingDefaults());
    }
}
