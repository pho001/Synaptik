import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.report.BenchmarkReport;
import tuning.session.BenchmarkEntry;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileBenchmarkReportStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.JsonFileProfileStore;
import tuning.store.TuningHistoryEntry;
import tuning.store.WorkloadFingerprint;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TuningStoreTest {
    @Test
    void profileAndBenchmarkStoresWriteFiles() throws Exception {
        ExecutionProfile profile = new ExecutionProfile(
                "store",
                "store",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        Path profilePath = Files.createTempFile("tuning-profile-", ".json");
        new JsonFileProfileStore(profile).save(profilePath, profile);
        assertTrue(Files.size(profilePath) > 0);

        BenchmarkReport report = tuning.session.BenchmarkSession.create(new tuning.session.BenchmarkRequest(
                new TensorRootWorkloadSpec("store_workload", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                List.of(BenchmarkEntry.candidate("store", profile)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        )).run();

        Path reportPath = Files.createTempFile("benchmark-report-", ".json");
        new JsonFileBenchmarkReportStore().saveBenchmark(reportPath, report);
        assertTrue(Files.size(reportPath) > 0);
    }

    @Test
    void bestProfileAndHistoryStoresWriteFiles() throws Exception {
        ExecutionProfile profile = new ExecutionProfile(
                "store-best",
                "store-best",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        Path bestPath = Files.createTempFile("best-profile-", ".json");
        Path historyPath = Files.createTempFile("tuning-history-", ".jsonl");

        var hardware = HardwareFingerprint.capture();
        var workload = WorkloadFingerprint.of(
                new TensorRootWorkloadSpec("store_workload", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                WorkloadMetadata.of("store_workload", WorkloadKind.GENERIC),
                profile
        );

        new JsonFileBestProfileStore().save(bestPath, new tuning.store.BestProfileRecord(
                hardware,
                workload,
                profile,
                1.23d,
                OffsetDateTime.now()
        ));
        new JsonFileTuningHistoryStore().append(historyPath, new TuningHistoryEntry(
                "fp-candidate",
                "candidate",
                true,
                1.0d,
                1.1d,
                1.0d,
                "",
                "summary",
                OffsetDateTime.now(),
                hardware,
                workload
        ));

        assertTrue(Files.size(bestPath) > 0);
        assertTrue(Files.size(historyPath) > 0);
    }
}
