import config.runtime.BlasProvider;
import runtime.contract.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.report.BenchmarkReport;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBenchmarkReportStore;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.TuningHistoryEntry;
import tuning.store.WorkloadFingerprint;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;
import tuning.workload.TensorRootWorkloadSpec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TuningStoreTest {
    @Test
    void profileAndBenchmarkStoresWriteFiles() throws Exception {
        ExecutionProfile profile = new ExecutionProfile(
                "store",
                "store",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        Path profilePath = Files.createTempFile("tuning-profile-", ".json");
        new JsonFileProfileStore(profile).save(profilePath, profile);
        assertTrue(Files.size(profilePath) > 0);

        BenchmarkReport report = tuning.benchmark.BenchmarkSession.create(new tuning.benchmark.BenchmarkRequest(
                new TensorRootWorkloadSpec("store_workload", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                List.of(BenchmarkEntry.candidate("store", profile)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
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
                config.compile.CompileConfig.inference(),
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

        CandidateMetadata metadata = CandidateMetadata.graphStandard("current")
                .withAttribute("graphParameter", "CURRENT_GRAPH_POLICY");

        new JsonFileBestProfileStore().save(bestPath, new tuning.store.BestProfileRecord(
                hardware,
                workload,
                profile,
                1.23d,
                OffsetDateTime.now(),
                "graph-autotune",
                "STANDARD",
                CandidateKind.GRAPH_STANDARD,
                metadata,
                "runtime-profile",
                true
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
                workload,
                CandidateKind.GRAPH_STANDARD,
                metadata,
                "runtime-profile",
                true
        ));

        assertTrue(Files.size(bestPath) > 0);
        assertTrue(Files.size(historyPath) > 0);
        var loadedBest = new JsonFileBestProfileStore().load(bestPath).orElseThrow();
        var loadedHistory = new JsonFileTuningHistoryStore().loadAll(historyPath).getFirst();
        assertEquals(CandidateKind.GRAPH_STANDARD, loadedBest.candidateKind());
        assertEquals(CandidateKind.GRAPH_STANDARD, loadedHistory.candidateKind());
        assertEquals("graph-autotune", loadedBest.candidateMetadata().candidateSpaceId());
        assertEquals("current", loadedBest.candidateMetadata().parameterVariant());
        assertEquals("CURRENT_GRAPH_POLICY", loadedBest.candidateMetadata().attributes().get("graphParameter"));
        assertEquals("graph-autotune", loadedHistory.candidateMetadata().candidateSpaceId());
        assertEquals("current", loadedHistory.candidateMetadata().parameterVariant());
        assertEquals("CURRENT_GRAPH_POLICY", loadedHistory.candidateMetadata().attributes().get("graphParameter"));
        assertEquals("runtime-profile", loadedBest.runtimeProfileId());
        assertEquals("runtime-profile", loadedHistory.runtimeProfileId());
        assertTrue(loadedBest.productionEligible());
        assertTrue(loadedHistory.productionEligible());
    }

    @Test
    void bestProfileRecordRebasesGraphPolicyOntoCurrentRuntimeProfile() {
        ExecutionProfile measuredWinner = new ExecutionProfile(
                "abc-f32-graph-autotune",
                "offload=cpu-only+cpuRegion=natural+cpuFusion=balanced",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
        RuntimeConfig currentRuntime = new RuntimeConfig(
                KernelTuningConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                new BlasConfig(BlasProvider.OPENBLAS_FFM, 4_000_000L, true, 1.5d, false)
        );
        PlatformRuntimeProfile currentRuntimeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "macos-arm64",
                "hardware",
                "TEST",
                new ExecutionProfile(
                        "platform-runtime",
                        "platform-runtime",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD_BACKWARD,
                        config.compile.CompileConfig.noGraphOptimizationBaseline(),
                        currentRuntime,
                        WorkloadProfile.none()
                )
        );
        WorkloadFingerprint workload = WorkloadFingerprint.of(
                new TensorRootWorkloadSpec("store_workload", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                WorkloadMetadata.of("store_workload", WorkloadKind.GENERIC),
                measuredWinner
        );
        var record = new tuning.store.BestProfileRecord(
                HardwareFingerprint.capture(),
                workload,
                measuredWinner,
                1.23d,
                OffsetDateTime.now(),
                "autotune",
                "STANDARD",
                CandidateKind.GRAPH_STANDARD,
                CandidateMetadata.graphStandard("current"),
                currentRuntimeProfile.metadata().platformProfileId(),
                true
        );

        ExecutionProfile rebound = record.rebaseOnRuntime(currentRuntimeProfile);

        assertEquals(measuredWinner.profileName(), rebound.profileName());
        assertEquals(measuredWinner.candidateName(), rebound.candidateName());
        assertEquals(measuredWinner.compile(), rebound.compile());
        assertEquals(BlasProvider.OPENBLAS_FFM, rebound.runtime().blas().provider());
        assertEquals(4_000_000L, rebound.runtime().blas().matmulMinWork());

        PlatformRuntimeProfile wrongDtypeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "macos-arm64-f64",
                "hardware",
                "TEST",
                new ExecutionProfile(
                        "platform-runtime-f64",
                        "platform-runtime-f64",
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD_BACKWARD,
                        config.compile.CompileConfig.noGraphOptimizationBaseline(),
                        currentRuntime,
                        WorkloadProfile.none()
                )
        );
        PlatformRuntimeProfile wrongModeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "macos-arm64-forward",
                "hardware",
                "TEST",
                new ExecutionProfile(
                        "platform-runtime-forward",
                        "platform-runtime-forward",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.noGraphOptimizationBaseline(),
                        currentRuntime,
                        WorkloadProfile.none()
                )
        );

        assertThrows(IllegalArgumentException.class, () -> record.rebaseOnRuntime(wrongDtypeProfile));
        assertThrows(IllegalArgumentException.class, () -> record.rebaseOnRuntime(wrongModeProfile));
    }
}
