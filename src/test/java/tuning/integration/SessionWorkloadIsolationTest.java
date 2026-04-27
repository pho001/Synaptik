package tuning.integration;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import graph.execution.trace.ExecutionTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.autotune.AutotuneRequest;
import tuning.autotune.AutotuneSession;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.calibration.PlatformCalibrationRequest;
import tuning.calibration.PlatformCalibrationScorePolicy;
import tuning.calibration.PlatformCalibrationSession;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.runtime.RuntimeProfileCandidate;
import tuning.candidate.Candidate;
import tuning.candidate.ListCandidateSpace;
import tuning.measure.MeasurementEngine;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.measure.MeasurementStatistics;
import tuning.benchmark.report.BenchmarkReport;
import tuning.search.FirstKSearchStrategy;
import tuning.search.SearchPolicy;
import tuning.store.BestProfileRecord;
import tuning.store.BestProfileStore;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformRuntimeProfileStore;
import tuning.store.TuningHistoryEntry;
import tuning.store.TuningHistoryStore;
import tuning.preset.TuningPreset;
import tuning.validate.ValidationEngine;
import tuning.validate.ValidationPolicy;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationResult;
import tuning.validate.ValidationTarget;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;
import tuning.workload.WorkloadSpec;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

final class SessionWorkloadIsolationTest {
    @Test
    void benchmarkSessionUsesFreshWorkloadForMeasurementAfterValidation() {
        TrackingWorkloadSpec workload = new TrackingWorkloadSpec("benchmark_isolation");
        TrackingValidationEngine validationEngine = new TrackingValidationEngine();
        TrackingMeasurementEngine measurementEngine = new TrackingMeasurementEngine(validationEngine);
        ExecutionProfile profile = profile("benchmark");

        BenchmarkReport report = BenchmarkSession.create(
                new BenchmarkRequest(
                        workload,
                        List.of(BenchmarkEntry.candidate("candidate", profile)),
                        new MeasurementPolicy(0, 1, 1, false, false, false, false, false),
                        ValidationPolicy.defaults(),
                        tuning.reporting.ReportPolicy.defaults()
                ),
                measurementEngine,
                validationEngine
        ).run();

        assertEquals(1, report.candidates().size());
        assertEquals(List.of(1), validationEngine.validatedIds);
        assertEquals(List.of(2), measurementEngine.measuredIds);
        assertEquals(2, workload.instantiatedIds.size());
    }

    @Test
    void autotuneSessionUsesFreshWorkloadForMeasurementAfterValidation() {
        TrackingWorkloadSpec workload = new TrackingWorkloadSpec("autotune_isolation");
        TrackingValidationEngine validationEngine = new TrackingValidationEngine();
        TrackingMeasurementEngine measurementEngine = new TrackingMeasurementEngine(validationEngine);
        ExecutionProfile profile = profile("autotune");
        Candidate candidate = new Candidate("candidate", profile);

        AutotuneSession.create(
                new AutotuneRequest(
                        workload,
                        new ListCandidateSpace(List.of(candidate)),
                        new MeasurementPolicy(0, 1, 1, false, false, false, false, false),
                        ValidationPolicy.defaults(),
                        new SearchPolicy(1, 1, 1, false),
                        PersistencePolicy.disabled()
                ),
                new FirstKSearchStrategy(1),
                measurementEngine,
                validationEngine,
                new InMemoryBestProfileStore(),
                new InMemoryHistoryStore()
        ).run();

        assertEquals(List.of(1), validationEngine.validatedIds);
        assertEquals(List.of(2), measurementEngine.measuredIds);
        assertEquals(List.of(1, 2, 3), workload.instantiatedIds);
    }

    @Test
    void platformCalibrationSessionUsesFreshWorkloadForMeasurementAfterValidation() throws Exception {
        TrackingWorkloadSpec workload = new TrackingWorkloadSpec("platform_isolation");
        TrackingValidationEngine validationEngine = new TrackingValidationEngine();
        TrackingMeasurementEngine measurementEngine = new TrackingMeasurementEngine(validationEngine);
        ExecutionProfile seed = profile("platform-seed");
        Path out = java.nio.file.Files.createTempFile("platform-isolation-", ".json");

        PlatformCalibrationSession.create(
                PlatformCalibrationRequest.fromSeedExecutionProfile(
                        "test-platform",
                        seed,
                        List.of(new PlatformCalibrationStep(
                                "step",
                                CalibrationFamilyId.MATMUL,
                                List.of(workload),
                                TuningPreset.QUICK,
                                base -> ignoredWorkload -> List.of(new RuntimeProfileCandidate("candidate", base, Map.of())),
                                PlatformCalibrationScorePolicy.averageMedianMs()
                        )),
                        out,
                        new MeasurementPolicy(0, 1, 1, false, false, false, false, false)
                ),
                measurementEngine,
                validationEngine,
                new InMemoryPlatformRuntimeProfileStore()
        ).run();

        assertEquals(List.of(1), validationEngine.validatedIds);
        assertEquals(List.of(2), measurementEngine.measuredIds);
        assertEquals(2, workload.instantiatedIds.size());
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private record TrackingWorkloadInstance(
            int id,
            Tensor root,
            ValidationTarget validationTarget,
            ValidationReference reference,
            WorkloadMetadata metadata
    ) implements WorkloadInstance {
    }

    private static final class TrackingWorkloadSpec implements WorkloadSpec {
        private final String name;
        private int nextId = 1;
        private final List<Integer> instantiatedIds = new ArrayList<>();

        private TrackingWorkloadSpec(String name) {
            this.name = name;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public WorkloadKind kind() {
            return WorkloadKind.GENERIC;
        }

        @Override
        public WorkloadInstance instantiate(WorkloadEnvironment environment) {
            int id = nextId++;
            instantiatedIds.add(id);
            return new TrackingWorkloadInstance(
                    id,
                    Tensor.scalar((double) id),
                    ValidationTarget.root(),
                    ValidationReference.none(),
                    WorkloadMetadata.of(name, WorkloadKind.GENERIC)
            );
        }
    }

    private static final class TrackingValidationEngine implements ValidationEngine {
        private final List<Integer> validatedIds = new ArrayList<>();
        private final List<WorkloadInstance> validatedInstances = new ArrayList<>();

        @Override
        public ValidationResult validate(Candidate candidate, WorkloadSpec workloadSpec, WorkloadInstance workload, ValidationPolicy policy) {
            TrackingWorkloadInstance tracking = (TrackingWorkloadInstance) workload;
            validatedIds.add(tracking.id());
            validatedInstances.add(workload);
            return new ValidationResult(true, "valid", "", Map.of());
        }
    }

    private static final class TrackingMeasurementEngine implements MeasurementEngine {
        private final TrackingValidationEngine validationEngine;
        private final List<Integer> measuredIds = new ArrayList<>();

        private TrackingMeasurementEngine(TrackingValidationEngine validationEngine) {
            this.validationEngine = validationEngine;
        }

        @Override
        public MeasurementResult measure(Candidate candidate, WorkloadInstance workload, MeasurementPolicy policy) {
            TrackingWorkloadInstance tracking = (TrackingWorkloadInstance) workload;
            measuredIds.add(tracking.id());
            assertFalse(validationEngine.validatedInstances.contains(workload));
            return new MeasurementResult(
                    policy,
                    new ExecutionTrace(null, null, null),
                    new MeasurementStatistics(1.0d, 1.0d, 1.0d)
            );
        }
    }

    private static final class InMemoryBestProfileStore implements BestProfileStore {
        @Override
        public void save(Path path, BestProfileRecord record) {
        }

        @Override
        public Optional<BestProfileRecord> load(Path path) {
            return Optional.empty();
        }
    }

    private static final class InMemoryHistoryStore implements TuningHistoryStore {
        @Override
        public void append(Path path, TuningHistoryEntry entry) {
        }

        @Override
        public List<TuningHistoryEntry> loadAll(Path path) {
            return List.of();
        }
    }

    private static final class InMemoryPlatformRuntimeProfileStore implements PlatformRuntimeProfileStore {
        @Override
        public void save(Path path, PlatformRuntimeProfile profile) {
        }

        @Override
        public Optional<PlatformRuntimeProfile> load(Path path) {
            return Optional.empty();
        }
    }
}
