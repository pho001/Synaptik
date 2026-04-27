import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.TensorSnapshot;
import tuning.validate.ValidationPolicy;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ValidationEngineTest {
    @Test
    void snapshotValidationAcceptsMatchingOutput() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "snapshot_ok",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("snapshot_ok", WorkloadKind.GENERIC)
        );

        Candidate candidate = new Candidate("ok", profile(ExecutionMode.FORWARD));
        var instance = workload.instantiate(new WorkloadEnvironment(candidate.profile()));
        var result = new DefaultValidationEngine().validate(candidate, workload, instance, ValidationPolicy.defaults());

        assertTrue(result.valid());
        assertEquals("valid", result.status());
    }

    @Test
    void snapshotValidationRejectsMismatchingOutput() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "snapshot_bad",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(7.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("snapshot_bad", WorkloadKind.GENERIC)
        );

        Candidate candidate = new Candidate("bad", profile(ExecutionMode.FORWARD));
        var instance = workload.instantiate(new WorkloadEnvironment(candidate.profile()));
        var result = new DefaultValidationEngine().validate(candidate, workload, instance, ValidationPolicy.defaults());

        assertFalse(result.valid());
        assertTrue(result.reason().contains("mismatch"));
    }

    @Test
    void baselineProfileValidationAcceptsEquivalentExecution() {
        ExecutionProfile baselineProfile = profile(ExecutionMode.FORWARD);
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "baseline_ok",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
                    Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
                    return a.add(b).mul(a);
                },
                environment -> ValidationReference.baselineProfile(baselineProfile, List.of()),
                environment -> WorkloadMetadata.of("baseline_ok", WorkloadKind.GENERIC)
        );

        Candidate candidate = new Candidate("baseline", profile(ExecutionMode.FORWARD));
        var instance = workload.instantiate(new WorkloadEnvironment(candidate.profile()));
        var result = new DefaultValidationEngine().validate(candidate, workload, instance, ValidationPolicy.defaults());

        assertTrue(result.valid());
    }

    @Test
    void validationUsesExplicitValidationRootInsteadOfBenchmarkRoot() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "validation_root",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor validationOutput = Tensor.scalar(2.0).add(Tensor.scalar(3.0));
                    validationOutput.setLabel("validation_target");
                    return validationOutput.sum();
                },
                environment -> ValidationTarget.label("validation_target"),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("validation_root", WorkloadKind.GENERIC)
        );

        Candidate candidate = new Candidate("validation_root", profile(ExecutionMode.FORWARD));
        var instance = workload.instantiate(new WorkloadEnvironment(candidate.profile()));
        var result = new DefaultValidationEngine().validate(candidate, workload, instance, ValidationPolicy.defaults());

        assertTrue(result.valid());
    }

    @Test
    void benchmarkSessionSkipsMeasurementForInvalidCandidate() {
        AtomicInteger instantiations = new AtomicInteger();
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "session_invalid",
                WorkloadKind.GENERIC,
                environment -> {
                    instantiations.incrementAndGet();
                    return Tensor.scalar(2.0).add(Tensor.scalar(3.0));
                },
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(99.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("session_invalid", WorkloadKind.GENERIC)
        );

        Candidate candidate = new Candidate("invalid", profile(ExecutionMode.FORWARD));
        var report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(BenchmarkEntry.candidate(candidate.name(), candidate.profile())),
                tuning.measure.MeasurementPolicy.defaults(),
                ValidationPolicy.defaults(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertEquals(1, report.candidates().size());
        assertFalse(report.candidates().getFirst().success());
        assertTrue(report.candidates().getFirst().measurement() == null);
        assertTrue(instantiations.get() >= 1);
    }

    private static ExecutionProfile profile(ExecutionMode mode) {
        return new ExecutionProfile(
                "validation-profile",
                "validation-profile",
                DataType.FLOAT64,
                mode,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
