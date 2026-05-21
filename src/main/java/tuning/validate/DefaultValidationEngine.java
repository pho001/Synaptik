package tuning.validate;

import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.compile.intent.BackendIntentPlan;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DefaultValidationEngine implements ValidationEngine {
    @Override
    public ValidationResult validate(
            Candidate candidate,
            WorkloadSpec workloadSpec,
            WorkloadInstance workload,
            ValidationPolicy policy
    ) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }
        if (workloadSpec == null) {
            throw new IllegalArgumentException("workloadSpec cannot be null");
        }
        if (workload == null) {
            throw new IllegalArgumentException("workload cannot be null");
        }
        if (policy == null || !policy.enabled()) {
            return ValidationResult.skipped();
        }

        ValidationReference reference = workload.reference();
        if (reference == null || reference.kind() == ValidationReferenceKind.NONE) {
            return ValidationResult.skipped();
        }

        return switch (reference.kind()) {
            case NONE -> ValidationResult.skipped();
            case SNAPSHOT -> validateAgainstSnapshot(
                    workload.root(),
                    workload.validationTarget(),
                    workload.backendIntentPlan(),
                    candidate.profile(),
                    (SnapshotValidationReference) reference.payload(),
                    policy
            );
            case BASELINE_PROFILE -> validateAgainstBaseline(
                    workloadSpec,
                    workload.root(),
                    workload.validationTarget(),
                    workload.backendIntentPlan(),
                    candidate.profile(),
                    (BaselineProfileValidationReference) reference.payload(),
                    policy
            );
        };
    }

    private ValidationResult validateAgainstSnapshot(
            Tensor executionRoot,
            ValidationTarget candidateValidationTarget,
            BackendIntentPlan candidateBackendIntentPlan,
            ExecutionProfile candidateProfile,
            SnapshotValidationReference reference,
            ValidationPolicy policy
    ) {
        Tensor candidateValidationRoot = candidateValidationTarget.resolve(executionRoot);
        execute(candidateValidationRoot, candidateProfile, candidateBackendIntentPlan);

        LinkedHashMap<String, Double> metrics = new LinkedHashMap<>();
        ValidationResult outputResult = compareTensor(
                "output",
                candidateValidationRoot,
                reference.output(),
                policy,
                metrics
        );
        if (!outputResult.valid()) {
            return outputResult;
        }

        if (policy.requireGradientMatch()) {
            ValidationResult gradResult = compareSnapshotGradients(
                    executionRoot,
                    reference.gradients(),
                    reference.gradientTargetLabels(),
                    policy,
                    metrics
            );
            if (!gradResult.valid()) {
                return gradResult;
            }
        }

        return new ValidationResult(true, "valid", "", metrics);
    }

    private ValidationResult validateAgainstBaseline(
            WorkloadSpec workloadSpec,
            Tensor candidateRoot,
            ValidationTarget candidateValidationTarget,
            BackendIntentPlan candidateBackendIntentPlan,
            ExecutionProfile candidateProfile,
            BaselineProfileValidationReference reference,
            ValidationPolicy policy
    ) {
        Tensor candidateValidationRoot = candidateValidationTarget.resolve(candidateRoot);
        execute(candidateValidationRoot, candidateProfile, candidateBackendIntentPlan);

        WorkloadInstance baselineWorkload = workloadSpec.instantiate(new WorkloadEnvironment(reference.baselineProfile()));
        Tensor baselineRoot = baselineWorkload.root();
        Tensor baselineValidationRoot = baselineWorkload.validationTarget().resolve(baselineRoot);
        execute(baselineValidationRoot, reference.baselineProfile(), baselineWorkload.backendIntentPlan());

        LinkedHashMap<String, Double> metrics = new LinkedHashMap<>();
        ValidationResult outputResult = compareTensor(
                "output",
                candidateValidationRoot,
                TensorSnapshot.capture("baseline_output", baselineValidationRoot),
                policy,
                metrics
        );
        if (!outputResult.valid()) {
            return outputResult;
        }

        if (policy.requireGradientMatch()) {
            ValidationResult gradResult = compareBaselineGradients(
                    candidateRoot,
                    baselineRoot,
                    reference.gradientTargetLabels(),
                    policy,
                    metrics
            );
            if (!gradResult.valid()) {
                return gradResult;
            }
        }

        return new ValidationResult(true, "valid", "", metrics);
    }

    private static void execute(
            Tensor root,
            ExecutionProfile profile,
            BackendIntentPlan backendIntentPlan
    ) {
        CompiledGraph.compile(root, profile.compile(), compileModeFor(profile.mode()), backendIntentPlan)
                .prepare(profile.runtime())
                .execute(profile.mode());
    }

    private static void execute(Tensor root, ExecutionProfile profile) {
        execute(root, profile, BackendIntentPlan.empty());
    }

    private static tensor.CompileMode compileModeFor(backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD
                ? tensor.CompileMode.TRAINING
                : tensor.CompileMode.INFERENCE_ONLY;
    }

    private ValidationResult compareSnapshotGradients(
            Tensor candidateRoot,
            Map<String, TensorSnapshot> expectedGradients,
            List<String> gradientTargetLabels,
            ValidationPolicy policy,
            Map<String, Double> metrics
    ) {
        List<String> labels = !gradientTargetLabels.isEmpty()
                ? gradientTargetLabels
                : new ArrayList<>(expectedGradients.keySet());
        if (labels.isEmpty()) {
            return ValidationResult.failure("Gradient validation requested but no gradient targets were provided.");
        }

        Map<String, Tensor> candidateTensors = tensorsByLabel(candidateRoot);
        for (String label : labels) {
            Tensor candidate = candidateTensors.get(label);
            TensorSnapshot expected = expectedGradients.get(label);
            if (candidate == null) {
                return ValidationResult.failure("Missing candidate tensor for gradient target: " + label);
            }
            if (expected == null) {
                return ValidationResult.failure("Missing expected gradient snapshot for target: " + label);
            }
            Tensor candidateGradient = candidate.getGradient();
            if (candidateGradient == null) {
                return ValidationResult.failure("Candidate gradient is missing for target: " + label);
            }
            ValidationResult result = compareTensor("grad:" + label, candidateGradient, expected, policy, metrics);
            if (!result.valid()) {
                return result;
            }
        }
        return new ValidationResult(true, "valid", "", metrics);
    }

    private ValidationResult compareBaselineGradients(
            Tensor candidateRoot,
            Tensor baselineRoot,
            List<String> gradientTargetLabels,
            ValidationPolicy policy,
            Map<String, Double> metrics
    ) {
        Map<String, Tensor> candidateTensors = tensorsByLabel(candidateRoot);
        Map<String, Tensor> baselineTensors = tensorsByLabel(baselineRoot);

        List<String> labels;
        if (!gradientTargetLabels.isEmpty()) {
            labels = gradientTargetLabels;
        } else {
            labels = candidateTensors.keySet().stream()
                    .filter(label -> {
                        Tensor c = candidateTensors.get(label);
                        Tensor b = baselineTensors.get(label);
                        return c != null && b != null && c.getGradient() != null && b.getGradient() != null;
                    })
                    .toList();
        }
        if (labels.isEmpty()) {
            return ValidationResult.failure("Gradient validation requested but no comparable gradients were found.");
        }

        for (String label : labels) {
            Tensor candidate = candidateTensors.get(label);
            Tensor baseline = baselineTensors.get(label);
            if (candidate == null || baseline == null) {
                return ValidationResult.failure("Missing tensor for gradient target: " + label);
            }
            if (candidate.getGradient() == null || baseline.getGradient() == null) {
                return ValidationResult.failure("Missing gradient for target: " + label);
            }
            ValidationResult result = compareTensor(
                    "grad:" + label,
                    candidate.getGradient(),
                    TensorSnapshot.capture("baseline_grad:" + label, baseline.getGradient()),
                    policy,
                    metrics
            );
            if (!result.valid()) {
                return result;
            }
        }
        return new ValidationResult(true, "valid", "", metrics);
    }

    private ValidationResult compareTensor(
            String prefix,
            Tensor actualTensor,
            TensorSnapshot expectedSnapshot,
            ValidationPolicy policy,
            Map<String, Double> metrics
    ) {
        if (actualTensor == null) {
            return ValidationResult.failure(prefix + " tensor is null");
        }
        if (expectedSnapshot == null) {
            return ValidationResult.failure(prefix + " expected snapshot is null");
        }
        if (actualTensor.getDataType() != expectedSnapshot.dataType()) {
            return ValidationResult.failure(prefix + " dtype mismatch: actual=" + actualTensor.getDataType() + ", expected=" + expectedSnapshot.dataType());
        }
        if (!expectedSnapshot.shapeEquals(actualTensor)) {
            return ValidationResult.failure(prefix + " shape mismatch: actual=" + Arrays.toString(actualTensor.getShapeUnsafe())
                    + ", expected=" + Arrays.toString(expectedSnapshot.shape()));
        }

        if (actualTensor.getDataType() == DataType.BOOL) {
            boolean[] actual = actualTensor.toBooleanArrayCopy();
            boolean[] expected = expectedSnapshot.boolValues();
            if (expected == null) {
                return ValidationResult.failure(prefix + " expected BOOL snapshot is missing bool values");
            }
            if (!Arrays.equals(actual, expected)) {
                return ValidationResult.failure(prefix + " bool values mismatch");
            }
            metrics.put(prefix + ".mismatchCount", 0.0d);
            return new ValidationResult(true, "valid", "", metrics);
        }

        double[] actual = actualTensor.toDoubleArrayCopy();
        double[] expected = expectedSnapshot.numericValues();
        if (expected == null) {
            return ValidationResult.failure(prefix + " expected numeric snapshot is missing numeric values");
        }
        if (actual.length != expected.length) {
            return ValidationResult.failure(prefix + " length mismatch");
        }

        double maxAbs = 0.0d;
        double maxRel = 0.0d;
        double absTolerance = policy.absTolerance(actualTensor.getDataType());
        double relTolerance = policy.relTolerance(actualTensor.getDataType());
        for (int i = 0; i < actual.length; i++) {
            double abs = Math.abs(actual[i] - expected[i]);
            double rel = abs / Math.max(1e-30d, Math.abs(expected[i]));
            maxAbs = Math.max(maxAbs, abs);
            maxRel = Math.max(maxRel, rel);
            if (abs > absTolerance && rel > relTolerance) {
                return ValidationResult.failure(prefix + " mismatch at index " + i
                        + ": actual=" + actual[i]
                        + ", expected=" + expected[i]
                        + ", abs=" + abs
                        + ", rel=" + rel);
            }
        }

        metrics.put(prefix + ".maxAbs", maxAbs);
        metrics.put(prefix + ".maxRel", maxRel);
        return new ValidationResult(true, "valid", "", metrics);
    }

    private Map<String, Tensor> tensorsByLabel(Tensor root) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Tensor tensor : root.topologicalSort()) {
            String label = tensor.getLabel();
            if (label == null || label.isBlank()) {
                continue;
            }
            out.putIfAbsent(label, tensor);
        }
        return out;
    }
}
