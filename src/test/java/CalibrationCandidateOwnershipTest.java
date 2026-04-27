import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.calibration.run.CalibrationSuite;
import tuning.preset.TuningPreset;

public class CalibrationCandidateOwnershipTest {
    @Test
    void standardFamilyCandidatesOnlyReportOwnedKnobs() {
        for (DataType dtype : java.util.List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            PlatformRuntimeProfile seed = PlatformRuntimeProfile.fromExecutionProfile(
                    "platform",
                    "hardware",
                    "TEST",
                    seed(dtype)
            );
            for (var family : CalibrationFamilyRegistry.standardSuite()) {
                for (var step : CalibrationSuite.stepsFor(family, "ownership-" + family.name(), TuningPreset.QUICK, dtype)) {
                    var candidates = step.candidateSpaceFactory().create(seed).generate(step.workloads().getFirst());
                    for (var candidate : candidates) {
                        CalibrationFamilyRegistry.validateCandidateChanges(step.family(), candidate.knobAssignments());
                    }
                }
            }
        }
    }

    private static ExecutionProfile seed(DataType dtype) {
        return new ExecutionProfile(
                "seed",
                "seed",
                dtype,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }
}
