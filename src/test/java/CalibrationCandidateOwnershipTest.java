import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.calibration.run.CalibrationSuite;
import tuning.ownership.TuningKnobOwner;
import tuning.ownership.TuningKnobOwnership;
import tuning.preset.TuningPreset;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
                        candidate.knobAssignments().keySet().forEach(knob ->
                                assertEquals(TuningKnobOwner.PLATFORM_DTYPE, TuningKnobOwnership.ownerOf(knob)));
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
                config.compile.CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }
}
