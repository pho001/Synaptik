package tuning.calibration;

import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.runtime.PlatformCalibrationCandidateSpaceFactory;
import tuning.preset.TuningPreset;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

/**
 * One ordered family of runtime-profile calibration.
 *
 * <p>A step receives the runtime profile selected by the previous step, generates
 * runtime candidates with {@link #candidateSpaceFactory()}, measures each
 * candidate across {@link #workloads()}, scores them with {@link #scorePolicy()},
 * and promotes the winner to the next step.</p>
 *
 * @param name display name for progress and reports
 * @param family calibration family that owns the mutated knobs
 * @param workloads workloads used to score candidates; must not be empty
 * @param preset preset supplying step-local benchmark policies
 * @param candidateSpaceFactory factory that creates runtime candidates from the current profile
 * @param scorePolicy scoring policy used to select the winning candidate
 */
public record PlatformCalibrationStep(
        String name,
        CalibrationFamilyId family,
        List<WorkloadSpec> workloads,
        TuningPreset preset,
        PlatformCalibrationCandidateSpaceFactory candidateSpaceFactory,
        PlatformCalibrationScorePolicy scorePolicy
) {
    public PlatformCalibrationStep {
        name = (name == null || name.isBlank()) ? "calibration-step" : name;
        family = family == null ? CalibrationFamilyId.SCHEDULER : family;
        workloads = workloads == null ? List.of() : List.copyOf(workloads);
        if (workloads.isEmpty()) {
            throw new IllegalArgumentException("workloads cannot be empty");
        }
        preset = preset == null ? TuningPreset.BALANCED : preset;
        Objects.requireNonNull(candidateSpaceFactory, "candidateSpaceFactory cannot be null");
        scorePolicy = scorePolicy == null ? PlatformCalibrationScorePolicy.averageMedianMs() : scorePolicy;
    }
}
