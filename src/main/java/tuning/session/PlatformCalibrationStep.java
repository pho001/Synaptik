package tuning.session;

import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

public record PlatformCalibrationStep(
        String name,
        PlatformCalibrationFamily family,
        List<WorkloadSpec> workloads,
        TuningPreset preset,
        PlatformCalibrationCandidateSpaceFactory candidateSpaceFactory,
        PlatformCalibrationScorePolicy scorePolicy
) {
    public PlatformCalibrationStep {
        name = (name == null || name.isBlank()) ? "calibration-step" : name;
        family = family == null ? PlatformCalibrationFamily.SCHEDULER : family;
        workloads = workloads == null ? List.of() : List.copyOf(workloads);
        if (workloads.isEmpty()) {
            throw new IllegalArgumentException("workloads cannot be empty");
        }
        preset = preset == null ? TuningPreset.BALANCED : preset;
        Objects.requireNonNull(candidateSpaceFactory, "candidateSpaceFactory cannot be null");
        scorePolicy = scorePolicy == null ? PlatformCalibrationScorePolicy.averageMedianMs() : scorePolicy;
    }
}
