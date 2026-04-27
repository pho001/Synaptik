package tuning.calibration.progress;

public record PlatformCalibrationProgressEvent(
        PlatformCalibrationProgressPhase phase,
        String platformId,
        String family,
        int familyStepIndex,
        int familyStepCount,
        String workloadName,
        int workloadIndex,
        int workloadCount,
        String candidateId,
        int candidateIndex,
        int candidateCount,
        String currentLeaderId,
        double currentLeaderScore,
        String message
) {
    public PlatformCalibrationProgressEvent {
        phase = phase == null ? PlatformCalibrationProgressPhase.STARTED : phase;
        platformId = platformId == null ? "" : platformId;
        family = family == null ? "" : family;
        workloadName = workloadName == null ? "" : workloadName;
        candidateId = candidateId == null ? "" : candidateId;
        currentLeaderId = currentLeaderId == null ? "" : currentLeaderId;
        message = message == null ? "" : message;
    }
}
