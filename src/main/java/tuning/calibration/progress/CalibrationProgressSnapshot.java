package tuning.calibration.progress;

public record CalibrationProgressSnapshot(
        PlatformCalibrationProgressPhase phase,
        String family,
        int familyIndex,
        int familyCount,
        String workloadName,
        int workloadIndex,
        int workloadCount,
        String candidateId,
        int candidateIndex,
        int candidateCount,
        String leaderId,
        double leaderScore,
        String message
) {
    public static CalibrationProgressSnapshot empty() {
        return new CalibrationProgressSnapshot(
                PlatformCalibrationProgressPhase.STARTED,
                "",
                0,
                0,
                "",
                0,
                0,
                "",
                0,
                0,
                "",
                Double.NaN,
                ""
        );
    }

    public CalibrationProgressSnapshot update(PlatformCalibrationProgressEvent event) {
        if (event == null) {
            return this;
        }
        return new CalibrationProgressSnapshot(
                event.phase(),
                choose(event.family(), family),
                event.familyStepIndex() > 0 ? event.familyStepIndex() : familyIndex,
                event.familyStepCount() > 0 ? event.familyStepCount() : familyCount,
                choose(event.workloadName(), workloadName),
                event.workloadIndex() > 0 ? event.workloadIndex() : workloadIndex,
                event.workloadCount() > 0 ? event.workloadCount() : workloadCount,
                choose(event.candidateId(), candidateId),
                event.candidateIndex() > 0 ? event.candidateIndex() : candidateIndex,
                event.candidateCount() > 0 ? event.candidateCount() : candidateCount,
                choose(event.currentLeaderId(), leaderId),
                Double.isFinite(event.currentLeaderScore()) ? event.currentLeaderScore() : leaderScore,
                choose(event.message(), message)
        );
    }

    private static String choose(String next, String current) {
        return next == null || next.isBlank() ? (current == null ? "" : current) : next;
    }
}
