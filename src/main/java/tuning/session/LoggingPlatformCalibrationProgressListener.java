package tuning.session;

import java.io.PrintStream;
import java.util.Locale;

public final class LoggingPlatformCalibrationProgressListener implements PlatformCalibrationProgressListener {
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_RESET = "\u001B[0m";
    private static final int[] PROGRESS_MILESTONES = {10, 25, 50, 75, 100};

    private enum Mode {
        AGGREGATED,
        VERBOSE
    }

    private final PrintStream out;
    private final int logEveryCandidates;
    private final long minIntervalMs;
    private final Mode mode;
    private long lastPrintedAtMs;

    private String activeFamily = "";
    private String activeWorkload = "";
    private int activeFamilyStepIndex;
    private int activeFamilyStepCount;
    private int activeWorkloadIndex;
    private int activeWorkloadCount;
    private int lastProgressMilestone;
    private String lastLeaderId = "";
    private double lastLeaderScore = Double.NaN;

    public LoggingPlatformCalibrationProgressListener(PrintStream out, int logEveryCandidates, long minIntervalMs) {
        this(out, logEveryCandidates, minIntervalMs, Mode.AGGREGATED);
    }

    private LoggingPlatformCalibrationProgressListener(PrintStream out, int logEveryCandidates, long minIntervalMs, Mode mode) {
        this.out = out == null ? System.out : out;
        this.logEveryCandidates = Math.max(1, logEveryCandidates);
        this.minIntervalMs = Math.max(0L, minIntervalMs);
        this.mode = mode == null ? Mode.AGGREGATED : mode;
    }

    public static LoggingPlatformCalibrationProgressListener defaults() {
        return new LoggingPlatformCalibrationProgressListener(System.out, 8, 2_000L, Mode.AGGREGATED);
    }

    public static LoggingPlatformCalibrationProgressListener throttledDefaults() {
        return defaults();
    }

    public static LoggingPlatformCalibrationProgressListener verboseDefaults() {
        return new LoggingPlatformCalibrationProgressListener(System.out, 1, 0L, Mode.VERBOSE);
    }

    @Override
    public void onEvent(PlatformCalibrationProgressEvent event) {
        if (event == null) {
            return;
        }
        if (mode == Mode.VERBOSE) {
            handleVerbose(event);
        } else {
            handleAggregated(event);
        }
    }

    private void handleVerbose(PlatformCalibrationProgressEvent event) {
        if (!shouldPrintVerbose(event)) {
            return;
        }
        lastPrintedAtMs = System.currentTimeMillis();
        out.println(formatVerbose(event));
        out.flush();
    }

    private void handleAggregated(PlatformCalibrationProgressEvent event) {
        switch (event.phase()) {
            case STARTED -> print("Platform calibration started");
            case FAMILY_STARTED -> {
                activeFamily = event.family();
                activeWorkload = "";
                activeFamilyStepIndex = event.familyStepIndex();
                activeFamilyStepCount = event.familyStepCount();
                activeWorkloadIndex = 0;
                activeWorkloadCount = event.workloadCount();
                lastProgressMilestone = 0;
                lastLeaderId = "";
                lastLeaderScore = Double.NaN;
                print(String.format(
                        Locale.US,
                        "[%d/%d] %s",
                        activeFamilyStepIndex,
                        activeFamilyStepCount,
                        activeFamily
                ));
            }
            case WORKLOAD_STARTED -> {
                activeWorkload = event.workloadName();
                activeWorkloadIndex = event.workloadIndex();
                activeWorkloadCount = event.workloadCount();
                lastProgressMilestone = 0;
                print(String.format(
                        Locale.US,
                        "  workload %d/%d: %s",
                        activeWorkloadIndex,
                        activeWorkloadCount,
                        activeWorkload
                ));
            }
            case CANDIDATE_MEASURED, CANDIDATE_INVALID, CANDIDATE_FAILED, CANDIDATE_SCORED -> {
                boolean leaderChanged = !event.currentLeaderId().isBlank()
                        && (!event.currentLeaderId().equals(lastLeaderId)
                        || Double.compare(event.currentLeaderScore(), lastLeaderScore) != 0);
                int reachedMilestone = reachedProgressMilestone(event.candidateIndex(), event.candidateCount(), lastProgressMilestone);
                boolean shouldPrintProgress = reachedMilestone > 0;
                if (leaderChanged || shouldPrintProgress || event.phase() == PlatformCalibrationProgressPhase.CANDIDATE_FAILED) {
                    if (reachedMilestone > 0) {
                        lastProgressMilestone = reachedMilestone;
                    }
                    if (!event.currentLeaderId().isBlank()) {
                        lastLeaderId = event.currentLeaderId();
                        lastLeaderScore = event.currentLeaderScore();
                    }
                    print(formatAggregatedCandidateProgress(event, reachedMilestone, leaderChanged));
                }
            }
            case FAMILY_COMPLETED -> print(String.format(
                    Locale.US,
                    "  winner: %s score=%s",
                    event.candidateId().isBlank() ? "n/a" : event.candidateId(),
                    formatScore(event.currentLeaderScore())
            ));
            case COMPLETED -> print("Platform calibration completed");
            case FAILED -> print("Platform calibration failed: " + event.message());
            default -> {
            }
        }
    }

    private boolean shouldPrintVerbose(PlatformCalibrationProgressEvent event) {
        return switch (event.phase()) {
            case STARTED, FAMILY_STARTED, WORKLOAD_STARTED, FAMILY_COMPLETED, COMPLETED, FAILED -> true;
            case CANDIDATE_INVALID, CANDIDATE_FAILED, CANDIDATE_SCORED -> true;
            case CANDIDATE_MEASURED -> event.candidateIndex() <= 3
                    || event.candidateIndex() % logEveryCandidates == 0
                    || timeElapsed();
            case CANDIDATE_VALIDATING, CANDIDATE_MEASURING -> false;
        };
    }

    private boolean timeElapsed() {
        return minIntervalMs == 0L || (System.currentTimeMillis() - lastPrintedAtMs) >= minIntervalMs;
    }

    private void print(String line) {
        lastPrintedAtMs = System.currentTimeMillis();
        out.println(line);
        out.flush();
    }

    private static String formatVerbose(PlatformCalibrationProgressEvent event) {
        StringBuilder sb = new StringBuilder("PlatformCalibrationProgress");
        sb.append(" phase=").append(event.phase().name());
        if (!event.family().isBlank()) {
            sb.append(" family=").append(event.family());
        }
        if (event.familyStepCount() > 0) {
            sb.append(" step=").append(event.familyStepIndex()).append("/").append(event.familyStepCount());
        }
        if (!event.workloadName().isBlank()) {
            sb.append(" workload=").append(event.workloadName());
        }
        if (event.workloadCount() > 0) {
            sb.append(" workloadPos=").append(event.workloadIndex()).append("/").append(event.workloadCount());
        }
        if (!event.candidateId().isBlank()) {
            sb.append(" candidate=").append(event.candidateId());
        }
        if (event.candidateCount() > 0) {
            sb.append(" candidatePos=").append(event.candidateIndex()).append("/").append(event.candidateCount());
        }
        if (!event.currentLeaderId().isBlank()) {
            sb.append(" leader=").append(event.currentLeaderId());
        }
        if (Double.isFinite(event.currentLeaderScore())) {
            sb.append(" leaderScore=").append(String.format(Locale.US, "%.6f", event.currentLeaderScore()));
        }
        if (!event.message().isBlank()) {
            sb.append(" message=").append(event.message());
        }
        return sb.toString();
    }

    private static String formatAggregatedCandidateProgress(
            PlatformCalibrationProgressEvent event,
            int progressMilestone,
            boolean leaderChanged
    ) {
        StringBuilder sb = new StringBuilder("  ");
        boolean wrote = false;
        if (progressMilestone > 0) {
            sb.append("progress=").append(progressMilestone).append('%');
            wrote = true;
        }
        if (leaderChanged && !event.currentLeaderId().isBlank()) {
            if (wrote) {
                sb.append(' ');
            }
            sb.append(ANSI_GREEN)
                    .append("leader=")
                    .append(event.currentLeaderId());
            if (Double.isFinite(event.currentLeaderScore())) {
                sb.append(" score=").append(formatScore(event.currentLeaderScore()));
            }
            sb.append(ANSI_RESET);
            wrote = true;
        }
        if (event.phase() == PlatformCalibrationProgressPhase.CANDIDATE_FAILED) {
            if (wrote) {
                sb.append(' ');
            }
            sb.append("failure=").append(event.message());
        }
        return sb.toString();
    }

    private static String formatScore(double score) {
        return Double.isFinite(score) ? String.format(Locale.US, "%.6f", score) : "n/a";
    }

    private static int reachedProgressMilestone(int candidateIndex, int candidateCount, int lastMilestone) {
        if (candidateIndex <= 0 || candidateCount <= 0) {
            return 0;
        }
        int percent = (int) Math.floor((candidateIndex * 100.0d) / candidateCount);
        int reached = 0;
        for (int milestone : PROGRESS_MILESTONES) {
            if (milestone > lastMilestone && percent >= milestone) {
                reached = milestone;
            }
        }
        return reached;
    }
}
