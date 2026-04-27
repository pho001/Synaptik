package tuning.calibration.progress;

import java.io.PrintStream;
import java.util.Locale;

public final class TerminalCalibrationProgressRenderer implements PlatformCalibrationProgressListener {
    private static final int PANEL_LINES = 8;

    private final PrintStream out;
    private final boolean colorEnabled;
    private final boolean liveRedrawEnabled;
    private final EtaEstimator estimator = new EtaEstimator();
    private CalibrationProgressSnapshot snapshot = CalibrationProgressSnapshot.empty();
    private boolean panelPrinted;
    private long lastRenderAtMs;

    public TerminalCalibrationProgressRenderer(PrintStream out, TerminalCapabilities capabilities) {
        this.out = out == null ? System.out : out;
        TerminalCapabilities safe = capabilities == null ? new TerminalCapabilities(false, false) : capabilities;
        this.colorEnabled = safe.colorEnabled();
        this.liveRedrawEnabled = safe.liveRedrawEnabled();
    }

    public static TerminalCalibrationProgressRenderer create(String colorMode, String progressMode) {
        return new TerminalCalibrationProgressRenderer(
                System.out,
                TerminalCapabilities.detect(colorMode, progressMode, System.out)
        );
    }

    @Override
    public void onEvent(PlatformCalibrationProgressEvent event) {
        if (event == null) {
            return;
        }
        snapshot = snapshot.update(event);
        if (!shouldRender(event)) {
            return;
        }
        render();
    }

    private boolean shouldRender(PlatformCalibrationProgressEvent event) {
        long now = System.currentTimeMillis();
        boolean structural = switch (event.phase()) {
            case STARTED, FAMILY_STARTED, WORKLOAD_STARTED, FAMILY_COMPLETED, COMPLETED, FAILED -> true;
            default -> false;
        };
        if (structural || now - lastRenderAtMs >= 250L) {
            lastRenderAtMs = now;
            return true;
        }
        return false;
    }

    private void render() {
        if (liveRedrawEnabled && panelPrinted) {
            out.print("\u001B[" + PANEL_LINES + "F");
        }
        String[] lines = panelLines();
        for (String line : lines) {
            if (liveRedrawEnabled) {
                out.print(AnsiPalette.CLEAR_LINE);
            }
            out.println(line);
        }
        panelPrinted = true;
        out.flush();
    }

    private String[] panelLines() {
        int completedCandidates = Math.max(0, snapshot.candidateIndex());
        int totalCandidates = Math.max(0, snapshot.candidateCount());
        int completedFamilies = Math.max(0, snapshot.familyIndex() - 1);
        if (snapshot.phase() == PlatformCalibrationProgressPhase.FAMILY_COMPLETED) {
            completedFamilies = snapshot.familyIndex();
        }
        int totalFamilies = Math.max(0, snapshot.familyCount());
        String family = snapshot.family().isBlank() ? "n/a" : snapshot.family();
        String workload = snapshot.workloadName().isBlank() ? "n/a" : snapshot.workloadName();
        String candidate = snapshot.candidateId().isBlank() ? "n/a" : snapshot.candidateId();
        String leader = snapshot.leaderId().isBlank() ? "n/a" : snapshot.leaderId();
        String score = Double.isFinite(snapshot.leaderScore())
                ? String.format(Locale.US, "%.6f", snapshot.leaderScore())
                : "n/a";
        String title = AnsiPalette.color("Synaptik calibration", AnsiPalette.CYAN + AnsiPalette.BOLD, colorEnabled);
        return new String[]{
                title + "  phase=" + snapshot.phase().name().toLowerCase(Locale.ROOT),
                "family   " + pos(snapshot.familyIndex(), totalFamilies) + "  " + family,
                "workload " + pos(snapshot.workloadIndex(), snapshot.workloadCount()) + "  " + workload,
                "candidate " + pos(snapshot.candidateIndex(), totalCandidates) + "  " + candidate,
                "best      " + AnsiPalette.color(leader, AnsiPalette.GREEN, colorEnabled) + "  score=" + score,
                "elapsed   " + EtaEstimator.format(estimator.elapsed())
                        + "  eta-current=" + EtaEstimator.format(estimator.remaining(completedCandidates, totalCandidates)),
                "eta-total " + EtaEstimator.format(estimator.remaining(completedFamilies, totalFamilies)),
                "message   " + snapshot.message()
        };
    }

    private static String pos(int index, int count) {
        return Math.max(0, index) + "/" + Math.max(0, count);
    }
}
