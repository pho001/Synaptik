package tuning.autotune;

import java.io.PrintStream;
import java.util.Locale;

public final class LoggingAutotuneProgressListener implements AutotuneProgressListener {
    private final PrintStream out;
    private final int logEveryCandidates;
    private final long minIntervalMs;
    private long lastPrintedAtMs;

    public LoggingAutotuneProgressListener(PrintStream out, int logEveryCandidates, long minIntervalMs) {
        this.out = out == null ? System.out : out;
        this.logEveryCandidates = Math.max(1, logEveryCandidates);
        this.minIntervalMs = Math.max(0L, minIntervalMs);
        this.lastPrintedAtMs = 0L;
    }

    public static LoggingAutotuneProgressListener defaults() {
        return new LoggingAutotuneProgressListener(System.out, 1, 0L);
    }

    public static LoggingAutotuneProgressListener throttledDefaults() {
        return new LoggingAutotuneProgressListener(System.out, 10, 2_000L);
    }

    @Override
    public void onEvent(AutotuneProgressEvent event) {
        if (event == null) {
            return;
        }
        if (!shouldPrint(event)) {
            return;
        }
        lastPrintedAtMs = System.currentTimeMillis();
        out.println(format(event));
        out.flush();
    }

    private boolean shouldPrint(AutotuneProgressEvent event) {
        return switch (event.phase()) {
            case STARTED, SEARCH_BATCH, ROUND_COMPLETED, COMPLETED -> true;
            case CANDIDATE_INVALID, CANDIDATE_FAILED -> true;
            case CANDIDATE_MEASURED -> event.evaluatedCount() <= 3
                    || event.evaluatedCount() % logEveryCandidates == 0
                    || timeElapsed();
            case CANDIDATE_VALIDATING, CANDIDATE_MEASURING -> false;
        };
    }

    private boolean timeElapsed() {
        return minIntervalMs == 0L || (System.currentTimeMillis() - lastPrintedAtMs) >= minIntervalMs;
    }

    private static String format(AutotuneProgressEvent event) {
        StringBuilder sb = new StringBuilder("AutotuneProgress");
        sb.append(" phase=").append(event.phase().name());
        sb.append(" round=").append(event.round());
        if (event.totalCandidateCount() > 0) {
            sb.append(" total=").append(event.totalCandidateCount());
        }
        if (event.selectedCount() > 0) {
            sb.append(" selected=").append(event.selectedCount());
        }
        sb.append(" evaluated=").append(event.evaluatedCount());
        sb.append(" valid=").append(event.validCount());
        if (!event.candidateName().isBlank()) {
            sb.append(" candidate=").append(event.candidateName());
        }
        if (!event.bestCandidateName().isBlank()) {
            sb.append(" best=").append(event.bestCandidateName());
        }
        if (Double.isFinite(event.bestMedianMs())) {
            sb.append(" bestMedianMs=").append(String.format(Locale.US, "%.6f", event.bestMedianMs()));
        }
        if (!event.message().isBlank()) {
            sb.append(" message=").append(event.message());
        }
        return sb.toString();
    }
}
