package tuning.calibration.progress;

import java.time.Duration;

public final class EtaEstimator {
    private final long startedAtNanos = System.nanoTime();

    public Duration elapsed() {
        return Duration.ofNanos(Math.max(0L, System.nanoTime() - startedAtNanos));
    }

    public Duration remaining(int completedUnits, int totalUnits) {
        if (completedUnits <= 0 || totalUnits <= 0 || completedUnits >= totalUnits) {
            return Duration.ZERO;
        }
        long elapsedNanos = Math.max(1L, System.nanoTime() - startedAtNanos);
        long estimatedTotal = (long) ((elapsedNanos / (double) completedUnits) * totalUnits);
        return Duration.ofNanos(Math.max(0L, estimatedTotal - elapsedNanos));
    }

    public static String format(Duration duration) {
        if (duration == null || duration.isZero() || duration.isNegative()) {
            return "00:00";
        }
        long seconds = duration.toSeconds();
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        long secs = seconds % 60;
        if (hours > 0) {
            return "%02d:%02d:%02d".formatted(hours, minutes, secs);
        }
        return "%02d:%02d".formatted(minutes, secs);
    }
}
