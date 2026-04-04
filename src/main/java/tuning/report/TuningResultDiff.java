package tuning.report;

import tuning.session.TuningResult;

public record TuningResultDiff(
        String previousBestProfile,
        String currentBestProfile,
        double previousBestMedianMs,
        double currentBestMedianMs,
        double bestSpeedupVsPrevious,
        int previousFinalistCount,
        int currentFinalistCount
) {
    public static TuningResultDiff compare(TuningResult previous, TuningResult current) {
        if (previous == null || current == null) {
            throw new IllegalArgumentException("previous and current tuning results cannot be null");
        }
        double previousMedian = previous.details().bestMedianMs();
        double currentMedian = current.details().bestMedianMs();
        return new TuningResultDiff(
                previous.bestProfile() == null ? "" : previous.bestProfile().candidateName(),
                current.bestProfile() == null ? "" : current.bestProfile().candidateName(),
                previousMedian,
                currentMedian,
                Double.isFinite(previousMedian) && Double.isFinite(currentMedian) && currentMedian > 0.0
                        ? previousMedian / currentMedian
                        : Double.NaN,
                previous.finalists().size(),
                current.finalists().size()
        );
    }
}
