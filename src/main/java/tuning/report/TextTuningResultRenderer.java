package tuning.report;

import tuning.session.TuningResult;

import java.util.Locale;

public final class TextTuningResultRenderer {
    private TextTuningResultRenderer() {
    }

    public static String render(TuningResult result) {
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Tuning Result\n");
        sb.append("bestProfile=").append(result.bestProfile() == null ? "n/a" : result.bestProfile().candidateName()).append('\n');
        sb.append("persisted=").append(result.persisted()).append('\n');
        sb.append("summary=").append(result.summary()).append("\n\n");

        TuningSummary details = result.details();
        sb.append("Details\n");
        sb.append("strategy=").append(details.strategyName()).append('\n');
        sb.append("selected=").append(details.selectedCount()).append('\n');
        sb.append("evaluated=").append(details.evaluatedCount()).append('\n');
        sb.append("valid=").append(details.validCount()).append('\n');
        sb.append("finalists=").append(details.finalistCount()).append('\n');
        sb.append("historyEntriesWritten=").append(details.historyEntriesWritten()).append('\n');
        sb.append("bestMedianMs=").append(Double.isFinite(details.bestMedianMs()) ? String.format(Locale.US, "%.6f", details.bestMedianMs()) : "n/a").append("\n\n");

        if (!result.finalists().isEmpty()) {
            sb.append("Finalists\n");
            sb.append(String.format(Locale.US, "%-24s %-12s %-12s %-12s%n", "name", "medianMs", "meanMs", "validation"));
            for (BenchmarkCandidateReport finalist : result.finalists()) {
                double median = finalist.measurement() == null ? Double.NaN : finalist.measurement().steadyStateStats().medianMs();
                double mean = finalist.measurement() == null ? Double.NaN : finalist.measurement().steadyStateStats().meanMs();
                sb.append(String.format(
                        Locale.US,
                        "%-24s %-12s %-12s %-12s%n",
                        finalist.candidate().name(),
                        Double.isFinite(median) ? String.format(Locale.US, "%.6f", median) : "n/a",
                        Double.isFinite(mean) ? String.format(Locale.US, "%.6f", mean) : "n/a",
                        finalist.validation().status()
                ));
            }
        }
        return sb.toString();
    }
}
