package tuning.report;

public record TuningSummary(
        String strategyName,
        int selectedCount,
        int evaluatedCount,
        int validCount,
        int finalistCount,
        int historyEntriesWritten,
        double bestMedianMs
) {
    public TuningSummary {
        strategyName = strategyName == null ? "search" : strategyName;
    }
}
