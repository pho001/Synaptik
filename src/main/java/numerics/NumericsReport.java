package numerics;

import java.util.Locale;

/**
 * User-facing diagnostics report comparing two execution-profile candidates on the numerics harness.
 */
public final class NumericsReport {
    /** Scenario name used to identify the benchmark-like workload. */
    public final String scenarioName;
    /** Display name of the baseline candidate. */
    public final String candidateAName;
    /** Display name of the candidate being compared against the baseline. */
    public final String candidateBName;
    /** Output signal drift metrics. */
    public final NumericsMetrics.SignalMetrics out;
    /** Gradient drift metrics for input A. */
    public final NumericsMetrics.SignalMetrics gradA;
    /** Gradient drift metrics for input B. */
    public final NumericsMetrics.SignalMetrics gradB;
    /** Gradient drift metrics for input C. */
    public final NumericsMetrics.SignalMetrics gradC;
    /** Broadcast workload output drift metrics. */
    public final NumericsMetrics.SignalMetrics broadcast;
    /** Aggregate drift metrics across all report signals. */
    public final NumericsMetrics.AggregateMetrics aggregate;
    /** Policy verdict assigned to the aggregate metrics. */
    public final NumericsPolicy.Verdict verdict;

    /**
     * Creates a complete numerics diagnostics report.
     */
    public NumericsReport(
            String scenarioName,
            String candidateAName,
            String candidateBName,
            NumericsMetrics.SignalMetrics out,
            NumericsMetrics.SignalMetrics gradA,
            NumericsMetrics.SignalMetrics gradB,
            NumericsMetrics.SignalMetrics gradC,
            NumericsMetrics.SignalMetrics broadcast,
            NumericsMetrics.AggregateMetrics aggregate,
            NumericsPolicy.Verdict verdict
    ) {
        this.scenarioName = scenarioName;
        this.candidateAName = candidateAName;
        this.candidateBName = candidateBName;
        this.out = out;
        this.gradA = gradA;
        this.gradB = gradB;
        this.gradC = gradC;
        this.broadcast = broadcast;
        this.aggregate = aggregate;
        this.verdict = verdict;
    }

    /**
     * Formats this report as stable, human-readable diagnostic text for CLI and logs.
     *
     * @return multi-line report string
     */
    public String toPrettyString() {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("Numerics Report\n");
        sb.append("scenario=").append(scenarioName)
                .append(", A=").append(candidateAName)
                .append(", B=").append(candidateBName).append('\n');
        sb.append(line("out", out)).append('\n');
        sb.append(line("gradA", gradA)).append('\n');
        sb.append(line("gradB", gradB)).append('\n');
        sb.append(line("gradC", gradC)).append('\n');
        sb.append(line("broadcast", broadcast)).append('\n');
        sb.append("aggregate: maxAbs=").append(fmt(aggregate.maxAbs))
                .append(", maxRel=").append(fmt(aggregate.maxRel))
                .append(", maxUlp=").append(aggregate.maxUlp)
                .append(", invalid=").append(aggregate.invalidCount).append('\n');
        sb.append("verdict=").append(verdict.status).append(" (").append(verdict.reason).append(")\n");
        return sb.toString();
    }

    private static String line(String name, NumericsMetrics.SignalMetrics m) {
        return name + ": maxAbs=" + fmt(m.maxAbs)
                + ", avgAbs=" + fmt(m.avgAbs)
                + ", maxRel=" + fmt(m.maxRel)
                + ", maxUlp=" + m.maxUlp
                + ", p50Ulp=" + m.p50Ulp
                + ", p95Ulp=" + m.p95Ulp
                + ", invalid=" + m.invalidCount;
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.3e", v);
    }
}
