package numerics;

import java.util.Locale;

public final class NumericsReport {
    public final String scenarioName;
    public final String candidateAName;
    public final String candidateBName;
    public final NumericsMetrics.SignalMetrics out;
    public final NumericsMetrics.SignalMetrics gradA;
    public final NumericsMetrics.SignalMetrics gradB;
    public final NumericsMetrics.SignalMetrics gradC;
    public final NumericsMetrics.SignalMetrics broadcast;
    public final NumericsMetrics.AggregateMetrics aggregate;
    public final NumericsPolicy.Verdict verdict;

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

