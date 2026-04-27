package tuning.autotune.report;

import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.autotune.TuningResult;

import java.util.Locale;

public final class JsonTuningResultRenderer {
    private JsonTuningResultRenderer() {
    }

    public static String render(TuningResult result) {
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"bestProfile\": \"").append(result.bestProfile() == null ? "" : escape(result.bestProfile().candidateName())).append("\",\n");
        sb.append("  \"persisted\": ").append(result.persisted()).append(",\n");
        sb.append("  \"summary\": \"").append(escape(result.summary())).append("\",\n");
        sb.append("  \"details\": {\n");
        sb.append("    \"strategy\": \"").append(escape(result.details().strategyName())).append("\",\n");
        sb.append("    \"selected\": ").append(result.details().selectedCount()).append(",\n");
        sb.append("    \"evaluated\": ").append(result.details().evaluatedCount()).append(",\n");
        sb.append("    \"valid\": ").append(result.details().validCount()).append(",\n");
        sb.append("    \"finalists\": ").append(result.details().finalistCount()).append(",\n");
        sb.append("    \"historyEntriesWritten\": ").append(result.details().historyEntriesWritten()).append(",\n");
        sb.append("    \"bestMedianMs\": ").append(format(result.details().bestMedianMs())).append("\n");
        sb.append("  },\n");
        sb.append("  \"finalists\": [\n");
        for (int i = 0; i < result.finalists().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            BenchmarkCandidateReport finalist = result.finalists().get(i);
            sb.append("    {");
            sb.append("\"name\": \"").append(escape(finalist.candidate().name())).append("\", ");
            sb.append("\"validation\": \"").append(escape(finalist.validation().status())).append("\"");
            if (finalist.measurement() != null) {
                sb.append(", \"medianMs\": ").append(format(finalist.measurement().steadyStateStats().medianMs()));
                sb.append(", \"meanMs\": ").append(format(finalist.measurement().steadyStateStats().meanMs()));
            }
            sb.append("}");
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
