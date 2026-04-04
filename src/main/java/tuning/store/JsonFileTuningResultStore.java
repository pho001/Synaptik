package tuning.store;

import tuning.report.JsonBenchmarkReportRenderer;
import tuning.session.TuningResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class JsonFileTuningResultStore implements TuningResultStore {
    @Override
    public void save(Path path, TuningResult result) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"bestProfileName\": \"")
                .append(result.bestProfile() == null ? "" : result.bestProfile().candidateName())
                .append("\",\n");
        sb.append("  \"persisted\": ").append(result.persisted()).append(",\n");
        sb.append("  \"summary\": \"").append(result.summary().replace("\"", "\\\"")).append("\",\n");
        sb.append("  \"finalists\": [\n");
        for (int i = 0; i < result.finalists().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            sb.append(JsonBenchmarkReportRenderer.render(
                    new tuning.report.BenchmarkReport(
                            "tuning_finalist",
                            java.time.OffsetDateTime.now(),
                            java.util.List.of(result.finalists().get(i)),
                            result.finalists().get(i).candidate().name()
                    )
            ).indent(4).stripTrailing());
        }
        sb.append("\n  ]\n");
        sb.append("}\n");

        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write tuning result to " + path, e);
        }
    }
}
