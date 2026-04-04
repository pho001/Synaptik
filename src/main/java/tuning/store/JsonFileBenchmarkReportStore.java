package tuning.store;

import tuning.report.BenchmarkReport;
import tuning.report.BenchmarkSuiteReport;
import tuning.report.JsonBenchmarkReportRenderer;
import tuning.report.JsonBenchmarkSuiteReportRenderer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class JsonFileBenchmarkReportStore implements BenchmarkReportStore {
    @Override
    public void saveBenchmark(Path path, BenchmarkReport report) {
        write(path, JsonBenchmarkReportRenderer.render(report));
    }

    @Override
    public void saveSuite(Path path, BenchmarkSuiteReport report) {
        write(path, JsonBenchmarkSuiteReportRenderer.render(report));
    }

    private static void write(Path path, String json) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (json == null) {
            throw new IllegalArgumentException("json cannot be null");
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, json, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write report to " + path, e);
        }
    }
}
