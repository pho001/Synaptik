package tuning.store;

import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;

import java.nio.file.Path;

public interface BenchmarkReportStore {
    void saveBenchmark(Path path, BenchmarkReport report);

    void saveSuite(Path path, BenchmarkSuiteReport report);
}
