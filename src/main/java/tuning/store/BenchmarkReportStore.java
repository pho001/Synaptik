package tuning.store;

import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;

import java.nio.file.Path;

/**
 * Persistence abstraction for benchmark reports.
 */
public interface BenchmarkReportStore {
    /**
     * Saves one workload benchmark report.
     *
     * @param path destination path
     * @param report benchmark report
     */
    void saveBenchmark(Path path, BenchmarkReport report);

    /**
     * Saves a benchmark suite report.
     *
     * @param path destination path
     * @param report suite report
     */
    void saveSuite(Path path, BenchmarkSuiteReport report);
}
